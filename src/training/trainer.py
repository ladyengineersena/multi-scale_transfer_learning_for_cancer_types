"""
Training loop for multi-scale transfer learning
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import os
from .losses import TransferLoss


class Trainer:
    """Trainer class for multi-scale transfer learning"""
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        self.optimizer = optim.AdamW(model.parameters(), lr=config.get('lr', 0.001),
                                     weight_decay=config.get('weight_decay', 0.0001))
        
        if config.get('scheduler') == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.get('epochs', 100),
                                               eta_min=config.get('min_lr', 1e-6))
        else:
            self.scheduler = StepLR(self.optimizer, step_size=config.get('lr_step', 30),
                                    gamma=config.get('lr_gamma', 0.1))
        
        class_weight = config.get('class_weight', None)
        if class_weight is not None:
            class_weight = torch.FloatTensor(class_weight).to(device)
        
        self.criterion = TransferLoss(class_weight=class_weight,
                                       lambda_domain=config.get('lambda_domain', 1.0),
                                       lambda_consistency=config.get('lambda_consistency', 0.1))
        
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.best_val_acc = 0.0
        self.best_model_path = None
        
    def train_epoch(self, source_loader, target_loader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        combined_loader = zip(source_loader, target_loader)
        pbar = tqdm(combined_loader, desc='Training')
        
        for (source_batch, target_batch) in pbar:
            source_high = source_batch['image_high'].to(self.device)
            source_mid = source_batch['image_mid'].to(self.device)
            source_low = source_batch['image_low'].to(self.device)
            source_labels = source_batch['label'].to(self.device)
            
            target_high = target_batch['image_high'].to(self.device)
            target_mid = target_batch['image_mid'].to(self.device)
            target_low = target_batch['image_low'].to(self.device)
            target_labels = target_batch['label'].to(self.device)
            
            source_domain_labels = torch.zeros(source_labels.size(0), dtype=torch.long).to(self.device)
            target_domain_labels = torch.ones(target_labels.size(0), dtype=torch.long).to(self.device)
            
            batch_high = torch.cat([source_high, target_high], dim=0)
            batch_mid = torch.cat([source_mid, target_mid], dim=0)
            batch_low = torch.cat([source_low, target_low], dim=0)
            batch_labels = torch.cat([source_labels, target_labels], dim=0)
            batch_domain = torch.cat([source_domain_labels, target_domain_labels], dim=0)
            
            self.optimizer.zero_grad()
            class_logits, domain_logits = self.model(batch_high, batch_mid, batch_low,
                                                     alpha=self.config.get('alpha', 1.0))
            
            loss, loss_dict = self.criterion(class_logits, batch_labels, domain_logits,
                                            batch_domain, alpha=self.config.get('alpha', 1.0))
            loss.backward()
            
            if self.config.get('clip_grad', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('clip_grad'))
            
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = class_logits.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        return total_loss / len(source_loader), 100. * correct / total
    
    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(val_loader, desc='Validation'):
            images_high = batch['image_high'].to(self.device)
            images_mid = batch['image_mid'].to(self.device)
            images_low = batch['image_low'].to(self.device)
            labels = batch['label'].to(self.device)
            
            outputs = self.model(images_high, images_mid, images_low)
            class_logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            loss = torch.nn.functional.cross_entropy(class_logits, labels)
            
            total_loss += loss.item()
            _, predicted = class_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(val_loader), 100. * correct / total
    
    def train(self, source_train_loader, source_val_loader, target_train_loader,
              target_val_loader, epochs, save_dir='checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            train_loss, train_acc = self.train_epoch(source_train_loader, target_train_loader)
            val_loss, val_acc = self.validate(source_val_loader)
            self.scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                model_path = os.path.join(save_dir, 'best_model.pth')
                self.best_model_path = model_path
                
                torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                           'optimizer_state_dict': self.optimizer.state_dict(),
                           'val_acc': val_acc, 'config': self.config}, model_path)
                print(f'Saved best model (Val Acc: {val_acc:.2f}%)')
        
        print(f'\nBest Val Accuracy: {self.best_val_acc:.2f}%')
        return self.history

