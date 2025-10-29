"""
Training script for multi-scale transfer learning
"""

import argparse
import torch
import os
import sys
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.multi_scale_transfer import get_model
from src.data.preprocessing import get_data_loaders
from src.data.augmentation import get_transforms
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train Multi-Scale Transfer Learning Model')
    parser.add_argument('--source-domain', type=str, required=True, help='Source domain')
    parser.add_argument('--target-domain', type=str, required=True, help='Target domain')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    config = load_config(args.config)
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    
    data_root = Path('data')
    source_dir = data_root / args.source_domain
    target_dir = data_root / args.target_domain
    
    train_transforms = get_transforms(mode='train', img_size=512)
    
    print('Loading data...')
    source_train_loader, source_val_loader, target_train_loader, target_val_loader = \
        get_data_loaders(str(source_dir), str(target_dir), batch_size=config['batch_size'],
                         num_workers=4, source_transforms=train_transforms,
                         target_transforms=train_transforms)
    
    print('Creating model...')
    model = get_model(num_classes=config.get('num_classes', 2),
                      backbone=config.get('backbone', 'resnet50'),
                      pretrained=config.get('pretrained', True),
                      use_domain_adapt=config.get('use_domain_adapt', True))
    
    trainer = Trainer(model, device, config)
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
    
    save_dir = config.get('save_dir', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    history = trainer.train(source_train_loader, source_val_loader,
                           target_train_loader, target_val_loader,
                           epochs=config['epochs'], save_dir=save_dir)
    
    print(f'\nBest validation accuracy: {trainer.best_val_acc:.2f}%')


if __name__ == '__main__':
    main()

