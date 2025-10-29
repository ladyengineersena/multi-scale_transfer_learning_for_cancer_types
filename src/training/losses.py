"""
Loss functions for multi-scale transfer learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransferLoss(nn.Module):
    """Combined loss for transfer learning with domain adaptation"""
    def __init__(self, class_weight=None, lambda_domain=1.0, lambda_consistency=0.1):
        super(TransferLoss, self).__init__()
        self.lambda_domain = lambda_domain
        self.lambda_consistency = lambda_consistency
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weight)
        self.domain_loss = nn.CrossEntropyLoss()
    
    def forward(self, class_logits, class_targets, domain_logits=None, domain_targets=None, alpha=1.0):
        cls_loss = self.ce_loss(class_logits, class_targets)
        loss_dict = {'classification': cls_loss.item()}
        total_loss = cls_loss
        
        if domain_logits is not None and domain_targets is not None:
            domain_loss = self.domain_loss(domain_logits, domain_targets) * alpha
            loss_dict['domain'] = domain_loss.item()
            total_loss = total_loss + self.lambda_domain * domain_loss
        
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict


class MultiScaleLoss(nn.Module):
    """Loss function for multi-scale features"""
    def __init__(self, class_weight=None, scale_weights=None):
        super(MultiScaleLoss, self).__init__()
        if scale_weights is None:
            scale_weights = {'high': 0.5, 'mid': 0.3, 'low': 0.2}
        self.scale_weights = scale_weights
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weight)
    
    def forward(self, logits_dict, targets):
        total_loss = 0.0
        for scale, logits in logits_dict.items():
            scale_loss = self.ce_loss(logits, targets)
            total_loss += self.scale_weights.get(scale, 1.0/len(logits_dict)) * scale_loss
        return total_loss

