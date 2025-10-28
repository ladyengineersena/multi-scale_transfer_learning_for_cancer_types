"""
Multi-Scale Transfer Learning Model for Histopathological Images
Supports transfer learning between different cancer types with multi-scale feature extraction
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F


class MultiScaleFeatureExtractor(nn.Module):
    """
    Extracts features at multiple scales from histopathological images
    """
    def __init__(self, backbone_name='resnet50', pretrained=True):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        # Load backbone
        if backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif backbone_name == 'densenet121':
            backbone = models.densenet121(pretrained=pretrained)
            self.feature_dim = 1024
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Extract layers before final classification
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
    def forward(self, x):
        """
        Extract features from input image
        Args:
            x: Input image tensor [B, C, H, W]
        Returns:
            features: Extracted features [B, feature_dim]
        """
        features = self.backbone(x)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)
        return features


class DomainAdaptationModule(nn.Module):
    """
    Gradient Reversal Layer for domain adaptation
    """
    def __init__(self, input_dim, hidden_dim=256):
        super(DomainAdaptationModule, self).__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 2)  # Source/Target domain
        )
        
    def forward(self, features, alpha=1.0):
        """
        Domain classification with gradient reversal
        Args:
            features: Feature tensor [B, feature_dim]
            alpha: Gradient reversal coefficient
        Returns:
            domain_logits: Domain classification logits
            reversed_features: Features with reversed gradients
        """
        reversed_features = GradientReversal.apply(features, alpha)
        domain_logits = self.domain_classifier(reversed_features)
        return domain_logits, reversed_features


class GradientReversal(torch.autograd.Function):
    """Gradient reversal layer for domain adversarial training"""
    
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.alpha = alpha
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class MultiScaleTransferModel(nn.Module):
    """
    Main multi-scale transfer learning model for cross-cancer classification
    """
    def __init__(self, num_classes=2, backbone_name='resnet50', 
                 pretrained_source=True, use_domain_adapt=True):
        super(MultiScaleTransferModel, self).__init__()
        
        self.use_domain_adapt = use_domain_adapt
        
        # Multi-scale feature extractors
        self.extractor_high = MultiScaleFeatureExtractor(backbone_name, pretrained_source)
        self.extractor_mid = MultiScaleFeatureExtractor(backbone_name, pretrained_source)
        self.extractor_low = MultiScaleFeatureExtractor(backbone_name, pretrained_source)
        
        feature_dim = self.extractor_high.feature_dim
        
        # Feature fusion
        fused_dim = feature_dim * 3
        self.fusion_layer = nn.Sequential(
            nn.Linear(fused_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification head
        self.classifier = nn.Linear(feature_dim // 2, num_classes)
        
        # Domain adaptation
        if use_domain_adapt:
            self.domain_adapter = DomainAdaptationModule(feature_dim // 2)
    
    def forward(self, x_high, x_mid, x_low, alpha=1.0, return_features=False):
        """
        Forward pass with multi-scale inputs
        Args:
            x_high: High resolution images [B, C, 512, 512]
            x_mid: Mid resolution images [B, C, 256, 256]
            x_low: Low resolution images [B, C, 128, 128]
            alpha: Domain adaptation parameter
            return_features: Whether to return intermediate features
        Returns:
            class_logits: Classification logits
            domain_logits: Domain classification logits (if domain_adapt)
            fused_features: Fused multi-scale features
        """
        # Extract features at different scales
        feat_high = self.extractor_high(x_high)
        feat_mid = self.extractor_mid(x_mid)
        feat_low = self.extractor_low(x_low)
        
        # Concatenate multi-scale features
        fused_input = torch.cat([feat_high, feat_mid, feat_low], dim=1)
        
        # Fuse features
        fused_features = self.fusion_layer(fused_input)
        
        # Classification
        class_logits = self.classifier(fused_features)
        
        if return_features:
            return class_logits, fused_features
        
        # Domain adaptation
        if self.use_domain_adapt and self.training:
            domain_logits, _ = self.domain_adapter(fused_features, alpha)
            return class_logits, domain_logits
        
        return class_logits
    
    def forward_single_scale(self, x, scale='high'):
        """
        Forward pass with single scale input
        Args:
            x: Input image
            scale: Scale type ('high', 'mid', 'low')
        Returns:
            Class logits
        """
        if scale == 'high':
            features = self.extractor_high(x)
        elif scale == 'mid':
            features = self.extractor_mid(x)
        elif scale == 'low':
            features = self.extractor_low(x)
        else:
            raise ValueError(f"Unknown scale: {scale}")
        
        # Use simplified classifier for single scale
        logits = self.classifier(features)
        return logits


def get_model(num_classes=2, backbone='resnet50', pretrained=True, 
              use_domain_adapt=True):
    """
    Factory function to create model
    """
    model = MultiScaleTransferModel(
        num_classes=num_classes,
        backbone_name=backbone,
        pretrained_source=pretrained,
        use_domain_adapt=use_domain_adapt
    )
    return model

