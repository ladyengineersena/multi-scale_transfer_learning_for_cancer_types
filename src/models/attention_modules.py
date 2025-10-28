"""
Attention mechanisms for multi-scale feature fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleAttention(nn.Module):
    """
    Attention mechanism to weight multi-scale features
    """
    def __init__(self, feature_dim):
        super(MultiScaleAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 3),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, feat_high, feat_mid, feat_low):
        """
        Apply attention weights to multi-scale features
        """
        concat_features = torch.cat([feat_high, feat_mid, feat_low], dim=1)
        weights = self.attention(concat_features)
        
        weighted_high = feat_high * weights[:, 0:1]
        weighted_mid = feat_mid * weights[:, 1:2]
        weighted_low = feat_low * weights[:, 2:3]
        
        return torch.cat([weighted_high, weighted_mid, weighted_low], dim=1)


class SpatialAttention(nn.Module):
    """Spatial attention for feature maps"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_combined = torch.cat([avg_out, max_out], dim=1)
        x_att = self.conv(x_combined)
        x_att = self.sigmoid(x_att)
        return x * x_att


class ChannelAttention(nn.Module):
    """Channel attention for feature maps"""
    def __init__(self, num_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(num_channels // reduction, num_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        out = self.sigmoid(out).unsqueeze(2).unsqueeze(3)
        return x * out

