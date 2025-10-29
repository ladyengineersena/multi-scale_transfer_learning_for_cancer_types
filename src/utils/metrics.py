"""
Evaluation metrics for classification
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch


def calculate_metrics(y_true, y_pred, y_proba=None):
    """Calculate comprehensive classification metrics"""
    metrics = {}
    
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_proba, torch.Tensor):
        y_proba = y_proba.cpu().numpy()
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
    
    metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None)
    metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None)
    metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None)
    
    if y_proba is not None:
        try:
            if y_proba.ndim == 1:
                metrics['auc'] = roc_auc_score(y_true, y_proba)
            else:
                metrics['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
        except:
            metrics['auc'] = None
    
    return metrics


def print_metrics(metrics, class_names=None):
    """Print metrics in a formatted way"""
    print('\n' + '='*50)
    print('MODEL PERFORMANCE METRICS')
    print('='*50)
    
    print(f'\nOverall Metrics:')
    print(f'  Accuracy:  {metrics["accuracy"]:.4f}')
    print(f'  Precision: {metrics["precision"]:.4f}')
    print(f'  Recall:    {metrics["recall"]:.4f}')
    print(f'  F1 Score:  {metrics["f1"]:.4f}')
    
    if 'auc' in metrics and metrics['auc'] is not None:
        print(f'  AUC:       {metrics["auc"]:.4f}')

