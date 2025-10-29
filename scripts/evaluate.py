"""
Evaluation script for multi-scale transfer learning model
"""

import argparse
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models.multi_scale_transfer import get_model
from src.data.preprocessing import HistoImageDataset
from src.data.augmentation import get_transforms
from src.utils.metrics import calculate_metrics, print_metrics
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Model')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test-data', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output-dir', type=str, default='results')
    return parser.parse_args()


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels, all_probas = [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            images_high = batch['image_high'].to(device)
            images_mid = batch['image_mid'].to(device)
            images_low = batch['image_low'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images_high, images_mid, images_low)
            class_logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            _, predicted = class_logits.max(1)
            probas = torch.softmax(class_logits, dim=1)
            
            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())
            all_probas.append(probas.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probas = torch.cat(all_probas).numpy()
    
    metrics = calculate_metrics(all_labels, all_preds, all_probas)
    return {'predictions': all_preds, 'labels': all_labels, 'metrics': metrics}


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {})
    
    model = get_model(num_classes=config.get('num_classes', 2),
                      backbone=config.get('backbone', 'resnet50'),
                      pretrained=False, use_domain_adapt=config.get('use_domain_adapt', True))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    test_transforms = get_transforms(mode='val', img_size=512)
    test_dataset = HistoImageDataset(args.test_data, mode='test', transforms=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    results = evaluate_model(model, test_loader, device)
    print_metrics(results['metrics'])
    
    import json
    with open(f'{args.output_dir}/results.json', 'w') as f:
        json.dump(results['metrics'], f, indent=2)


if __name__ == '__main__':
    main()

