# Load select models from RobustBench for baseline evaluation on CIFAR-10
import torch
import os
import argparse
import csv
from datetime import datetime
from torchvision import datasets, transforms
import re

#importing model architectures
from wideresnet import wideresnet28_10, wideresnet28_2, wideresnet40_2, wideresnet16_8
from vit import vit32, vit56, vit110
from resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
from plainnet import plainnet20, plainnet32, plainnet44, plainnet56, plainnet110

def evaluate_model(model, dataloader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def parse_model_name_and_create(model_name):
    """Parse model directory name and create the corresponding model architecture"""
    
 
    dropout = 0.3 if 'dropout' in model_name else 0.0
    
    clean_name = model_name.replace('_dropout', '').replace('_nodropout', '')
    
    #WideResNet models
    if 'wideresnet' in clean_name:
        if 'wideresnet28_10' in clean_name:
            return wideresnet28_10(dropout_rate=dropout)
        elif 'wideresnet28_2' in clean_name:
            return wideresnet28_2(dropout_rate=dropout)
        elif 'wideresnet40_2' in clean_name:
            return wideresnet40_2(dropout_rate=dropout)
        elif 'wideresnet16_8' in clean_name:
            return wideresnet16_8(dropout_rate=dropout)
    
    #ViT models
    elif 'vit' in clean_name:
        if 'vit32' in clean_name:
            return vit32(dropout=dropout)
        elif 'vit56' in clean_name:
            return vit56(dropout=dropout)
        elif 'vit110' in clean_name:
            return vit110(dropout=dropout)
    
    #ResNet models
    elif 'resnet' in clean_name:
        # Extract just the number
        if 'resnet20' in clean_name:
            return resnet20(dropout=dropout)
        elif 'resnet32' in clean_name:
            return resnet32(dropout=dropout)
        elif 'resnet44' in clean_name:
            return resnet44(dropout=dropout)
        elif 'resnet56' in clean_name:
            return resnet56(dropout=dropout)
        elif 'resnet110' in clean_name:
            return resnet110(dropout=dropout)
        elif 'resnet1202' in clean_name:
            return resnet1202(dropout=dropout)
    
    #PlainNet models
    elif 'plainnet' in clean_name:
        if 'plainnet20' in clean_name:
            return plainnet20(dropout=dropout)
        elif 'plainnet32' in clean_name:
            return plainnet32(dropout=dropout)
        elif 'plainnet44' in clean_name:
            return plainnet44(dropout=dropout)
        elif 'plainnet56' in clean_name:
            return plainnet56(dropout=dropout)
        elif 'plainnet110' in clean_name:
            return plainnet110(dropout=dropout)
    
    raise ValueError(f"Could not parse model name: {model_name}")

#saved models have prefix, so clean before using 
def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from state_dict keys (from DataParallel)"""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

def load_model_from_checkpoint(model_path, model_name, device):
    """Load model from checkpoint and handle state_dict"""
    checkpoint = torch.load(model_path, map_location=device)
    
   
    model = parse_model_name_and_create(model_name)
    
    #load state dict
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            # Could be full model or state_dict
            if isinstance(checkpoint['model'], dict):
                state_dict = checkpoint['model']
            else:
                return checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present
        state_dict = remove_module_prefix(state_dict)
        model.load_state_dict(state_dict)
    else:
        model = checkpoint
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Baseline Model Evaluation')
    parser.add_argument('--results-dir', type=str, default='./results',
                      help='Directory containing model subdirectories.')
    parser.add_argument('--batch-size', type=int, default=128,
                      help='Batch size for evaluation.')
    parser.add_argument('--output', type=str, default='baseline_results.csv',
                      help='Output CSV file for results.')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    transform = transforms.Compose([
       transforms.ToTensor(),
       normalize,
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, 
                                             shuffle=False, pin_memory=True, num_workers=4)
    
    #directory where all models is saved
    if not os.path.exists(args.results_dir):
        raise ValueError(f'Results directory {args.results_dir} does not exist')
    
    model_dirs = [d for d in os.listdir(args.results_dir) 
                  if os.path.isdir(os.path.join(args.results_dir, d))]
    
    if len(model_dirs) == 0:
        print(f'No model directories found in {args.results_dir}')
        return
    
    print(f'Found {len(model_dirs)} model(s) to evaluate\n')
    print('='*60)
    
    results = []
    
    #Evaluate each model
    for model_name in sorted(model_dirs):
        model_path = os.path.join(args.results_dir, model_name, 'model.th')
        
        if not os.path.exists(model_path):
            print(f'[{model_name}] No model.th found, skipping...')
            continue
        
        try:
            print(f'[{model_name}] Loading model...')
            model = load_model_from_checkpoint(model_path, model_name, device)
            
            print(f'[{model_name}] Evaluating...')
            accuracy = evaluate_model(model, test_loader, device)
            results.append({
                'model_name': model_name,
                'accuracy': accuracy,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            print(f'[{model_name}] Test Accuracy: {accuracy:.2f}%')
            print('-'*60)
            
        except Exception as e:
            print(f'[{model_name}] Error: {str(e)}')
            print('-'*60)
            continue
    
    #Save results to CSV
    if results:
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['model_name', 'accuracy', 'timestamp'])
            writer.writeheader()
            writer.writerows(results)
        print(f'\nResults saved to {args.output}')
    
    #Print summary
    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f'{result["model_name"]:30s}: {result["accuracy"]:6.2f}%')
    print('='*60)

if __name__ == '__main__':
    main()