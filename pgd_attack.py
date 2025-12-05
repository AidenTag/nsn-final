#PGD Adversarial Attack Evaluation for CIFAR-10 models
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import csv
from datetime import datetime
from torchvision import datasets, transforms
import numpy as np

#import model architectures
from wideresnet import wideresnet28_10, wideresnet28_2, wideresnet40_2, wideresnet16_8
from vit import vit32, vit56, vit110
from resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
from plainnet import plainnet20, plainnet32, plainnet44, plainnet56, plainnet110


def pgd_attack(model, images, labels, epsilon=8/255, alpha=2/255, num_iter=10, device='cpu'):
    #create a copy of images and set requires_grad
    adversarial_images = images.clone().detach().to(device).requires_grad_(True)
    labels = labels.to(device)
    
    for i in range(num_iter):
        outputs = model(adversarial_images)
        loss = F.cross_entropy(outputs, labels)
        
        model.zero_grad()
        if adversarial_images.grad is not None:
            adversarial_images.grad.zero_()
        
        loss.backward()
        
        data_grad = adversarial_images.grad.data
        sign_data_grad = data_grad.sign()
        
        adversarial_images = adversarial_images.detach() + alpha * sign_data_grad
        
        #clip
        perturbation = torch.clamp(adversarial_images - images, min=-epsilon, max=epsilon)
        adversarial_images = images + perturbation
        
        adversarial_images = torch.clamp(adversarial_images, min=-3.0, max=3.0)
        adversarial_images.requires_grad_(True)
    
    return adversarial_images.detach()


def evaluate_robustness(model, testloader, epsilon=8/255, alpha=2/255, num_iter=10, 
                        device='cpu', max_samples=1000):

    model.to(device)
    model.eval()
    
    clean_correct = 0
    robust_correct = 0
    total = 0
    
    print(f"  Evaluating on {max_samples} samples...")
    
    for i, (images, labels) in enumerate(testloader):
        if total >= max_samples:
            break
        
        batch_size = min(images.size(0), max_samples - total)
        images = images[:batch_size].to(device)
        labels = labels[:batch_size].to(device)
        
        #calculaing clean accuracy
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            clean_correct += (predicted == labels).sum().item()
        
        #generate adversarial examples
        adversarial_images = pgd_attack(model, images, labels, epsilon, alpha, num_iter, device)
        
        #calculating robust accuracy 
        with torch.no_grad():
            outputs = model(adversarial_images)
            _, predicted = torch.max(outputs, 1)
            robust_correct += (predicted == labels).sum().item()
        
        total += batch_size
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {total}/{max_samples} samples")
    
    clean_accuracy = 100.0 * clean_correct / total
    robust_accuracy = 100.0 * robust_correct / total
    
    return clean_accuracy, robust_accuracy


def parse_model_name_and_create(model_name):
    
    #determine if dropout was used
    dropout = 0.3 if 'dropout' in model_name else 0.0
    
    #remove dropout/nodropout suffix for parsing
    clean_name = model_name.replace('_dropout', '').replace('_nodropout', '')
    
    #WideResNet 
    if 'wideresnet' in clean_name:
        if 'wideresnet28_10' in clean_name:
            return wideresnet28_10(dropout_rate=dropout)
        elif 'wideresnet28_2' in clean_name:
            return wideresnet28_2(dropout_rate=dropout)
        elif 'wideresnet40_2' in clean_name:
            return wideresnet40_2(dropout_rate=dropout)
        elif 'wideresnet16_8' in clean_name:
            return wideresnet16_8(dropout_rate=dropout)
    
    #ViT 
    elif 'vit' in clean_name:
        if 'vit32' in clean_name:
            return vit32(dropout=dropout)
        elif 'vit56' in clean_name:
            return vit56(dropout=dropout)
        elif 'vit110' in clean_name:
            return vit110(dropout=dropout)
    
    #ResNet 
    elif 'resnet' in clean_name:
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
    
    #PlainNet 
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

#model is saved with prefix, so need to clean it up before deploying
def remove_module_prefix(state_dict):
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
    
    #load the dictionary of models 
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            if isinstance(checkpoint['model'], dict):
                state_dict = checkpoint['model']
            else:
                return checkpoint['model']
        else:
            state_dict = checkpoint
        
        #remove 'module.' prefix if present
        state_dict = remove_module_prefix(state_dict)
        model.load_state_dict(state_dict)
    else:
        model = checkpoint
    
    return model


def main():
    parser = argparse.ArgumentParser(description='PGD Adversarial Attack Evaluation on CIFAR-10')
    parser.add_argument('--results-dir', type=str, default='./results',
                      help='Directory containing model subdirectories.')
    parser.add_argument('--batch-size', type=int, default=128,
                      help='Batch size for evaluation.')
    parser.add_argument('--epsilon', type=float, default=8/255,
                      help='Maximum perturbation (default: 8/255 for CIFAR-10)')
    parser.add_argument('--alpha', type=float, default=2/255,
                      help='Step size for PGD (default: 2/255)')
    parser.add_argument('--num-iter', type=int, default=10,
                      help='Number of PGD iterations (default: 10)')
    parser.add_argument('--max-samples', type=int, default=1000,
                      help='Number of test samples to evaluate (default: 1000)')
    parser.add_argument('--output', type=str, default='pgd_results.csv',
                      help='Output CSV file for results.')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'\nPGD Attack Parameters:')
    print(f'  Epsilon (max perturbation): {args.epsilon:.4f} ({args.epsilon*255:.2f}/255)')
    print(f'  Alpha (step size): {args.alpha:.4f} ({args.alpha*255:.2f}/255)')
    print(f'  Number of iterations: {args.num_iter}')
    print(f'  Max samples per model: {args.max_samples}')
    print()
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    transform = transforms.Compose([
       transforms.ToTensor(),
       normalize,
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, 
                                             shuffle=False, pin_memory=True, num_workers=4)
    
    if not os.path.exists(args.results_dir):
        raise ValueError(f'Results directory {args.results_dir} does not exist')
    
    model_dirs = [d for d in os.listdir(args.results_dir) 
                  if os.path.isdir(os.path.join(args.results_dir, d))]
    
    if len(model_dirs) == 0:
        print(f'No model directories found in {args.results_dir}')
        return
    
    print(f'Found {len(model_dirs)} model(s) to evaluate\n')
    print('='*70)
    
    results = []
    
    #evaluate every model in result folder 
    for model_name in sorted(model_dirs):
        model_path = os.path.join(args.results_dir, model_name, 'model.th')
        
        if not os.path.exists(model_path):
            print(f'[{model_name}] No model.th found, skipping...')
            continue
        
        try:
            print(f'[{model_name}] Loading model...')
            model = load_model_from_checkpoint(model_path, model_name, device)
            
            print(f'[{model_name}] Running PGD attack...')
            clean_acc, robust_acc = evaluate_robustness(
                model, test_loader, args.epsilon, args.alpha, args.num_iter, 
                device, args.max_samples
            )
            
            attack_success_rate = clean_acc - robust_acc
            robustness_drop = (attack_success_rate / clean_acc * 100) if clean_acc > 0 else 0
            
            results.append({
                'model_name': model_name,
                'clean_accuracy': clean_acc,
                'robust_accuracy': robust_acc,
                'attack_success_rate': attack_success_rate,
                'robustness_drop_pct': robustness_drop,
                'epsilon': args.epsilon,
                'num_iter': args.num_iter,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            print(f'[{model_name}] Clean Acc: {clean_acc:.2f}% | Robust Acc: {robust_acc:.2f}% | '
                  f'Attack Success: {attack_success_rate:.2f}%')
            print('-'*70)
            
        except Exception as e:
            print(f'[{model_name}] Error: {str(e)}')
            print('-'*70)
            continue
    
    #result saved to csv (defualt baseline_results.csv)
    if results:
        with open(args.output, 'w', newline='') as f:
            fieldnames = ['model_name', 'clean_accuracy', 'robust_accuracy', 
                         'attack_success_rate', 'robustness_drop_pct', 
                         'epsilon', 'num_iter', 'timestamp']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f'\nResults saved to {args.output}')
    
    #summary 
    print('\n' + '='*70)
    print('SUMMARY (sorted by robust accuracy)')
    print('='*70)
    print(f'{"Model":<30} {"Clean Acc":<12} {"Robust Acc":<12} {"Attack Success":<15}')
    print('-'*70)
    for result in sorted(results, key=lambda x: x['robust_accuracy'], reverse=True):
        print(f'{result["model_name"]:<30} '
              f'{result["clean_accuracy"]:>10.2f}% '
              f'{result["robust_accuracy"]:>10.2f}% '
              f'{result["attack_success_rate"]:>13.2f}%')
    print('='*70)


if __name__ == '__main__':
    main()