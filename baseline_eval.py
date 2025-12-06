# Load select models from RobustBench for baseline evaluation on CIFAR-10

import torch
import os
import argparse
from torchvision import datasets, transforms
import resnet
import plainnet
import wideresnet
import vit

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

def main():
   parser = argparse.ArgumentParser(description='Baseline Model Evaluation')
   parser.add_argument('--arch', '-a', type=str, required=True,
                     help='Model architecture (e.g., resnet20, plainnet32, vit32, wideresnet28_10)')
   parser.add_argument('--model-dir', type=str, default=None,
                     help='Directory for model checkpoint to evaluate.')
   parser.add_argument('--batch-size', type=int, default=128,
                     help='Batch size for evaluation.')
   parser.add_argument('--dropout', default=0.0, type=float,
                     help='Dropout rate (must match training config)')
   
   args = parser.parse_args()
   
   device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

   print(f"Using device: {device}")
   
   # Prepare CIFAR-10 test dataset (use standard 32x32 size for CIFAR-10 models)

   normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
   
   transform = transforms.Compose([
      transforms.ToTensor(),
      normalize,
   ])

   test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)
   
   test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
   
   # Create model architecture
   if args.arch.startswith('resnet'):
      model = resnet.__dict__[args.arch](dropout=args.dropout)
   elif args.arch.startswith('plainnet'):
      model = plainnet.__dict__[args.arch](dropout=args.dropout)
   elif args.arch.startswith('vit'):
      model = vit.__dict__[args.arch](dropout=args.dropout)
   elif args.arch.startswith('wideresnet'):
      model = wideresnet.__dict__[args.arch](dropout=args.dropout)
   else:
      raise ValueError(f"Unknown architecture: {args.arch}")
   
   # Load checkpoint (state_dict saved by train.py)
   model_ckpt = os.path.join('./results', args.model_dir, 'model.th')
   if os.path.exists(model_ckpt):
      checkpoint = torch.load(model_ckpt, map_location=device)
      # Handle both DataParallel and regular state dicts
      state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
      # Remove 'module.' prefix if present (from DataParallel)
      if list(state_dict.keys())[0].startswith('module.'):
         state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
      model.load_state_dict(state_dict)
      print(f"Loaded checkpoint from {model_ckpt}")
   else:
      raise ValueError(f'No valid checkpoint found at {model_ckpt}')
   
   accuracy = evaluate_model(model, test_loader, device)
   print(f'Test Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
      main()