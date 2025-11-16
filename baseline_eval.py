# Load select models from RobustBench for baseline evaluation on CIFAR-10

import torch
import argparse
from robustbench import load_model
from torchvision import datasets, transforms

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
   parser = argparse.ArgumentParser(description='Baseline Model Evaluation with RobustBench')
   parser.add_argument('--models', nargs='+', 
                     default=None,
                     help='List of RobustBench model names to evaluate')
   parser.add_argument('--batch-size', type=int, default=128,
                     help='Batch size for evaluation')
   parser.add_argument('--quick', action='store_true',
                     help='Quick mode with reduced dataset size')
   
   args = parser.parse_args()
   
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
   # Prepare CIFAR-10 test dataset (use standard 32x32 size for CIFAR-10 models)
   transform = transforms.Compose([
      transforms.ToTensor(),
   ])
   
   test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
   
   if args.quick:
      test_dataset.data = test_dataset.data[:1000]
      test_dataset.targets = test_dataset.targets[:1000]
   
   test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
   
   # RobustBench model names for CIFAR-10
   # See https://robustbench.github.io/ for full list
   available_models = {
      'Wang2023Better_WRN-28-10': 'Wang2023Better_WRN-28-10',
      'Pang2022Robustness_WRN34-10': 'Pang2022Robustness_WRN34-10', 
      'Rebuffi2021Fixing_70_16_cutmix_extra': 'Rebuffi2021Fixing_70_16_cutmix_extra',
      'Gowal2021Improving_70_16_ddpm_100m': 'Gowal2021Improving_70_16_ddpm_100m',
      'Cui2023Decoupled_WRN-28-10': 'Cui2023Decoupled_WRN-28-10',
   }
   
   # Default models if none specified
   if not args.models or args.models == ['resnet50', 'wide_resnet50_2', 'vision_transformer']:
      models_to_eval = ['Wang2023Better_WRN-28-10', 'Pang2022Robustness_WRN34-10']
   else:
      models_to_eval = args.models
   
   print(f"\nAvailable RobustBench models: {list(available_models.keys())}")
   print(f"Evaluating: {models_to_eval}\n")
   
   for model_name in models_to_eval:
      # Get full model name if short name provided
      full_model_name = available_models.get(model_name, model_name)
      
      try:
         print(f"\n{'='*60}")
         print(f"Loading model: {full_model_name}")
         print(f"{'='*60}")
         
         # Load robust model from RobustBench
         model = load_model(
            model_name=full_model_name,
            dataset='cifar10',
            threat_model='Linf'
         )
         
         print(f"Model loaded successfully!")
         
         # Evaluate
         accuracy = evaluate_model(model, test_loader, device)
         print(f"\nAccuracy of {full_model_name} on CIFAR-10 test set: {accuracy:.2f}%")
         
         # Clean up
         del model
         torch.cuda.empty_cache()
         
      except Exception as e:
         print(f"Error loading/evaluating {full_model_name}: {e}")
         print(f"Try one of these models: {list(available_models.keys())}")
         continue

if __name__ == '__main__':
      main()