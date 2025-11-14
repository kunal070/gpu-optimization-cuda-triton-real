"""
Training Script for CNN with Custom Kernels

Location: train_cnn.py (in project root)

This script trains the CNN on MNIST and compares performance
across different backends (CUDA, Triton, PyTorch).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import json
from datetime import datetime
import sys
import os

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from cnn_mnist import create_model


class CustomCrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss that can use your custom loss kernels
    For now, uses PyTorch's implementation
    TODO: Integrate your custom loss_functions.cu
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, output, target):
        return self.criterion(output, target)


def get_data_loaders(batch_size=64, num_workers=2):
    """Load MNIST dataset"""
    print("Loading MNIST dataset...")
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        './data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Download and load test data
    test_dataset = datasets.MNIST(
        './data', 
        train=False, 
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1000, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Test samples: {len(test_dataset)}")
    print(f"✅ Batch size: {batch_size}\n")
    
    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    batch_times = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_start = time.time()
        
        # Move to device
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Print progress
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} '
                  f'Acc: {100.*correct/total:.2f}% '
                  f'Time: {batch_time*1000:.2f}ms/batch')
    
    # Epoch statistics
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    avg_batch_time = sum(batch_times) / len(batch_times)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'avg_batch_time_ms': avg_batch_time * 1000,
        'samples_per_sec': len(train_loader.dataset) / sum(batch_times)
    }


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    
    test_loss = 0
    correct = 0
    total = 0
    
    inference_times = []
    
    with torch.no_grad():
        for data, target in test_loader:
            start = time.time()
            
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            inference_times.append(time.time() - start)
            
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'avg_inference_time_ms': avg_inference_time * 1000,
        'samples_per_sec': len(test_loader.dataset) / sum(inference_times)
    }


def train_and_evaluate(backend='cuda', use_fusion=True, num_epochs=5, 
                       batch_size=64, learning_rate=0.001):
    """
    Complete training and evaluation pipeline
    
    Args:
        backend: 'cuda', 'triton', or 'pytorch'
        use_fusion: Use fused LayerNorm+GELU
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
    """
    print("\n" + "="*80)
    print(f"TRAINING CNN - Backend: {backend}, Fusion: {use_fusion}")
    print("="*80 + "\n")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    
    # Create model
    model = create_model(backend=backend, use_fusion=use_fusion, device=device)
    
    # Loss and optimizer
    criterion = CustomCrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    results = {
        'backend': backend,
        'use_fusion': use_fusion,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'device': str(device),
        'epochs': []
    }
    
    print("\nStarting training...\n")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print('='*60)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, 
                                    device, epoch)
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        # Print results
        print(f"\nTrain Loss: {train_metrics['loss']:.4f}, "
              f"Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Test Loss: {test_metrics['loss']:.4f}, "
              f"Test Acc: {test_metrics['accuracy']:.2f}%")
        print(f"Train Speed: {train_metrics['samples_per_sec']:.0f} samples/sec")
        print(f"Inference Speed: {test_metrics['samples_per_sec']:.0f} samples/sec")
        
        # Save epoch results
        results['epochs'].append({
            'epoch': epoch,
            'train': train_metrics,
            'test': test_metrics
        })
    
    total_time = time.time() - start_time
    results['total_training_time'] = total_time
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Final Test Accuracy: {results['epochs'][-1]['test']['accuracy']:.2f}%")
    print("="*80 + "\n")
    
    return results, model


def compare_all_backends(num_epochs=5, batch_size=64):
    """Compare all backends: CUDA, Triton, PyTorch"""
    backends_to_test = ['cuda', 'triton', 'pytorch']
    all_results = {}
    
    print("\n" + "="*80)
    print("COMPARING ALL BACKENDS")
    print("="*80 + "\n")
    
    for backend in backends_to_test:
        print(f"\nTesting {backend.upper()} backend...\n")
        
        try:
            # Test with fusion
            results_fused, _ = train_and_evaluate(
                backend=backend,
                use_fusion=True,
                num_epochs=num_epochs,
                batch_size=batch_size
            )
            all_results[f'{backend}_fused'] = results_fused
            
            # Test without fusion
            results_unfused, _ = train_and_evaluate(
                backend=backend,
                use_fusion=False,
                num_epochs=num_epochs,
                batch_size=batch_size
            )
            all_results[f'{backend}_unfused'] = results_unfused
            
        except Exception as e:
            print(f"❌ {backend} backend failed: {e}\n")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cnn_comparison_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print(f"Results saved to: {filename}\n")
    
    # Print summary
    print_comparison_summary(all_results)
    
    return all_results


def print_comparison_summary(results):
    """Print summary comparing all backends"""
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80 + "\n")
    
    for name, result in results.items():
        if 'epochs' not in result or len(result['epochs']) == 0:
            continue
            
        final_epoch = result['epochs'][-1]
        print(f"{name}:")
        print(f"  Final Test Accuracy: {final_epoch['test']['accuracy']:.2f}%")
        print(f"  Training Speed: {final_epoch['train']['samples_per_sec']:.0f} samples/sec")
        print(f"  Inference Speed: {final_epoch['test']['samples_per_sec']:.0f} samples/sec")
        print(f"  Total Time: {result['total_training_time']:.2f}s")
        print()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CNN on MNIST')
    parser.add_argument('--backend', type=str, default='cuda',
                       choices=['cuda', 'triton', 'pytorch'],
                       help='Backend to use')
    parser.add_argument('--fusion', action='store_true',
                       help='Use kernel fusion')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--compare-all', action='store_true',
                       help='Compare all backends')
    
    args = parser.parse_args()
    
    if args.compare_all:
        compare_all_backends(num_epochs=args.epochs, batch_size=args.batch_size)
    else:
        results, model = train_and_evaluate(
            backend=args.backend,
            use_fusion=args.fusion,
            num_epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Save model
        torch.save(model.state_dict(), f'cnn_{args.backend}_epoch{args.epochs}.pth')
        print(f"Model saved to cnn_{args.backend}_epoch{args.epochs}.pth")


if __name__ == '__main__':
    main()