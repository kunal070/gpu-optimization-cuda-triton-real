"""
CNN Model for MNIST with Custom CUDA/Triton Kernels

Location: models/cnn_mnist.py

This integrates your custom LayerNorm, GELU, Swish, and Loss functions
into a complete CNN for MNIST classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import custom kernels
try:
    import cuda_ops
    HAS_CUDA = True
    print("✅ Custom CUDA kernels loaded")
except ImportError:
    HAS_CUDA = False
    print("⚠️  Custom CUDA kernels not available")

try:
    from triton_kernels import (
        layernorm_triton, 
        gelu_triton, 
        swish_triton,
        layernorm_gelu_fused_triton
    )
    HAS_TRITON = True
    print("✅ Triton kernels loaded")
except ImportError:
    HAS_TRITON = False
    print("⚠️  Triton kernels not available")


class CustomLayerNorm(nn.Module):
    """
    Custom LayerNorm wrapper that uses your CUDA/Triton implementation
    Falls back to PyTorch if custom kernels unavailable
    """
    def __init__(self, normalized_shape, eps=1e-5, backend='cuda'):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.backend = backend
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x):
        """Apply LayerNorm using custom kernels or PyTorch fallback"""
        if self.backend == 'cuda' and HAS_CUDA:
            return cuda_ops.layernorm(x, self.gamma, self.beta, self.eps)
        elif self.backend == 'triton' and HAS_TRITON:
            return layernorm_triton(x, self.gamma, self.beta, self.eps)
        else:
            # PyTorch fallback
            return F.layer_norm(x, (self.normalized_shape,), 
                              self.gamma, self.beta, self.eps)


class CustomGELU(nn.Module):
    """Custom GELU activation using your implementations"""
    def __init__(self, use_fast=True, backend='cuda'):
        super().__init__()
        self.use_fast = use_fast
        self.backend = backend
    
    def forward(self, x):
        if self.backend == 'cuda' and HAS_CUDA:
            return cuda_ops.gelu(x, self.use_fast)
        elif self.backend == 'triton' and HAS_TRITON:
            return gelu_triton(x, self.use_fast)
        else:
            return F.gelu(x)


class CustomSwish(nn.Module):
    """Custom Swish activation using your implementations"""
    def __init__(self, backend='cuda'):
        super().__init__()
        self.backend = backend
    
    def forward(self, x):
        if self.backend == 'cuda' and HAS_CUDA:
            return cuda_ops.swish(x)
        elif self.backend == 'triton' and HAS_TRITON:
            return swish_triton(x)
        else:
            return x * torch.sigmoid(x)


class FusedLayerNormGELU(nn.Module):
    """Fused LayerNorm + GELU - demonstrates kernel fusion benefits"""
    def __init__(self, normalized_shape, eps=1e-5, backend='cuda'):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.backend = backend
        
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x):
        if self.backend == 'cuda' and HAS_CUDA:
            return cuda_ops.layernorm_gelu_fused(x, self.gamma, self.beta, self.eps)
        elif self.backend == 'triton' and HAS_TRITON:
            return layernorm_gelu_fused_triton(x, self.gamma, self.beta, self.eps)
        else:
            # Unfused fallback
            x = F.layer_norm(x, (self.normalized_shape,), self.gamma, self.beta, self.eps)
            return F.gelu(x)


class CNN_MNIST(nn.Module):
    """
    CNN for MNIST Classification
    
    Architecture:
        Conv1 (1->32) -> LayerNorm -> GELU -> MaxPool
        Conv2 (32->64) -> LayerNorm -> Swish -> MaxPool  
        FC1 (3136->256) -> Fused LayerNorm+GELU
        FC2 (256->10)
    
    Args:
        backend: 'cuda', 'triton', or 'pytorch'
        use_fusion: Whether to use fused LayerNorm+GELU
    """
    def __init__(self, backend='cuda', use_fusion=True):
        super().__init__()
        self.backend = backend
        self.use_fusion = use_fusion
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 14x14 -> 14x14
        self.pool = nn.MaxPool2d(2, 2)  # Reduces by 2x
        
        # Custom normalization layers
        self.norm1 = CustomLayerNorm(32, backend=backend)
        self.norm2 = CustomLayerNorm(64, backend=backend)
        
        # Custom activation layers
        self.gelu = CustomGELU(use_fast=True, backend=backend)
        self.swish = CustomSwish(backend=backend)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 256)  # 7x7 after two pooling layers
        
        if use_fusion:
            # Use fused LayerNorm + GELU for FC layer
            self.norm_fc = FusedLayerNormGELU(256, backend=backend)
        else:
            # Separate operations
            self.norm_fc = CustomLayerNorm(256, backend=backend)
        
        self.fc2 = nn.Linear(256, 10)
        
        print(f"\n{'='*60}")
        print(f"CNN Model Initialized")
        print(f"{'='*60}")
        print(f"Backend: {backend}")
        print(f"Kernel Fusion: {use_fusion}")
        print(f"CUDA Available: {HAS_CUDA}")
        print(f"Triton Available: {HAS_TRITON}")
        print(f"{'='*60}\n")
    
    def forward(self, x):
        # Block 1: Conv -> LayerNorm -> GELU -> Pool
        x = self.conv1(x)  # [B, 32, 28, 28]
        
        # LayerNorm expects [B, H, W, C] or [B, seq_len, features]
        # Permute: [B, C, H, W] -> [B, H, W, C]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, 28, 28, 32]
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # Back to [B, C, H, W]
        
        x = self.gelu(x)
        x = self.pool(x)  # [B, 32, 14, 14]
        
        # Block 2: Conv -> LayerNorm -> Swish -> Pool
        x = self.conv2(x)  # [B, 64, 14, 14]
        
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, 14, 14, 64]
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        x = self.swish(x)
        x = self.pool(x)  # [B, 64, 7, 7]
        
        # Flatten
        x = x.view(x.size(0), -1)  # [B, 3136]
        
        # FC Block: Linear -> Fused LayerNorm+GELU (or separate)
        x = self.fc1(x)  # [B, 256]
        
        if self.use_fusion:
            # Single fused operation
            x = self.norm_fc(x)
        else:
            # Separate operations
            x = self.norm_fc(x)
            x = self.gelu(x)
        
        # Output
        x = self.fc2(x)  # [B, 10]
        
        return x
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(backend='cuda', use_fusion=True, device='cuda'):
    """
    Factory function to create CNN model
    
    Args:
        backend: 'cuda', 'triton', or 'pytorch'
        use_fusion: Use fused LayerNorm+GELU
        device: 'cuda' or 'cpu'
    
    Returns:
        model: CNN_MNIST instance
    """
    model = CNN_MNIST(backend=backend, use_fusion=use_fusion)
    model = model.to(device)
    
    print(f"Model Parameters: {model.count_parameters():,}")
    print(f"Device: {device}\n")
    
    return model


if __name__ == '__main__':
    # Test model creation
    print("Testing CNN Model Creation...\n")
    
    # Test with different backends
    for backend in ['cuda', 'triton', 'pytorch']:
        try:
            print(f"\nTesting {backend} backend...")
            model = create_model(backend=backend, use_fusion=True, device='cpu')
            
            # Test forward pass
            x = torch.randn(2, 1, 28, 28)
            output = model(x)
            print(f"✅ {backend} backend works! Output shape: {output.shape}")
        except Exception as e:
            print(f"❌ {backend} backend failed: {e}")
    
    print("\n" + "="*60)
    print("Model testing complete!")
    print("="*60)