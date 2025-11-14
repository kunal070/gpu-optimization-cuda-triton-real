# GPU Optimization with CUDA and Triton

A comprehensive deep learning optimization project exploring performance improvements through low-level GPU kernel programming using CUDA and Triton.

## 🎯 Project Goals

- Learn GPU architecture and parallel programming fundamentals
- Write efficient CUDA kernels for core neural network operations
- Use Triton to express high-level GPU kernels with fine-grained control
- Integrate and benchmark custom kernels within PyTorch
- Compare performance against PyTorch's built-in CUDA operators

## 🏗️ Project Structure

```
yash-luli/
├── cuda_kernels/          # CUDA C++ kernel implementations
│   ├── basic/             # Phase 1: Basic kernels (vector add, matmul)
│   ├── layernorm/         # Phase 2: LayerNorm kernel
│   ├── gelu/              # Phase 2: GELU activation kernel
│   └── fused/             # Phase 3: Fused operations
├── triton_kernels/        # Phase 4: Triton implementations
│   ├── layernorm.py
│   ├── gelu.py
│   └── fused.py
├── pytorch_extensions/    # Phase 5: PyTorch C++/CUDA extensions
│   ├── cuda_ops/          # CUDA extension module
│   └── setup.py
├── benchmarks/            # Benchmark scripts
│   └── benchmark.py
├── tests/                 # Unit tests
│   └── test_basic_kernels.py
├── examples/              # Usage examples
│   └── simple_example.py
├── docs/                  # Documentation
│   └── GPU_OPTIMIZATION_GUIDE.md
├── requirements.txt
├── setup.py
├── QUICKSTART.md          # Quick start guide
└── PROJECT_SUMMARY.md     # Detailed project summary
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- CUDA Toolkit 12.x
- PyTorch with CUDA support (cu121/cu124)
- NVIDIA GPU (Ampere or Ada recommended)
- Visual Studio Build Tools (MSVC v143) for Windows
- Ninja build system

### Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Build PyTorch extensions:
```bash
python setup.py build_ext --inplace
```

## 📚 Learning Phases

### Phase 1: Fundamentals
- GPU hardware understanding (threads, warps, SMs, memory hierarchy)
- Basic CUDA kernels (vector addition, matrix multiplication)
- Compilation and verification

### Phase 2: Custom CUDA Kernel Design
- Layer Normalization implementation
- GELU activation implementation
- Profiling and optimization

### Phase 3: Kernel Fusion
- Fused LayerNorm + GELU kernel
- Memory transfer reduction
- Performance analysis

### Phase 4: Triton Implementation
- Re-implement operations in Triton
- Compare with CUDA implementations
- Advanced Triton optimizations

### Phase 5: PyTorch Integration
- Create PyTorch C++/CUDA extensions
- Register custom operations
- Benchmark against native PyTorch

### Phase 6: Advanced Topics (Optional)
- Mixed-precision training (FP16/BF16)
- Tensor Cores usage
- Triton auto-tuning
- Multi-GPU scaling

## 🧪 Running Tests and Benchmarks

### Run Tests
```bash
python tests/test_basic_kernels.py
```

### Run Benchmarks
```bash
python benchmarks/benchmark.py
```

### Run Examples
```bash
python examples/simple_example.py
```

## 📖 Documentation

See `docs/` for detailed explanations of:
- GPU parallelization strategies
- Optimization techniques applied
- Performance analysis and results

## 🔧 Development

### Quick Start

See `QUICKSTART.md` for detailed installation and usage instructions.

### Building CUDA Kernels

CUDA kernels are compiled as part of the PyTorch extension build process:

```bash
python setup.py build_ext --inplace
```

### Testing

All kernels (CUDA and Triton) are tested together:

```bash
python tests/test_basic_kernels.py
```

## 📊 Expected Performance Gains

- LayerNorm: 1.5-2x speedup
- GELU: 1.2-1.5x speedup
- Fused LayerNorm+GELU: 2-3x speedup (vs separate ops)
- Matrix Multiplication: 1.3-2x speedup (with optimizations)

*Actual gains depend on input sizes, GPU architecture, and specific optimizations applied.*

