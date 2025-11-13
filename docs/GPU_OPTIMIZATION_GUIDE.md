# GPU Optimization Guide

## Table of Contents
1. [GPU Architecture Fundamentals](#gpu-architecture-fundamentals)
2. [CUDA Programming Basics](#cuda-programming-basics)
3. [Optimization Techniques](#optimization-techniques)
4. [Triton Programming](#triton-programming)
5. [Performance Analysis](#performance-analysis)

## GPU Architecture Fundamentals

### Key Concepts

#### Thread Hierarchy
- **Thread**: Smallest unit of execution
- **Warp**: Group of 32 threads that execute in lockstep (SIMT - Single Instruction Multiple Threads)
- **Block**: Collection of threads that can share memory and synchronize
- **Grid**: Collection of blocks

#### Memory Hierarchy
1. **Registers**: Fastest, per-thread, limited quantity (~255 per thread)
2. **Shared Memory**: Fast, per-block, ~48KB per SM (Streaming Multiprocessor)
3. **Global Memory**: Slowest, accessible by all threads, large capacity
4. **L1/L2 Cache**: Automatic caching of global memory accesses

#### Streaming Multiprocessors (SMs)
- Each SM contains:
  - Warp schedulers
  - Register file
  - Shared memory
  - L1 cache
  - Special function units (SFUs)

### Memory Access Patterns

#### Coalesced Access
- Threads in a warp should access contiguous memory locations
- Example: `thread[i]` accesses `array[i]` (coalesced)
- Bad: `thread[i]` accesses `array[random[i]]` (not coalesced)

#### Bank Conflicts (Shared Memory)
- Shared memory is divided into banks (typically 32)
- Multiple threads accessing the same bank causes serialization
- Solution: Use padding or different access patterns

## CUDA Programming Basics

### Kernel Launch Syntax

```cuda
kernel_name<<<num_blocks, threads_per_block>>>(arguments);
```

- `num_blocks`: Number of blocks in the grid
- `threads_per_block`: Number of threads per block
- Maximum threads per block: 1024 (typically use 256 or 512)

### Thread Indexing

```cuda
// 1D indexing
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D indexing
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

### Synchronization

```cuda
__syncthreads();  // Synchronize all threads in a block
```

### Memory Types

```cuda
__global__ void kernel();     // Called from host, runs on device
__device__ void function();   // Called from device, runs on device
__host__ void function();     // Called from host, runs on host
```

## Optimization Techniques

### 1. Memory Coalescing

**Goal**: Minimize global memory transactions

**Technique**: Ensure threads in a warp access contiguous memory

```cuda
// Good: Coalesced access
__global__ void good_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;  // Contiguous access
    }
}

// Bad: Strided access
__global__ void bad_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx * 2] = data[idx * 2] * 2.0f;  // Strided access
    }
}
```

### 2. Shared Memory Usage

**Goal**: Reduce global memory traffic by caching frequently used data

**Example**: Tiled matrix multiplication

```cuda
__shared__ float tile_A[TILE_SIZE][TILE_SIZE];
__shared__ float tile_B[TILE_SIZE][TILE_SIZE];

// Load tile into shared memory
tile_A[threadIdx.y][threadIdx.x] = A[...];
__syncthreads();  // Wait for all threads to finish loading

// Compute using shared memory
float sum = 0.0f;
for (int k = 0; k < TILE_SIZE; k++) {
    sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
}
```

### 3. Warp-Level Primitives

**Goal**: Efficient reduction operations within a warp

```cuda
// Warp shuffle (no shared memory needed)
__device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

### 4. Loop Unrolling

**Goal**: Reduce loop overhead and enable better instruction scheduling

```cuda
// Compiler hint for unrolling
#pragma unroll
for (int i = 0; i < 4; i++) {
    // ...
}
```

### 5. Kernel Fusion

**Goal**: Combine multiple operations to reduce memory transfers

**Benefits**:
- Intermediate results stay in registers
- Reduced global memory traffic
- Lower kernel launch overhead

**Example**: Fused LayerNorm + GELU
```cuda
// Instead of:
// 1. LayerNorm kernel -> write to global memory
// 2. GELU kernel -> read from global memory

// Do:
// 1. LayerNorm computation
// 2. GELU computation (using register values)
// 3. Write final result to global memory (one write instead of two)
```

## Triton Programming

### Key Differences from CUDA

1. **Block-level programming**: Each program processes one block
2. **Automatic memory management**: No explicit memory allocation
3. **Python-like syntax**: More readable and maintainable
4. **Auto-tuning**: Built-in support for finding optimal configurations

### Basic Triton Kernel

```python
@triton.jit
def kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)  # Program ID (block ID)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute
    y = x * 2.0
    
    # Store
    tl.store(output_ptr + offsets, y, mask=mask)
```

### Auto-tuning

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def optimized_kernel(...):
    # ...
```

## Performance Analysis

### Profiling Tools

1. **Nsight Systems**: System-level profiling
   ```bash
   nsys profile python benchmark.py
   ```

2. **Nsight Compute**: Kernel-level analysis
   ```bash
   ncu --set full python benchmark.py
   ```

3. **PyTorch Profiler**: Built-in profiling
   ```python
   with torch.profiler.profile(
       activities=[torch.profiler.ProfilerActivity.CUDA]
   ) as prof:
       # Your code
   print(prof.key_averages().table())
   ```

### Key Metrics

- **Throughput**: Operations per second
- **Bandwidth**: Memory bandwidth utilization
- **Occupancy**: Ratio of active warps to maximum warps per SM
- **Register usage**: Per-thread register count
- **Shared memory usage**: Per-block shared memory

### Optimization Checklist

- [ ] Memory accesses are coalesced
- [ ] Shared memory is used effectively
- [ ] Bank conflicts are minimized
- [ ] Warp-level primitives are used for reductions
- [ ] Kernel fusion is applied where beneficial
- [ ] Block size is optimized (typically 256 or 512)
- [ ] Occupancy is reasonable (aim for >50%)
- [ ] Register usage is not excessive

## Common Pitfalls

1. **Divergent warps**: If-else statements can serialize execution
2. **Excessive shared memory**: Can limit occupancy
3. **Too many registers**: Can reduce occupancy
4. **Non-coalesced access**: Major performance killer
5. **Synchronization overhead**: Too many `__syncthreads()` calls
6. **Kernel launch overhead**: Many small kernels vs. fewer large kernels

## Best Practices

1. **Start simple**: Get correctness first, then optimize
2. **Profile before optimizing**: Measure to identify bottlenecks
3. **Use shared memory wisely**: Cache frequently accessed data
4. **Fuse operations**: Combine kernels when possible
5. **Test on different sizes**: Performance can vary significantly
6. **Consider Triton**: Often easier and performs well
7. **Document optimizations**: Explain why choices were made

## Resources

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Triton Documentation](https://triton-lang.org/)
- [PyTorch CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [NVIDIA Nsight Tools](https://developer.nvidia.com/nsight-tools)

