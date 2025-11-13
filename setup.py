from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get CUDA path
cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
if cuda_home is None:
    raise RuntimeError("CUDA_HOME or CUDA_PATH environment variable must be set")

# Define source files
cuda_sources = [
    'pytorch_extensions/cuda_ops/cuda_ops.cpp',
    'pytorch_extensions/cuda_ops/cuda_ops_kernel.cu',
]

# Include directories
include_dirs = [
    'cuda_kernels',
    'pytorch_extensions/cuda_ops',
    os.path.join(cuda_home, 'include'),
]

# Compiler flags for C++
cxx_flags = [
    '-O3',
    '-std=c++17',
]

# Compiler flags for NVCC
nvcc_flags = [
    '-O3',
    '--use_fast_math',
    '-arch=sm_75',  # For RTX 20xx series
    '-arch=sm_80',  # For A100
    '-arch=sm_86',  # For RTX 30xx series
    '-arch=sm_89',  # For RTX 40xx series
    '--ptxas-options=-v',
    '--expt-relaxed-constexpr',
    '--expt-extended-lambda',
]

# Library directories
library_dirs = [
    os.path.join(cuda_home, 'lib', 'x64'),
]

# Libraries to link
libraries = ['cudart']

# Extra link arguments to fix msvcprt.lib issue
extra_link_args = []
if os.name == 'nt':  # Windows
    # Use /MD flag consistently and avoid msvcprt.lib
    extra_link_args = ['/MD']

setup(
    name='gpu-optimization',
    ext_modules=[
        CUDAExtension(
            name='cuda_ops',
            sources=cuda_sources,
            include_dirs=include_dirs,
            extra_compile_args={
                'cxx': cxx_flags,
                'nvcc': nvcc_flags,
            },
            library_dirs=library_dirs,
            libraries=libraries,
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.10',
)