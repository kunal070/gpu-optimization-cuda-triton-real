"""
Workaround for msvcprt.lib issue by patching the build process
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import distutils.ccompiler
import distutils.sysconfig

# Patch distutils to use correct runtime library
def patch_distutils():
    """Patch distutils to avoid msvcprt.lib"""
    original_customize = distutils.sysconfig.customize_compiler
    
    def patched_customize(compiler):
        original_customize(compiler)
        # Remove any /MD flags that cause issues
        if hasattr(compiler, 'compiler'):
            compiler.compiler = [arg for arg in compiler.compiler if arg != '/MD']
        if hasattr(compiler, 'compiler_so'):
            compiler.compiler_so = [arg for arg in compiler.compiler_so if arg != '/MD']
        if hasattr(compiler, 'linker'):
            compiler.linker = [arg for arg in compiler.linker if arg != '/MD']
        if hasattr(compiler, 'linker_so'):
            compiler.linker_so = [arg for arg in compiler.linker_so if arg != '/MD']
    
    distutils.sysconfig.customize_compiler = patched_customize

patch_distutils()

# Get CUDA path
cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
if cuda_home is None:
    cuda_home = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"

print(f"Using CUDA from: {cuda_home}")

# Source files
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

# NVCC flags
nvcc_flags = [
    '-O3',
    '--use_fast_math',
    '-arch=sm_89',
    '--ptxas-options=-v',
    '--expt-relaxed-constexpr',
    '--expt-extended-lambda',
]

# Build extension
setup(
    name='gpu-optimization',
    ext_modules=[
        CUDAExtension(
            name='cuda_ops',
            sources=cuda_sources,
            include_dirs=include_dirs,
            extra_compile_args={
                'cxx': ['/O2', '/W3'],  # Windows-style optimization
                'nvcc': nvcc_flags,
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.10',
)