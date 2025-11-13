"""
Check Visual Studio installation and C++ tools
"""
import os
import sys
from pathlib import Path

def check_msvc():
    """Check MSVC installation"""
    print("Checking MSVC installation...")
    
    msvc_base = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC")
    
    if msvc_base.exists():
        versions = list(msvc_base.iterdir())
        print(f"✓ MSVC found: {len(versions)} version(s)")
        for v in versions:
            print(f"  - {v.name}")
            
            # Check for required libraries
            lib_path = v / "lib" / "x64"
            if lib_path.exists():
                libs = list(lib_path.glob("*.lib"))
                print(f"    Found {len(libs)} libraries")
                
                # Check for specific problematic library
                msvcprt = lib_path / "msvcprt.lib"
                msvcrt = lib_path / "msvcrt.lib"
                
                if msvcprt.exists():
                    print(f"    ✓ msvcprt.lib found")
                else:
                    print(f"    ✗ msvcprt.lib NOT found")
                
                if msvcrt.exists():
                    print(f"    ✓ msvcrt.lib found")
                else:
                    print(f"    ✗ msvcrt.lib NOT found")
            else:
                print(f"    ✗ lib/x64 directory not found")
    else:
        print("✗ MSVC not found")
        return False
    
    return True

def check_windows_sdk():
    """Check Windows SDK"""
    print("\nChecking Windows SDK...")
    
    sdk_base = Path(r"C:\Program Files (x86)\Windows Kits\10\Lib")
    
    if sdk_base.exists():
        versions = list(sdk_base.iterdir())
        print(f"✓ Windows SDK found: {len(versions)} version(s)")
        for v in versions:
            print(f"  - {v.name}")
            
            ucrt_path = v / "ucrt" / "x64"
            um_path = v / "um" / "x64"
            
            if ucrt_path.exists():
                print(f"    ✓ UCRT libraries found")
            else:
                print(f"    ✗ UCRT libraries NOT found")
            
            if um_path.exists():
                print(f"    ✓ UM libraries found")
            else:
                print(f"    ✗ UM libraries NOT found")
    else:
        print("✗ Windows SDK not found")
        return False
    
    return True

def check_cuda():
    """Check CUDA installation"""
    print("\nChecking CUDA...")
    
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        print(f"✓ CUDA_PATH: {cuda_path}")
        
        cuda_lib = Path(cuda_path) / "lib" / "x64"
        if cuda_lib.exists():
            libs = list(cuda_lib.glob("*.lib"))
            print(f"  Found {len(libs)} CUDA libraries")
        else:
            print("  ✗ CUDA lib/x64 not found")
    else:
        print("✗ CUDA_PATH not set")
        return False
    
    return True

def check_environment():
    """Check environment variables"""
    print("\nChecking environment variables...")
    
    important_vars = [
        'DISTUTILS_USE_SDK',
        'CUDA_PATH',
        'CUDA_HOME',
        'VSCMD_VER',
        'VCINSTALLDIR',
    ]
    
    for var in important_vars:
        value = os.environ.get(var)
        if value:
            print(f"✓ {var}: {value}")
        else:
            print(f"✗ {var}: not set")

def check_pytorch():
    """Check PyTorch installation"""
    print("\nChecking PyTorch...")
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  cuDNN version: {torch.backends.cudnn.version()}")
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    
    return True

def main():
    print("="*60)
    print("Visual Studio and Build Tools Diagnostic")
    print("="*60 + "\n")
    
    msvc_ok = check_msvc()
    sdk_ok = check_windows_sdk()
    cuda_ok = check_cuda()
    check_environment()
    pytorch_ok = check_pytorch()
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"MSVC:        {'✓ OK' if msvc_ok else '✗ MISSING'}")
    print(f"Windows SDK: {'✓ OK' if sdk_ok else '✗ MISSING'}")
    print(f"CUDA:        {'✓ OK' if cuda_ok else '✗ MISSING'}")
    print(f"PyTorch:     {'✓ OK' if pytorch_ok else '✗ MISSING'}")
    
    if not (msvc_ok and sdk_ok):
        print("\n⚠ RECOMMENDATION:")
        print("Install 'Desktop development with C++' workload in Visual Studio Installer")
        print("This includes all required C++ libraries and headers.")
    
    if msvc_ok and sdk_ok and cuda_ok and pytorch_ok:
        print("\n✓ All components installed correctly!")
        print("If build still fails, try Solution 4 below.")

if __name__ == '__main__':
    main()