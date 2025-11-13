"""
Find MSVC libraries and diagnose the msvcprt.lib issue
"""
import os
from pathlib import Path

def find_msvc_libs():
    """Find all MSVC lib directories and their contents"""
    print("="*60)
    print("Searching for MSVC Libraries")
    print("="*60 + "\n")
    
    # Check the MSVC path from your error
    msvc_path = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.36.32532")
    
    if not msvc_path.exists():
        print(f"✗ MSVC path not found: {msvc_path}")
        # Try to find any MSVC version
        base_path = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC")
        if base_path.exists():
            versions = list(base_path.iterdir())
            print(f"\nAvailable MSVC versions:")
            for v in versions:
                print(f"  - {v.name}")
            if versions:
                msvc_path = versions[0]
                print(f"\nUsing: {msvc_path}")
    
    lib_path = msvc_path / "lib" / "x64"
    
    if lib_path.exists():
        print(f"\n✓ Found lib directory: {lib_path}\n")
        
        # List all .lib files
        lib_files = sorted(lib_path.glob("*.lib"))
        
        print(f"Found {len(lib_files)} library files:")
        
        # Check for specific runtime libraries
        runtime_libs = [
            "msvcrt.lib",      # Multithreaded DLL
            "msvcprt.lib",     # C++ runtime (older)
            "msvcprtd.lib",    # C++ runtime debug
            "libcmt.lib",      # Static runtime
            "libcpmt.lib",     # Static C++ runtime
        ]
        
        print("\nRuntime library status:")
        for lib_name in runtime_libs:
            lib_file = lib_path / lib_name
            if lib_file.exists():
                size = lib_file.stat().st_size / 1024
                print(f"  ✓ {lib_name:20s} ({size:.1f} KB)")
            else:
                print(f"  ✗ {lib_name:20s} NOT FOUND")
        
        print(f"\nAll libraries in directory:")
        for lib in lib_files[:20]:  # Show first 20
            print(f"  - {lib.name}")
        if len(lib_files) > 20:
            print(f"  ... and {len(lib_files) - 20} more")
    else:
        print(f"✗ Lib directory not found: {lib_path}")
    
    # Check Windows SDK
    print("\n" + "="*60)
    print("Checking Windows SDK Libraries")
    print("="*60 + "\n")
    
    sdk_path = Path(r"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0")
    
    if sdk_path.exists():
        for subdir in ['ucrt', 'um']:
            lib_dir = sdk_path / subdir / "x64"
            if lib_dir.exists():
                libs = list(lib_dir.glob("*.lib"))
                print(f"✓ {subdir}/x64: {len(libs)} libraries")

def check_link_exe():
    """Check the linker"""
    print("\n" + "="*60)
    print("Checking Linker")
    print("="*60 + "\n")
    
    link_exe = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.36.32532\bin\HostX64\x64\link.exe")
    
    if link_exe.exists():
        print(f"✓ Linker found: {link_exe}")
        size = link_exe.stat().st_size / 1024 / 1024
        print(f"  Size: {size:.1f} MB")
    else:
        print(f"✗ Linker not found: {link_exe}")

def suggest_fix():
    """Suggest how to fix the issue"""
    print("\n" + "="*60)
    print("SOLUTION")
    print("="*60 + "\n")
    
    print("The 'msvcprt.lib' is an OLD library name that doesn't exist in modern Visual Studio.")
    print("Modern VS uses different runtime libraries.\n")
    
    print("Option 1 (RECOMMENDED): Install Desktop C++ Development")
    print("-" * 50)
    print("1. Open 'Visual Studio Installer'")
    print("2. Click 'Modify' on Visual Studio Build Tools 2022")
    print("3. Check 'Desktop development with C++'")
    print("4. Install\n")
    
    print("Option 2: Try building without CUDA optimizations")
    print("-" * 50)
    print("Use PyTorch-only implementations (no compilation needed)\n")
    
    print("Option 3: Use WSL2 + Linux")
    print("-" * 50)
    print("Install Ubuntu in WSL2 and build there (easier!)\n")

if __name__ == '__main__':
    find_msvc_libs()
    check_link_exe()
    suggest_fix()