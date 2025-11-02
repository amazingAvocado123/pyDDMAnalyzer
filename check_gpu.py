#!/usr/bin/env python3
"""
GPU Diagnostics for pyDDMAnalyzer
Detects GPU hardware and checks driver/PyTorch configuration
"""

import subprocess
import sys

def run_command(cmd):
    """Run a shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return result.stdout.strip(), result.returncode
    except Exception as e:
        return f"Error: {e}", 1

print("=" * 60)
print("GPU DIAGNOSTICS")
print("=" * 60)

# 1. Check for NVIDIA GPU
print("\n1. CHECKING FOR NVIDIA GPU...")
output, code = run_command("lspci | grep -i nvidia")
if code == 0 and output:
    print(f"✓ NVIDIA GPU detected:")
    print(f"  {output}")
    
    # Check nvidia-smi
    output, code = run_command("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")
    if code == 0:
        print(f"✓ NVIDIA driver installed:")
        print(f"  {output}")
    else:
        print("✗ NVIDIA driver NOT detected (nvidia-smi not found)")
        print("  Install with: sudo apt install nvidia-driver")
else:
    print("✗ No NVIDIA GPU found")

# 2. Check for Intel GPU
print("\n2. CHECKING FOR INTEL GPU...")
output, code = run_command("lspci | grep -i vga")
if "Intel" in output:
    print(f"✓ Intel GPU detected:")
    print(f"  {output}")
    
    # Check for Intel compute runtime
    output, code = run_command("dpkg -l | grep intel-opencl")
    if code == 0 and output:
        print(f"✓ Intel OpenCL runtime detected")
    else:
        print("⚠ Intel compute runtime may not be installed")
        print("  For Intel Arc/Iris Xe, you may need intel-compute-runtime")
else:
    print("✗ No Intel GPU found")

# 3. Check for AMD GPU
print("\n3. CHECKING FOR AMD GPU...")
output, code = run_command("lspci | grep -i amd")
if "VGA" in output or "Display" in output:
    print(f"✓ AMD GPU detected:")
    print(f"  {output}")
    
    output, code = run_command("rocm-smi --showproductname 2>/dev/null")
    if code == 0:
        print(f"✓ ROCm installed")
    else:
        print("✗ ROCm NOT detected")
        print("  AMD GPU support requires ROCm: https://rocm.docs.amd.com/")
else:
    print("✗ No AMD GPU found")

# 4. Check PyTorch installation
print("\n4. CHECKING PYTORCH INSTALLATION...")
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    
    # Check CUDA support
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Check Intel XPU support
    try:
        print(f"  Intel XPU available: {torch.xpu.is_available()}")
        if torch.xpu.is_available():
            print(f"  XPU count: {torch.xpu.device_count()}")
    except AttributeError:
        print(f"  Intel XPU extension: NOT INSTALLED")
        print(f"    Install with: pip install intel-extension-for-pytorch")
    
except ImportError:
    print("✗ PyTorch NOT INSTALLED")
    print("  Install with: pip install torch")

# 5. Full GPU listing
print("\n5. ALL GRAPHICS DEVICES:")
output, code = run_command("lspci | grep -i 'vga\\|3d\\|display'")
if output:
    for line in output.split('\n'):
        print(f"  {line}")

print("\n" + "=" * 60)
print("RECOMMENDATIONS:")
print("=" * 60)

# Generate recommendations
output, _ = run_command("lspci | grep -i 'vga\\|3d\\|display'")

if "NVIDIA" in output:
    print("\n📌 NVIDIA GPU SETUP:")
    print("1. Install NVIDIA driver: sudo apt install nvidia-driver")
    print("2. Reboot")
    print("3. Install PyTorch with CUDA:")
    print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
elif "Intel" in output and ("Arc" in output or "Iris" in output or "Xe" in output):
    print("\n📌 INTEL GPU SETUP (Arc/Iris Xe):")
    print("1. Install Intel compute runtime:")
    print("   Follow: https://dgpu-docs.intel.com/driver/installation.html")
    print("2. Install PyTorch:")
    print("   pip3 install torch torchvision torchaudio")
    print("3. Install Intel extension:")
    print("   pip3 install intel-extension-for-pytorch")
    
elif "AMD" in output:
    print("\n📌 AMD GPU SETUP:")
    print("1. Install ROCm: https://rocm.docs.amd.com/")
    print("2. Install PyTorch with ROCm:")
    print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0")
    
else:
    print("\n📌 No dedicated GPU detected - CPU mode will be used")
    print("Install PyTorch: pip3 install torch torchvision torchaudio")

print("\n" + "=" * 60)