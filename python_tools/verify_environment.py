#!/usr/bin/env python3
"""
Environment verification script for Semantic Sentinel.
Checks all required dependencies and their versions.
"""

import sys
import importlib
from typing import Tuple, Optional

def check_module(module_name: str, min_version: Optional[str] = None) -> Tuple[bool, str]:
    """Check if a module is installed and optionally verify minimum version."""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        if min_version and version != 'unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                return False, f"{module_name}: {version} (requires >= {min_version})"
        return True, f"{module_name}: {version}"
    except ImportError:
        return False, f"{module_name}: NOT INSTALLED"

def main():
    print("=" * 60)
    print("Semantic Sentinel - Environment Verification")
    print("=" * 60)
    print()

    # Python version check
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python: {py_version}")
    if sys.version_info < (3, 8):
        print("  WARNING: Python 3.8+ recommended")
    print()

    # Required packages
    packages = [
        ("torch", "2.0.0"),
        ("torchvision", "0.15.0"),
        ("clip", None),  # CLIP doesn't follow semver
        ("onnx", "1.14.0"),
        ("onnxruntime", "1.15.0"),
        ("cv2", "4.8.0"),  # opencv-python
        ("PIL", "10.0.0"),  # Pillow
        ("numpy", "1.24.0"),
        ("tqdm", "4.65.0"),
    ]

    all_ok = True
    print("Package Status:")
    print("-" * 40)

    for pkg_name, min_ver in packages:
        ok, msg = check_module(pkg_name, min_ver)
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} {msg}")
        if not ok:
            all_ok = False

    print()
    print("-" * 40)

    # CUDA availability check
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            device_name = torch.cuda.get_device_name(0)
            print(f"CUDA: Available (v{cuda_version})")
            print(f"GPU: {device_name}")
        else:
            print("CUDA: Not available (CPU mode)")
    except Exception as e:
        print(f"CUDA: Check failed ({e})")

    print()
    print("=" * 60)
    if all_ok:
        print("Environment verification: PASSED")
        return 0
    else:
        print("Environment verification: FAILED")
        print("Please install missing packages with:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())