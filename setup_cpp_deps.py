#!/usr/bin/env python3
"""Download and set up C++ dependencies for Phase 3 build."""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path
import subprocess

DEPS_DIR = Path(__file__).parent / "deps"
ORT_VERSION = "1.23.2"
ORT_URL = f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/onnxruntime-win-x64-{ORT_VERSION}.zip"
JSON_URL = "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp"

MINGW_BIN = r"C:\Users\91891\mingw64\bin"
CMAKE_BIN = r"C:\Program Files\CMake\bin"


def download(url: str, dest: Path, desc: str):
    if dest.exists():
        print(f"  [skip] {desc} already at {dest.name}")
        return
    print(f"  Downloading {desc}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  -> {dest.name} ({dest.stat().st_size // 1024} KB)")


def main():
    print("=" * 60)
    print("Semantic Sentinel - C++ Dependency Setup")
    print("=" * 60)
    print()

    DEPS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. ONNX Runtime
    ort_zip = DEPS_DIR / f"onnxruntime-win-x64-{ORT_VERSION}.zip"
    ort_dir = DEPS_DIR / "onnxruntime"

    if not ort_dir.exists():
        download(ORT_URL, ort_zip, f"ONNX Runtime {ORT_VERSION}")
        print("  Extracting ONNX Runtime...")
        with zipfile.ZipFile(ort_zip, "r") as z:
            z.extractall(DEPS_DIR)
        # Rename extracted dir
        extracted = DEPS_DIR / f"onnxruntime-win-x64-{ORT_VERSION}"
        extracted.rename(ort_dir)
        print(f"  -> {ort_dir}")
    else:
        print(f"  [skip] ONNX Runtime already at {ort_dir}")

    # 2. nlohmann/json single header
    json_dir = DEPS_DIR / "json" / "nlohmann"
    json_hpp = json_dir / "json.hpp"
    download(JSON_URL, json_hpp, "nlohmann/json v3.11.3")

    # 3. Generate MinGW import library from ONNX Runtime DLL
    ort_dll = ort_dir / "lib" / "onnxruntime.dll"
    ort_imp_lib = ort_dir / "lib" / "libonnxruntime.a"

    if not ort_imp_lib.exists() and ort_dll.exists():
        print("  Generating MinGW import library from ORT DLL...")
        gendef_exe = Path(MINGW_BIN) / "gendef.exe"
        dlltool_exe = Path(MINGW_BIN) / "dlltool.exe"

        if not gendef_exe.exists() or not dlltool_exe.exists():
            print(f"  WARNING: gendef/dlltool not found in {MINGW_BIN}")
            print("  MinGW import lib not generated. Build may fail.")
        else:
            def_file = ort_dir / "lib" / "onnxruntime.def"
            subprocess.run(
                [str(gendef_exe), str(ort_dll)],
                check=True, cwd=str(ort_dir / "lib")
            )
            subprocess.run(
                [str(dlltool_exe), "-D", "onnxruntime.dll",
                 "-d", str(def_file), "-l", str(ort_imp_lib)],
                check=True
            )
            print(f"  -> {ort_imp_lib}")
    elif ort_imp_lib.exists():
        print(f"  [skip] MinGW import lib already at {ort_imp_lib.name}")

    print()
    print("Dependency paths:")
    print(f"  ORT headers:    {ort_dir / 'include'}")
    print(f"  ORT lib dir:    {ort_dir / 'lib'}")
    print(f"  ORT DLL:        {ort_dll}")
    print(f"  nlohmann/json:  {json_dir}")
    print()
    print("Done. Now run:")
    print(f"  cmake -G \"MinGW Makefiles\" -DCMAKE_BUILD_TYPE=Release -S . -B build")
    print(f"  cmake --build build")


if __name__ == "__main__":
    sys.exit(main())
