#!/usr/bin/env python3
"""Capture package versions and GPU models without user/host identifiers."""
import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = ["torch", "numpy", "pandas", "scipy", "evo2", "vortex", "grelu", "enformer-pytorch", "pytorch-lightning"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        gpu = None
    record = {"python": platform.python_version(), "os": platform.system(), "architecture": platform.machine(),
              "package_versions": versions, "gpu_models_memory_driver": gpu,
              "note": "Current environment only; does not reconstruct historical job versions."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2)+"\n")
