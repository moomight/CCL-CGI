# -*- coding: utf-8 -*-
"""
Run manifest generation for experiment reproducibility.
Captures Git state, configuration, environment, and hardware info.
"""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict


def get_git_info() -> Dict[str, Any]:
    """Get Git repository information."""
    git_info = {
        "commit_hash": None,
        "branch": None,
        "is_dirty": None,
        "remote_url": None,
    }

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            git_info["commit_hash"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            git_info["branch"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            git_info["is_dirty"] = bool(result.stdout.strip())

        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            git_info["remote_url"] = result.stdout.strip()

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    return git_info


def get_package_versions() -> Dict[str, str]:
    """Get versions of key packages."""
    versions = {}
    packages = {
        "torch": lambda: __import__("torch").__version__,
        "numpy": lambda: __import__("numpy").__version__,
        "scipy": lambda: __import__("scipy").__version__,
        "pandas": lambda: __import__("pandas").__version__,
        "h5py": lambda: __import__("h5py").__version__,
        "scikit-learn": lambda: __import__("sklearn").__version__,
        "lightning": lambda: __import__("lightning").__version__,
    }
    for name, getter in packages.items():
        try:
            versions[name] = getter()
        except Exception:
            pass
    return versions


def get_cuda_info() -> Dict[str, Any]:
    """Get CUDA and GPU information."""
    cuda_info = {
        "available": False,
        "version": None,
        "device_count": 0,
        "devices": [],
    }
    try:
        import torch

        cuda_info["available"] = torch.cuda.is_available()
        if cuda_info["available"]:
            cuda_info["version"] = torch.version.cuda
            cuda_info["device_count"] = torch.cuda.device_count()
            for i in range(cuda_info["device_count"]):
                props = torch.cuda.get_device_properties(i)
                cuda_info["devices"].append({
                    "id": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                })
    except Exception:
        pass
    return cuda_info


def get_system_info() -> Dict[str, str]:
    """Get system information."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def args_to_dict(args) -> Dict[str, Any]:
    """Convert argparse Namespace or SimpleNamespace to dict."""
    return vars(args) if hasattr(args, "__dict__") else dict(args)


def get_config_settings() -> Dict[str, Any]:
    """Get determinism settings from config.py."""
    settings = {}
    try:
        from config import CUDNN_DETERMINISTIC, CUDNN_BENCHMARK
        settings["cudnn_deterministic"] = CUDNN_DETERMINISTIC
        settings["cudnn_benchmark"] = CUDNN_BENCHMARK
    except Exception:
        pass
    return settings


def generate_run_manifest(args, output_dir: str = "history") -> str:
    """
    Generate a run manifest capturing all experiment metadata.

    Args:
        args: Command-line arguments (argparse.Namespace or SimpleNamespace)
        output_dir: Directory to save manifest files (default: "history")

    Returns:
        Path to the saved manifest file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "git": get_git_info(),
        "configuration": {
            "arguments": args_to_dict(args),
            "config_settings": get_config_settings(),
        },
        "environment": {
            "system": get_system_info(),
            "packages": get_package_versions(),
            "cuda": get_cuda_info(),
        },
    }

    os.makedirs(output_dir, exist_ok=True)

    model_name = getattr(args, "model_name", "CCL_CGI")
    dataset_name = getattr(args, "dataset_name", "CCL-CGI")
    filename = f"run_manifest_{model_name}_{dataset_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return filepath


def print_manifest_summary(manifest_path: str):
    """Print a brief summary of the saved manifest."""
    print("=" * 80)
    print("Run Manifest Generated")
    print("=" * 80)
    print(f"  Manifest saved to: {manifest_path}")

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        git = manifest.get("git", {})
        if git.get("commit_hash"):
            dirty_flag = " (dirty)" if git.get("is_dirty") else ""
            print(f"  Git commit: {git['commit_hash'][:8]}{dirty_flag}")

        cuda = manifest.get("environment", {}).get("cuda", {})
        if cuda.get("available"):
            print(f"  CUDA: {cuda.get('version', 'unknown')} ({cuda.get('device_count', 0)} device(s))")
        else:
            print("  CUDA: Not available (CPU mode)")

        print("=" * 80)
    except Exception:
        pass
