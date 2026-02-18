"""Auto-configure a compatible PyTorch build for the current machine."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TorchTarget:
    """Target torch install plan."""

    label: str
    install_args: list[str]
    reason: str


def _run(cmd: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run command and capture text output."""
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def _detect_nvidia_smi_cuda() -> tuple[int, int] | None:
    """Return max CUDA version reported by nvidia-smi as (major, minor)."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = _run(["nvidia-smi"], check=True)
    except subprocess.CalledProcessError:
        return None

    match = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", result.stdout)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _choose_target(cuda_version: tuple[int, int] | None) -> TorchTarget:
    """Choose torch channel for this runtime.

    Notes:
    - For very old drivers (CUDA <= 11.4), modern Python 3.11-compatible CUDA wheels
      are not reliable, so we force CPU-only torch unless driver/toolkit is upgraded.
    """
    if cuda_version is None:
        return TorchTarget(
            label="cpu",
            install_args=[
                "uv",
                "pip",
                "install",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
                "torch==2.5.1",
            ],
            reason="No usable nvidia-smi/GPU detected.",
        )

    major, minor = cuda_version
    if major > 12 or (major == 12 and minor >= 1):
        return TorchTarget(
            label="cu121",
            install_args=[
                "uv",
                "pip",
                "install",
                "--index-url",
                "https://download.pytorch.org/whl/cu121",
                "torch==2.5.1",
            ],
            reason=f"Detected CUDA capability {major}.{minor}.",
        )

    if major == 11 and minor >= 8:
        return TorchTarget(
            label="cu118",
            install_args=[
                "uv",
                "pip",
                "install",
                "--index-url",
                "https://download.pytorch.org/whl/cu118",
                "torch==2.5.1",
            ],
            reason=f"Detected CUDA capability {major}.{minor}.",
        )

    return TorchTarget(
        label="cpu",
        install_args=[
            "uv",
            "pip",
            "install",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
            "torch==2.5.1",
        ],
        reason=f"Detected CUDA capability {major}.{minor}, which is too old for reliable Python 3.11 CUDA wheels.",
    )


def _apply_target(target: TorchTarget) -> None:
    """Install selected torch target."""
    print(f"[torch-config] target={target.label}")
    print(f"[torch-config] reason={target.reason}")
    subprocess.run(["uv", "pip", "uninstall", "torch"], check=False)
    _run(target.install_args, check=True)


def _verify() -> None:
    """Print runtime torch/CUDA summary."""
    code = (
        "import torch; "
        "print('torch', torch.__version__); "
        "print('torch_cuda', torch.version.cuda); "
        "print('cuda_available', torch.cuda.is_available()); "
        "print('cuda_devices', torch.cuda.device_count())"
    )
    result = _run(["uv", "run", "python", "-c", code], check=True)
    print(result.stdout.strip())


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Auto-configure a compatible PyTorch build.")
    parser.add_argument("--apply", action="store_true", help="Apply changes with uv pip install.")
    args = parser.parse_args()

    cuda_version = _detect_nvidia_smi_cuda()
    target = _choose_target(cuda_version)

    print(f"[torch-config] detected_cuda={cuda_version}")
    print(f"[torch-config] planned_target={target.label}")
    print(f"[torch-config] reason={target.reason}")

    if args.apply:
        _apply_target(target)
        _verify()
    else:
        print("[torch-config] dry-run mode. Re-run with --apply to enforce target.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
