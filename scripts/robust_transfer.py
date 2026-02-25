"""Resumable transfer of AlphaGenome weights to Citra and CSHL HPC."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RemoteHost:
    """Remote transfer target."""

    name: str
    ssh_target: str
    remote_parent: str


DEFAULT_SOURCE = "/Users/christen/Downloads/alphagenome-jax-all_folds-v1"
DEFAULT_KEY = "~/.ssh/id_ed25519_citra"
DEFAULT_HOSTS: tuple[RemoteHost, ...] = (
    RemoteHost(
        name="citra",
        ssh_target="trevor@143.48.59.3",
        remote_parent="~/alphagenome_weights",
    ),
    RemoteHost(
        name="hpc",
        ssh_target="christen@bamdev4.cshl.edu",
        remote_parent="/grid/wsbs/home_norepl/christen/alphagenome_weights",
    ),
)


def _run(cmd: list[str]) -> tuple[int, str]:
    """Run a command and return exit code and combined output."""
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout.strip()


def _ssh_base(key_path: str | None) -> list[str]:
    """Build common ssh options for resilient non-interactive transfer."""
    base = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
    if key_path:
        base.extend(["-i", key_path])
    return base


def _remote_ready(host: RemoteHost, ssh_base: list[str]) -> bool:
    """Ensure remote parent dir exists and ssh is reachable."""
    cmd = ssh_base + [host.ssh_target, f"mkdir -p {host.remote_parent} && echo OK"]
    code, out = _run(cmd)
    if code == 0 and "OK" in out:
        return True
    print(f"[{host.name}] SSH unavailable: {out}")
    return False


def _sync_host(source_dir: Path, host: RemoteHost, ssh_base: list[str]) -> bool:
    """Rsync source directory to a remote host."""
    if not _remote_ready(host, ssh_base):
        return False

    ssh_cmd = " ".join(ssh_base)
    rsync_cmd = [
        "rsync",
        "-az",
        "--partial",
        "--append-verify",
        "--timeout=30",
        "--info=progress2",
        "-e",
        ssh_cmd,
        str(source_dir),
        f"{host.ssh_target}:{host.remote_parent}/",
    ]
    code, out = _run(rsync_cmd)
    if code != 0:
        print(f"[{host.name}] rsync failed:\n{out}")
        return False

    remote_name = source_dir.name
    verify_cmd = ssh_base + [
        host.ssh_target,
        (
            "set -e; "
            f"test -d {host.remote_parent}/{remote_name}; "
            f"find {host.remote_parent}/{remote_name} -type f | wc -l; "
            f"du -sh {host.remote_parent}/{remote_name}"
        ),
    ]
    verify_code, verify_out = _run(verify_cmd)
    if verify_code == 0:
        print(f"[{host.name}] Transfer verified:\n{verify_out}")
        return True

    print(f"[{host.name}] Transfer completed but verify failed:\n{verify_out}")
    return False


def _parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE)
    parser.add_argument("--ssh-key", default=DEFAULT_KEY)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--sleep-seconds", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    """Run one-shot or looping transfer."""
    args = _parse_args()
    source_dir = Path(args.source_dir).expanduser().resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Source dir not found: {source_dir}")

    key_path = os.path.expanduser(args.ssh_key) if args.ssh_key else None
    if key_path and not os.path.exists(key_path):
        print(f"SSH key not found ({key_path}); proceeding with default SSH agent/config.")
        key_path = None
    ssh_base = _ssh_base(key_path)

    while True:
        results = [
            _sync_host(source_dir=source_dir, host=host, ssh_base=ssh_base)
            for host in DEFAULT_HOSTS
        ]
        if not args.loop:
            break
        if all(results):
            print("All transfers verified. Exiting.")
            break
        print(f"Sleeping {args.sleep_seconds}s before retry...")
        time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
