#!/usr/bin/env python3
"""Sync experiment result files from Citra/HPC into repo-local analysis storage."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HostSpec:
    name: str
    ssh_target: str
    remote_root: str


DEFAULT_HOSTS = {
    "citra": HostSpec("citra", "143.48.59.3", "$HOME/ALBench-S2F"),
    "hpc": HostSpec("hpc", "bamdev4.cshl.edu", "/grid/wsbs/home_norepl/christen/ALBench-S2F"),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def fetch_result_json(host: HostSpec, experiment: str, host_out_dir: Path) -> int:
    """Fetch result.json files for an experiment from one remote host."""
    host_out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = host_out_dir / "results.tgz"

    if host.name == "hpc":
        remote_cmd = (
            f"bash -lc 'cd {host.remote_root} && "
            f"if [ -d outputs/{experiment} ]; then "
            f'find outputs/{experiment} -type f -name "result.json" -print0 '
            "| tar --null -czf - --files-from -; "
            "else "
            "tar -czf - --files-from /dev/null; "
            "fi'"
        )
    else:
        remote_cmd = (
            f"cd {host.remote_root} && "
            f"if [ -d outputs/{experiment} ]; then "
            f"find outputs/{experiment} -type f -name 'result.json' -print0 "
            "| tar --null -czf - --files-from -; "
            "else "
            "tar -czf - --files-from /dev/null; "
            "fi"
        )

    with archive_path.open("wb") as f:
        proc = subprocess.Popen(["ssh", host.ssh_target, remote_cmd], stdout=f)
        if proc.wait() != 0:
            raise RuntimeError(f"Failed to fetch from {host.name}")

    extract_dir = host_out_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xzf", str(archive_path), "-C", str(extract_dir)], check=True)

    return len(list(extract_dir.rglob("result.json")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="exp0_yeast_scaling")
    parser.add_argument(
        "--hosts",
        nargs="+",
        default=["citra", "hpc"],
        choices=sorted(DEFAULT_HOSTS.keys()),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=repo_root() / "outputs" / "analysis" / "synced",
        help="Root directory for synced remote outputs.",
    )
    args = parser.parse_args()

    out_root = args.out_root / args.experiment
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Sync target: {out_root}")
    for host_name in args.hosts:
        host = DEFAULT_HOSTS[host_name]
        host_dir = out_root / host.name
        count = fetch_result_json(host, args.experiment, host_dir)
        print(f"{host.name}: synced {count} result.json files")


if __name__ == "__main__":
    main()
