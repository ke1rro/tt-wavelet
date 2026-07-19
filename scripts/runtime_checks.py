"""Shared post-device validation checks."""

import re
import subprocess
import sys
from pathlib import Path


def parse_runtime_architecture(output: str) -> str:
    match = re.search(r"(?:lwt|ilwt)_architecture:\s*(\S+)", output)
    if match is None:
        raise RuntimeError("device output did not report its architecture")
    architecture = match.group(1).lower()
    if architecture in {"wormhole", "wormhole_b0"}:
        return "wormhole_b0"
    if architecture == "blackhole":
        return architecture
    raise RuntimeError(
        f"unsupported device architecture in runtime output: {architecture}"
    )


def check_consistent_architecture(
    observed_architecture: str | None, candidate_architecture: str
) -> str:
    if (
        observed_architecture is not None
        and observed_architecture != candidate_architecture
    ):
        raise RuntimeError(
            "hardware architecture changed during validation: "
            f"{observed_architecture} -> {candidate_architecture}"
        )
    return candidate_architecture


def run_ncrisc_elf_gate(project_root: Path, architecture: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "check_ncrisc_elf_size.py"),
            "--architecture",
            architecture,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    print(result.stdout, end="")
    if result.returncode != 0:
        raise RuntimeError(
            "NCRISC ELF-size gate failed:\n" + result.stdout + result.stderr
        )
