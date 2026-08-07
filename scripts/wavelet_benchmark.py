#!/usr/bin/env python3
# ruff: noqa: B023, BLE001, PYI034
"""Trustworthy precision, preflight, performance, and plotting for TT wavelets.

Tenstorrent ``device_time_ms`` values come from the TT-Metal device profiler's
complete program kernel span. Host dispatch and synchronization are retained as
separate diagnostics. PyWavelets values are CPU API wall-clock time.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import select
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pywt

ROOT = Path(__file__).resolve().parents[1]
WAVELET_DIR = ROOT / "wavelets"
STANDALONE_RUNNER = ROOT / "build" / "tt_wavelet_benchmark_runner"
SET_ENV = ROOT / "scripts" / "set_env.sh"
BOUNDARY_MODES = (
    "symmetric",
    "zero",
    "constant",
    "periodic",
    "antisymmetric",
    "smooth",
    "reflect",
    "antireflect",
)
TT_BACKENDS = ("tt-wavelet", "ttnn-wavelet")
KNOWN_NUMERICAL_RISK_SCHEMES = frozenset({"coif17"})
ALL_BACKENDS = ("pywavelets", *TT_BACKENDS)
DEVICE_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
TIMING_METHOD = "tt_metal_device_profiler_complete_program_kernel_span"
RUNNER_RESPONSE_TIMEOUT_SECONDS = float(
    os.environ.get("TT_WAVELET_RUNNER_TIMEOUT_SECONDS", "120")
)
RUNNER_HEARTBEAT_SECONDS = 10.0
PROGRESS_STREAM = os.fdopen(os.dup(sys.stderr.fileno()), "w", buffering=1)

PRECISION_FIELDS = (
    "case_id",
    "dimension",
    "transform",
    "wavelet",
    "boundary_mode",
    "category",
    "backend",
    "comparison",
    "band_or_output",
    "logical_input",
    "logical_output",
    "input_scale",
    "max_coeff",
    "abs_tol",
    "rel_tol",
    "max_abs",
    "mean_abs",
    "rmse",
    "pcc",
    "abs_passed",
    "pcc_passed",
    "passed",
    "status",
    "error_type",
    "error_message",
)

PERFORMANCE_FIELDS = (
    "case_id",
    "dimension",
    "transform",
    "wavelet",
    "boundary_mode",
    "category",
    "backend",
    "architecture",
    "device_model",
    "logical_input",
    "physical_input",
    "logical_output",
    "physical_output",
    "layout",
    "core_count",
    "planner_decision",
    "memory_config",
    "program_config_sizes_bytes",
    "program_config_max_bytes",
    "warmup_count",
    "repeat_count",
    "timing_mechanism",
    "primary_metric",
    "samples_ms",
    "median_ms",
    "min_ms",
    "mean_ms",
    "stddev_ms",
    "enqueue_or_dispatch_samples_ms",
    "enqueue_or_dispatch_median_ms",
    "sync_wait_samples_ms",
    "sync_wait_median_ms",
    "host_api_total_samples_ms",
    "host_api_total_median_ms",
    "status",
    "error_type",
    "error_message",
)


@dataclass(frozen=True)
class SchemeMetadata:
    name: str
    tap_size: int
    step_count: int
    coefficient_count: int
    max_abs_coeff: float
    max_radius: int
    complexity: float
    category: str


def _scheme_category(tap_size: int, step_count: int) -> str:
    # Preserve the repository's established tolerance buckets while deriving
    # them from checked-in factorization metadata instead of a copied name map.
    if tap_size <= 12 and step_count <= 9:
        return "compact"
    if 18 <= tap_size <= 24 and 13 <= step_count <= 15:
        return "medium"
    return "large-sensitive"


def load_scheme_metadata() -> dict[str, SchemeMetadata]:
    schemes: dict[str, SchemeMetadata] = {}
    for path in sorted(WAVELET_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        steps = payload["steps"]
        coefficients = [
            float(value) for step in steps for value in step.get("coefficients", [])
        ]
        max_radius = max(
            (
                max(
                    abs(int(step.get("shift", 0))),
                    abs(
                        int(step.get("shift", 0))
                        + len(step.get("coefficients", []))
                        - 1
                    ),
                )
                for step in steps
            ),
            default=0,
        )
        tap_size = int(payload["tap_size"])
        step_count = len(steps)
        coefficient_count = len(coefficients)
        max_abs_coeff = max((abs(value) for value in coefficients), default=1.0)
        # Factorization size dominates compilation/runtime complexity; the
        # logarithmic coefficient term distinguishes numerically sensitive peers.
        complexity = (
            float(tap_size)
            + 2.0 * step_count
            + 0.5 * coefficient_count
            + 2.0 * math.log10(max(1.0, max_abs_coeff))
            + max_radius
        )
        schemes[path.stem] = SchemeMetadata(
            name=path.stem,
            tap_size=tap_size,
            step_count=step_count,
            coefficient_count=coefficient_count,
            max_abs_coeff=max_abs_coeff,
            max_radius=max_radius,
            complexity=complexity,
            category=_scheme_category(tap_size, step_count),
        )
    if len(schemes) != 106:
        raise RuntimeError(
            f"Expected 106 wavelet schemes, found {len(schemes)} in {WAVELET_DIR}"
        )
    return schemes


def select_performance_wavelets(
    schemes: dict[str, SchemeMetadata], seed: int
) -> list[str]:
    chooser = random.Random(seed)
    chosen: list[str] = []
    for category in ("compact", "medium", "large-sensitive"):
        candidates = sorted(
            scheme.name
            for scheme in schemes.values()
            if scheme.category == category and scheme.name != "coif17"
        )
        if not candidates:
            raise RuntimeError(f"No eligible {category} wavelet schemes")
        chosen.append(chooser.choice(candidates))
    chosen.append("coif17")
    return chosen


def tolerance_for(scheme: SchemeMetadata) -> tuple[float, float]:
    scale = max(1.0, scheme.max_abs_coeff)
    if scheme.category == "compact":
        return 1.0e-4 * scale, 1.0e-5
    if scheme.category == "medium":
        return 1.0e-3 * scale, 1.0e-5
    return 5.0e-3 * scale, 1.0e-5


def stable_case_seed(seed: int, *parts: object) -> int:
    digest = hashlib.sha256(
        "|".join((str(seed), *(str(part) for part in parts))).encode()
    ).digest()
    return int.from_bytes(digest[:8], "little")


def signal_1d(length: int, seed: int, *parts: object) -> np.ndarray:
    index = np.arange(length, dtype=np.float64)
    rng = np.random.default_rng(stable_case_seed(seed, *parts, length, "1d"))
    values = (
        np.sin(index * 0.071)
        + 0.37 * np.cos(index * 0.017)
        + 1.0e-4 * index
        + rng.uniform(-0.02, 0.02, length)
    )
    return values.astype(np.float32)


def signal_2d(height: int, width: int, seed: int, *parts: object) -> np.ndarray:
    y = np.arange(height, dtype=np.float64).reshape(-1, 1)
    x = np.arange(width, dtype=np.float64).reshape(1, -1)
    rng = np.random.default_rng(stable_case_seed(seed, *parts, height, width, "2d"))
    values = (
        np.sin(x * 0.071)
        + 0.37 * np.cos(y * 0.047)
        + 2.0e-5 * x * y
        + rng.uniform(-0.02, 0.02, (height, width))
    )
    return values.astype(np.float32)


def run_git(*args: str, cwd: Path = ROOT) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def git_diff_digest(cwd: Path) -> str:
    result = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=cwd,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return "unknown"
    return hashlib.sha256(result.stdout).hexdigest()


def _json_configuration_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple | list):
        return [_json_configuration_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_configuration_value(item) for key, item in value.items()
        }
    return value


def benchmark_configuration(
    command: str,
    args: argparse.Namespace,
    selected_wavelets: Sequence[str],
    schemes: dict[str, SchemeMetadata],
) -> dict[str, Any]:
    excluded = {"command", "output_dir", "overwrite", "resume"}
    arguments = {
        key: _json_configuration_value(value)
        for key, value in sorted(vars(args).items())
        if key not in excluded
    }
    return {
        "command": command,
        "arguments": arguments,
        "selected_wavelets": list(selected_wavelets),
        "selected_wavelet_metadata": [
            asdict(schemes[name]) for name in selected_wavelets
        ],
        "repository_git_sha": run_git("rev-parse", "HEAD"),
        "repository_worktree_diff_sha256": git_diff_digest(ROOT),
        "tt_metal_git_sha": run_git("rev-parse", "HEAD", cwd=ROOT / "tt-metal"),
        "tt_metal_worktree_diff_sha256": git_diff_digest(ROOT / "tt-metal"),
        "planner_environment": {
            name: os.environ.get(name)
            for name in (
                "TT_WAVELET_L1_SIGNAL_BUDGET_BYTES",
                "TT_WAVELET_LWT_MAX_CORES",
                "TT_WAVELET_LWT_WORKSPACE_LAYOUT",
            )
        },
    }


def configuration_fingerprint(configuration: dict[str, Any]) -> str:
    encoded = json.dumps(
        configuration, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def create_metadata(
    command: str,
    args: argparse.Namespace,
    selected_wavelets: Sequence[str],
    schemes: dict[str, SchemeMetadata],
) -> dict[str, Any]:
    configuration = benchmark_configuration(command, args, selected_wavelets, schemes)
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": command,
        "repository_git_sha": run_git("rev-parse", "HEAD"),
        "repository_dirty": bool(run_git("status", "--porcelain")),
        "tt_metal_git_sha": run_git("rev-parse", "HEAD", cwd=ROOT / "tt-metal"),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pywavelets_version": pywt.__version__,
        "architecture": "detected-during-device-phase",
        "device_information": "detected-during-device-phase",
        "random_seed": args.seed,
        "selected_wavelets": list(selected_wavelets),
        "selected_wavelet_metadata": [
            asdict(schemes[name]) for name in selected_wavelets
        ],
        "boundary_modes": list(args.boundary_modes),
        "backends": list(args.backends),
        "warmup_count": getattr(args, "warmup_runs", None),
        "repeat_count": getattr(args, "repeats", None),
        "input_scale": getattr(args, "input_scale", None),
        "preflight_input_scale": getattr(args, "preflight_input_scale", None),
        "configuration": configuration,
        "configuration_fingerprint": configuration_fingerprint(configuration),
        "timing_methodology": {
            "tenstorrent_primary": TIMING_METHOD,
            "tenstorrent_definition": (
                "last kernel end minus first kernel start across all cores/RISCs for the complete operation program"
            ),
            "tenstorrent_diagnostics": [
                "enqueue_or_dispatch_ms",
                "sync_wait_ms",
                "host_api_total_ms",
            ],
            "pywavelets_primary": "cpu_api_time_ms",
            "excluded_from_tt_primary": [
                "process startup",
                "device open/close",
                "JIT compilation",
                "program construction/cache lookup",
                "H2D",
                "D2H",
                "host dispatch",
                "host synchronization wait",
            ],
        },
    }


def prepare_output_dir(path: Path, overwrite: bool, resume: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if overwrite:
            shutil.rmtree(path)
        elif not resume:
            raise RuntimeError(
                f"Output directory exists and is non-empty: {path}; use --resume or --overwrite"
            )
    path.mkdir(parents=True, exist_ok=True)


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def metadata_matches(path: Path, metadata: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        existing.get("configuration_fingerprint")
        == metadata["configuration_fingerprint"]
    )


def verify_or_write_metadata(
    path: Path, metadata: dict[str, Any], resume: bool
) -> None:
    if resume and path.exists():
        if not metadata_matches(path, metadata):
            raise RuntimeError(
                f"Dataset configuration differs from {path}; use --overwrite for a new run"
            )
        return
    write_metadata(path, metadata)


def update_device_metadata(
    path: Path, architecture: str, device_information: str
) -> None:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["architecture"] = architecture
    metadata["device_information"] = device_information
    write_metadata(path, metadata)


class CsvResults:
    def __init__(
        self,
        path: Path,
        fields: Sequence[str],
        resume: bool,
        *,
        require_passed_for_resume: bool = False,
    ):
        self.path = path
        self.fields = tuple(fields)
        self.completed: set[str] = set()
        self.precision_input_scale = 1.0
        self.require_passed_for_resume = require_passed_for_resume
        if resume and path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if (
                        row.get("status") == "ok"
                        and row.get("case_id")
                        and (
                            not require_passed_for_resume
                            or row.get("passed", "").lower() == "true"
                        )
                    ):
                        self.completed.add(row["case_id"])
        exists = path.exists() and path.stat().st_size > 0
        self.handle = path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.handle, fieldnames=self.fields, extrasaction="ignore"
        )
        if not exists:
            self.writer.writeheader()
            self.handle.flush()

    def write(self, row: dict[str, Any]) -> None:
        normalized = {field: row.get(field, "") for field in self.fields}
        for key, value in normalized.items():
            if isinstance(value, (list, tuple, dict)):
                normalized[key] = json.dumps(value, separators=(",", ":"))
            elif value is None:
                normalized[key] = ""
        self.writer.writerow(normalized)
        self.handle.flush()
        if (
            row.get("status") == "ok"
            and row.get("case_id")
            and (
                not self.require_passed_for_resume
                or str(row.get("passed", "")).lower() == "true"
            )
        ):
            self.completed.add(str(row["case_id"]))

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> CsvResults:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def classify_exception(error: BaseException) -> str:
    message = str(error).lower()
    if isinstance(error, TimeoutError):
        return "timeout_failure"
    if "program size" in message or "kernel config" in message or "compile" in message:
        return "compile_failure"
    if "profiler" in message or "duration" in message or "timing" in message:
        return "timing_failure"
    if isinstance(error, (ValueError, KeyError, TypeError)):
        return "validation_failure"
    return "runtime_failure"


def error_row(
    case_id: str, base: dict[str, Any], error: BaseException
) -> dict[str, Any]:
    return {
        **base,
        "case_id": case_id,
        "status": "error",
        "error_type": classify_exception(error),
        "error_message": str(error).replace("\n", "\\n"),
    }


def metric_values(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    actual_flat = np.asarray(actual, dtype=np.float64).reshape(-1)
    reference_flat = np.asarray(reference, dtype=np.float64).reshape(-1)
    if actual_flat.shape != reference_flat.shape:
        raise ValueError(
            f"Shape mismatch: actual={actual.shape}, reference={reference.shape}"
        )
    if not np.isfinite(actual_flat).all() or not np.isfinite(reference_flat).all():
        raise ValueError("Precision comparison contains NaN or infinity")
    difference = actual_flat - reference_flat
    max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
    mean_abs = float(np.mean(np.abs(difference))) if difference.size else 0.0
    rmse = float(np.sqrt(np.mean(difference * difference))) if difference.size else 0.0
    actual_centered = actual_flat - np.mean(actual_flat)
    reference_centered = reference_flat - np.mean(reference_flat)
    denominator = float(
        np.linalg.norm(actual_centered) * np.linalg.norm(reference_centered)
    )
    pcc = (
        float(np.dot(actual_centered, reference_centered) / denominator)
        if denominator > 1.0e-20
        else (1.0 if max_abs == 0 else 0.0)
    )
    return {"max_abs": max_abs, "mean_abs": mean_abs, "rmse": rmse, "pcc": pcc}


def precision_metric_row(
    *,
    case_id: str,
    dimension: int,
    transform: str,
    wavelet: str,
    mode: str,
    backend: str,
    comparison: str,
    band_or_output: str,
    logical_input: str,
    logical_output: str,
    scheme: SchemeMetadata,
    actual: np.ndarray,
    reference: np.ndarray,
    input_scale: float = 1.0,
) -> dict[str, Any]:
    abs_tol, rel_tol = tolerance_for(scheme)
    abs_tol *= input_scale
    metrics = metric_values(actual, reference)
    reference_scale = float(np.max(np.abs(reference))) if reference.size else 0.0
    effective_tol = abs_tol + rel_tol * reference_scale
    abs_passed = metrics["max_abs"] <= effective_tol
    pcc_passed = metrics["pcc"] >= 0.999
    return {
        "case_id": case_id,
        "dimension": f"{dimension}d",
        "transform": transform,
        "wavelet": wavelet,
        "boundary_mode": mode,
        "category": scheme.category,
        "backend": backend,
        "comparison": comparison,
        "band_or_output": band_or_output,
        "logical_input": logical_input,
        "logical_output": logical_output,
        "input_scale": input_scale,
        "max_coeff": scheme.max_abs_coeff,
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
        **metrics,
        "abs_passed": abs_passed,
        "pcc_passed": pcc_passed,
        # Preserve the repository's established acceptance rule: either the
        # factorization-scaled absolute tolerance or PCC may establish parity.
        "passed": abs_passed or pcc_passed,
        "status": "ok",
        "error_type": "",
        "error_message": "",
    }


def timing_statistics(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise RuntimeError("Timing produced no samples")
    finite = [float(value) for value in values]
    if any(not math.isfinite(value) or value <= 0.0 for value in finite):
        raise RuntimeError(f"Timing produced invalid samples: {finite}")
    return {
        "samples_ms": finite,
        "median_ms": statistics.median(finite),
        "min_ms": min(finite),
        "mean_ms": statistics.fmean(finite),
        "stddev_ms": statistics.pstdev(finite),
    }


class StandaloneSession(AbstractContextManager["StandaloneSession"]):
    def __init__(self, log_dir: Path):
        if not STANDALONE_RUNNER.is_file():
            raise RuntimeError(
                f"Standalone runner not built: {STANDALONE_RUNNER}; run cmake --build build --target tt_wavelet_benchmark_runner"
            )
        log_dir.mkdir(parents=True, exist_ok=True)
        self._stderr = (log_dir / "tt-wavelet.stderr.log").open("a", encoding="utf-8")
        self._stdout_noise = (log_dir / "tt-wavelet.stdout.log").open(
            "a", encoding="utf-8"
        )
        self._process = subprocess.Popen(
            [str(SET_ENV), str(STANDALONE_RUNNER), "-"],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            bufsize=1,
        )

    def _terminate_runner(self) -> None:
        if self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=10)

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("Standalone runner pipes are unavailable")
        if self._process.poll() is not None:
            raise RuntimeError(
                f"Standalone runner exited with code {self._process.returncode}"
            )
        self._process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self._process.stdin.flush()
        started = time.monotonic()
        deadline = started + RUNNER_RESPONSE_TIMEOUT_SECONDS
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self._terminate_runner()
                raise TimeoutError(
                    "Standalone runner produced no response within "
                    f"{RUNNER_RESPONSE_TIMEOUT_SECONDS:.0f} seconds for {request['case_id']}"
                )
            readable, _, _ = select.select(
                [self._process.stdout],
                [],
                [],
                min(RUNNER_HEARTBEAT_SECONDS, remaining),
            )
            if not readable:
                elapsed = time.monotonic() - started
                print(
                    f"[tt-wavelet active] {request['case_id']} "
                    f"elapsed={elapsed:.0f}s (compile or device execution)",
                    file=PROGRESS_STREAM,
                    flush=True,
                )
                if self._process.poll() is not None:
                    raise RuntimeError(
                        f"Standalone runner exited with code {self._process.returncode}"
                    )
                continue
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"Standalone runner ended while waiting for {request['case_id']}"
                )
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                self._stdout_noise.write(line)
                self._stdout_noise.flush()
                continue
            if response.get("case_id") != request["case_id"]:
                self._stdout_noise.write(line)
                self._stdout_noise.flush()
                continue
            return response

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._process.stdin is not None and not self._process.stdin.closed:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                pass
        if self._process.poll() is None:
            try:
                return_code = self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._terminate_runner()
                return_code = self._process.returncode
        else:
            return_code = self._process.returncode
        self._stderr.close()
        self._stdout_noise.close()
        if exc is None and return_code != 0:
            raise RuntimeError(f"Standalone runner exited with code {return_code}")


def configure_profiler_environment() -> None:
    os.environ.update(
        {
            "TT_METAL_DEVICE_PROFILER": "1",
            "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
            "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
            "TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY": "1",
            "TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES": "1",
            "TT_LOGGER_LEVEL": "FATAL",
        }
    )
    os.environ.pop("TT_METAL_SLOW_DISPATCH_MODE", None)


class TtnnSession(AbstractContextManager["TtnnSession"]):
    def __init__(self, log_dir: Path | None = None) -> None:
        configure_profiler_environment()
        self._saved_fds: tuple[int, int] | None = None
        self._log_files: tuple[Any, Any] | None = None
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            stdout_log = (log_dir / "ttnn.stdout.log").open("a", encoding="utf-8")
            stderr_log = (log_dir / "ttnn.stderr.log").open("a", encoding="utf-8")
            sys.stdout.flush()
            sys.stderr.flush()
            self._saved_fds = (os.dup(1), os.dup(2))
            self._log_files = (stdout_log, stderr_log)
            os.dup2(stdout_log.fileno(), 1)
            os.dup2(stderr_log.fileno(), 2)
        try:
            import torch
            import ttnn

            self.torch = torch
            self.ttnn = ttnn
            self.device = ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(1, 1), physical_device_ids=[0]
            )
            self.device.enable_program_cache()
            self.architecture = str(self.device.arch()).replace("Arch.", "").lower()
            self.device_information = repr(self.device)
        except BaseException:
            self._restore_logs()
            raise

    def _restore_logs(self) -> None:
        if self._saved_fds is None or self._log_files is None:
            return
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self._saved_fds[0], 1)
        os.dup2(self._saved_fds[1], 2)
        os.close(self._saved_fds[0])
        os.close(self._saved_fds[1])
        self._log_files[0].close()
        self._log_files[1].close()
        self._saved_fds = None
        self._log_files = None

    def to_device_1d(self, values: np.ndarray) -> Any:
        tensor = self.torch.from_numpy(
            np.ascontiguousarray(values.astype(np.float32, copy=False))
        )
        return self.ttnn.from_torch(
            tensor,
            dtype=self.ttnn.float32,
            layout=self.ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=self.ttnn.DRAM_MEMORY_CONFIG,
        )

    def to_device_2d(self, values: np.ndarray) -> Any:
        tensor = self.torch.from_numpy(
            np.ascontiguousarray(values.astype(np.float32, copy=False))
        )
        return self.ttnn.from_torch(
            tensor,
            dtype=self.ttnn.float32,
            layout=self.ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=self.ttnn.DRAM_MEMORY_CONFIG,
        )

    def to_numpy(self, tensor: Any, shape: tuple[int, ...]) -> np.ndarray:
        return (
            self.ttnn.to_torch(tensor)
            .float()
            .cpu()
            .numpy()
            .reshape(-1)[: math.prod(shape)]
            .reshape(shape)
        )

    def drain_profiler(self) -> None:
        self.ttnn.ReadDeviceProfiler(self.device)
        self.ttnn.profiler.get_latest_programs_perf_data()

    def _read_one_device_duration(self) -> float:
        self.ttnn.ReadDeviceProfiler(self.device)
        by_device = self.ttnn.profiler.get_latest_programs_perf_data()
        durations: list[float] = []
        for programs in by_device.values():
            for program in programs:
                analysis = program.program_analyses_results.get(DEVICE_DURATION_KEY)
                if analysis is not None:
                    durations.append(float(analysis.duration) / 1.0e6)
        if len(durations) != 1:
            raise RuntimeError(
                f"Device profiler returned {len(durations)} program durations, expected exactly one"
            )
        return durations[0]

    def measure(
        self,
        operation: Callable[[], Any],
        warmup_runs: int,
        repeats: int,
    ) -> dict[str, list[float]]:
        for _ in range(warmup_runs):
            operation()
            self.ttnn.synchronize_device(self.device)
        self.drain_profiler()
        device_times: list[float] = []
        dispatch_times: list[float] = []
        sync_times: list[float] = []
        host_times: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            operation()
            enqueued = time.perf_counter()
            self.ttnn.synchronize_device(self.device)
            synchronized = time.perf_counter()
            dispatch_times.append((enqueued - start) * 1.0e3)
            sync_times.append((synchronized - enqueued) * 1.0e3)
            host_times.append((synchronized - start) * 1.0e3)
            device_times.append(self._read_one_device_duration())
        return {
            "device_time_ms": device_times,
            "enqueue_or_dispatch_ms": dispatch_times,
            "sync_wait_ms": sync_times,
            "host_api_total_ms": host_times,
        }

    def __exit__(self, *_: object) -> None:
        try:
            self.ttnn.close_mesh_device(self.device)
        finally:
            self._restore_logs()


def ttnn_padded_shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.padded_shape)


def ttnn_padded_shapes(*tensors: Any) -> str:
    shapes = [ttnn_padded_shape(tensor) for tensor in tensors]
    return json.dumps(shapes[0] if len(shapes) == 1 else shapes, separators=(",", ":"))


def write_f32(path: Path, values: np.ndarray) -> None:
    np.ascontiguousarray(values, dtype=np.float32).tofile(path)


def read_f32(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    expected = math.prod(shape)
    if values.size != expected:
        raise RuntimeError(f"Expected {expected} floats in {path}, found {values.size}")
    return values.reshape(shape)


def pywt_bands_2d(
    values: np.ndarray, wavelet: str, mode: str
) -> tuple[np.ndarray, ...]:
    ll, (hl, lh, hh) = pywt.dwt2(values, wavelet, mode=mode)
    return tuple(np.asarray(band, dtype=np.float32) for band in (ll, lh, hl, hh))


def pywt_idwt_2d(
    bands: Sequence[np.ndarray], wavelet: str, mode: str, shape: tuple[int, int]
) -> np.ndarray:
    ll, lh, hl, hh = bands
    reconstructed = pywt.idwt2((ll, (hl, lh, hh)), wavelet, mode=mode)
    return np.asarray(reconstructed[: shape[0], : shape[1]], dtype=np.float32)


def progress(label: str, index: int, total: int, detail: str = "") -> None:
    suffix = f" {detail}" if detail else ""
    print(
        f"[{label}] {index}/{total}{suffix}",
        file=PROGRESS_STREAM,
        flush=True,
    )


def _precision_id(
    backend: str,
    wavelet: str,
    mode: str,
    dimension: int,
    transform: str,
    comparison: str,
    band: str,
) -> str:
    return "/".join(
        (backend, wavelet, mode, f"{dimension}d", transform, comparison, band)
    )


def _precision_base(
    dimension: int,
    transform: str,
    wavelet: str,
    mode: str,
    backend: str,
    comparison: str,
    band: str,
    scheme: SchemeMetadata,
    logical_input: str,
    logical_output: str,
    input_scale: float = 1.0,
) -> dict[str, Any]:
    abs_tol, rel_tol = tolerance_for(scheme)
    abs_tol *= input_scale
    return {
        "dimension": f"{dimension}d",
        "transform": transform,
        "wavelet": wavelet,
        "boundary_mode": mode,
        "category": scheme.category,
        "backend": backend,
        "comparison": comparison,
        "band_or_output": band,
        "logical_input": logical_input,
        "logical_output": logical_output,
        "input_scale": input_scale,
        "max_coeff": scheme.max_abs_coeff,
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
    }


def _write_precision_metric(
    results: CsvResults,
    *,
    dimension: int,
    transform: str,
    wavelet: str,
    mode: str,
    backend: str,
    comparison: str,
    band: str,
    scheme: SchemeMetadata,
    logical_input: str,
    logical_output: str,
    actual: np.ndarray,
    reference: np.ndarray,
) -> None:
    case_id = _precision_id(
        backend, wavelet, mode, dimension, transform, comparison, band
    )
    if case_id in results.completed:
        return
    results.write(
        precision_metric_row(
            case_id=case_id,
            dimension=dimension,
            transform=transform,
            wavelet=wavelet,
            mode=mode,
            backend=backend,
            comparison=comparison,
            band_or_output=band,
            logical_input=logical_input,
            logical_output=logical_output,
            scheme=scheme,
            actual=actual,
            reference=reference,
            input_scale=results.precision_input_scale,
        )
    )


def _write_precision_error(
    results: CsvResults,
    *,
    dimension: int,
    transform: str,
    wavelet: str,
    mode: str,
    backend: str,
    comparison: str,
    scheme: SchemeMetadata,
    logical_input: str,
    error: BaseException,
) -> None:
    case_id = _precision_id(
        backend, wavelet, mode, dimension, transform, comparison, "operation"
    )
    if case_id in results.completed:
        return
    base = _precision_base(
        dimension,
        transform,
        wavelet,
        mode,
        backend,
        comparison,
        "operation",
        scheme,
        logical_input,
        "",
        results.precision_input_scale,
    )
    results.write(error_row(case_id, base, error))


def _write_precision_case_errors(
    results: CsvResults,
    *,
    wavelet: str,
    mode: str,
    backend: str,
    scheme: SchemeMetadata,
    length: int,
    height: int,
    width: int,
    error: BaseException,
) -> None:
    for dimension, transform, comparison, logical_input in (
        (1, "lwt", "forward-vs-pywavelets", str(length)),
        (1, "ilwt", "inverse-reference-coefficients", str(length)),
        (2, "lwt_2d", "forward-vs-pywavelets", f"{height}x{width}"),
        (2, "ilwt_2d", "inverse-reference-coefficients", f"{height}x{width}"),
    ):
        _write_precision_error(
            results,
            dimension=dimension,
            transform=transform,
            wavelet=wavelet,
            mode=mode,
            backend=backend,
            comparison=comparison,
            scheme=scheme,
            logical_input=logical_input,
            error=error,
        )


def run_pywavelets_precision(
    args: argparse.Namespace,
    schemes: dict[str, SchemeMetadata],
    results: CsvResults,
) -> None:
    total = len(args.wavelets) * len(args.boundary_modes)
    for index, (wavelet, mode) in enumerate(
        ((wavelet, mode) for wavelet in args.wavelets for mode in args.boundary_modes),
        start=1,
    ):
        progress("precision pywavelets", index, total)
        scheme = schemes[wavelet]
        try:
            x1 = signal_1d(args.length, args.seed, wavelet, mode) * args.input_scale
            approximation, detail = (
                np.asarray(value, dtype=np.float32)
                for value in pywt.dwt(x1, wavelet, mode=mode)
            )
            reconstructed = np.asarray(
                pywt.idwt(approximation, detail, wavelet, mode=mode)[: args.length],
                dtype=np.float32,
            )
            shape = (args.height, args.width)
            x2 = signal_2d(*shape, args.seed, wavelet, mode) * args.input_scale
            bands = pywt_bands_2d(x2, wavelet, mode)
            reconstructed_2d = pywt_idwt_2d(bands, wavelet, mode, shape)
        except Exception as error:
            _write_precision_case_errors(
                results,
                wavelet=wavelet,
                mode=mode,
                backend="pywavelets",
                scheme=scheme,
                length=args.length,
                height=args.height,
                width=args.width,
                error=error,
            )
            continue
        for name, value in (("approximation", approximation), ("detail", detail)):
            _write_precision_metric(
                results,
                dimension=1,
                transform="lwt",
                wavelet=wavelet,
                mode=mode,
                backend="pywavelets",
                comparison="forward-vs-pywavelets",
                band=name,
                scheme=scheme,
                logical_input=str(args.length),
                logical_output=str(value.size),
                actual=value,
                reference=value,
            )
        _write_precision_metric(
            results,
            dimension=1,
            transform="ilwt",
            wavelet=wavelet,
            mode=mode,
            backend="pywavelets",
            comparison="inverse-reference-coefficients",
            band="reconstructed",
            scheme=scheme,
            logical_input=str(approximation.size),
            logical_output=str(args.length),
            actual=reconstructed,
            reference=x1,
        )
        _write_precision_metric(
            results,
            dimension=1,
            transform="ilwt",
            wavelet=wavelet,
            mode=mode,
            backend="pywavelets",
            comparison="roundtrip",
            band="reconstructed",
            scheme=scheme,
            logical_input=str(approximation.size),
            logical_output=str(args.length),
            actual=reconstructed,
            reference=x1,
        )

        for name, value in zip(("LL", "LH", "HL", "HH"), bands):
            _write_precision_metric(
                results,
                dimension=2,
                transform="lwt_2d",
                wavelet=wavelet,
                mode=mode,
                backend="pywavelets",
                comparison="forward-vs-pywavelets",
                band=name,
                scheme=scheme,
                logical_input=f"{shape[0]}x{shape[1]}",
                logical_output=f"{value.shape[0]}x{value.shape[1]}",
                actual=value,
                reference=value,
            )
        for comparison in ("inverse-reference-coefficients", "roundtrip"):
            _write_precision_metric(
                results,
                dimension=2,
                transform="ilwt_2d",
                wavelet=wavelet,
                mode=mode,
                backend="pywavelets",
                comparison=comparison,
                band="reconstructed",
                scheme=scheme,
                logical_input=f"{bands[0].shape[0]}x{bands[0].shape[1]}",
                logical_output=f"{shape[0]}x{shape[1]}",
                actual=reconstructed_2d,
                reference=x2,
            )


def _runner_error(response: dict[str, Any]) -> RuntimeError:
    return RuntimeError(
        f"{response.get('error_type', 'runtime_failure')}: {response.get('error_message', 'unknown runner error')}"
    )


def run_standalone_precision(
    args: argparse.Namespace,
    schemes: dict[str, SchemeMetadata],
    results: CsvResults,
    output_dir: Path,
    metadata_path: Path,
) -> None:
    total = len(args.wavelets) * len(args.boundary_modes)
    with tempfile.TemporaryDirectory(
        prefix="ttwv-precision-"
    ) as temporary_name, StandaloneSession(output_dir / "logs") as session:
        temporary = Path(temporary_name)
        architecture_recorded = False
        for index, (wavelet, mode) in enumerate(
            (
                (wavelet, mode)
                for wavelet in args.wavelets
                for mode in args.boundary_modes
            ),
            start=1,
        ):
            progress("precision tt-wavelet", index, total)
            scheme = schemes[wavelet]
            prefix = temporary / f"case-{index}"
            try:
                x1 = signal_1d(args.length, args.seed, wavelet, mode) * args.input_scale
                approximation_ref, detail_ref = (
                    np.asarray(value, dtype=np.float32)
                    for value in pywt.dwt(x1, wavelet, mode=mode)
                )
            except Exception as error:
                _write_precision_case_errors(
                    results,
                    wavelet=wavelet,
                    mode=mode,
                    backend="tt-wavelet",
                    scheme=scheme,
                    length=args.length,
                    height=args.height,
                    width=args.width,
                    error=error,
                )
                continue
            input_1d = prefix.with_suffix(".input1d.f32")
            approximation_path = prefix.with_suffix(".approximation-ref.f32")
            detail_path = prefix.with_suffix(".detail-ref.f32")
            write_f32(input_1d, x1)
            write_f32(approximation_path, approximation_ref)
            write_f32(detail_path, detail_ref)

            forward_prefix = Path(str(prefix) + ".forward1d")
            forward_request = {
                "case_id": f"runner/{wavelet}/{mode}/1d/lwt",
                "dimension": 1,
                "transform": "lwt",
                "wavelet": wavelet,
                "boundary_mode": mode,
                "length": args.length,
                "input_paths": [str(input_1d)],
                "capture_outputs": True,
                "output_prefix": str(forward_prefix),
                "warmup_runs": 0,
                "repeats": 1,
            }
            try:
                response = session.run(forward_request)
                if response.get("status") != "ok":
                    raise _runner_error(response)
                if not architecture_recorded:
                    update_device_metadata(
                        metadata_path,
                        str(response.get("architecture", "unknown")),
                        "standalone MeshDevice unit mesh device 0",
                    )
                    architecture_recorded = True
                for name, reference in (
                    ("approximation", approximation_ref),
                    ("detail", detail_ref),
                ):
                    actual = read_f32(
                        Path(f"{forward_prefix}.{name}.f32"), reference.shape
                    )
                    _write_precision_metric(
                        results,
                        dimension=1,
                        transform="lwt",
                        wavelet=wavelet,
                        mode=mode,
                        backend="tt-wavelet",
                        comparison="forward-vs-pywavelets",
                        band=name,
                        scheme=scheme,
                        logical_input=str(args.length),
                        logical_output=str(reference.size),
                        actual=actual,
                        reference=reference,
                    )
            except Exception as error:
                _write_precision_error(
                    results,
                    dimension=1,
                    transform="lwt",
                    wavelet=wavelet,
                    mode=mode,
                    backend="tt-wavelet",
                    comparison="forward-vs-pywavelets",
                    scheme=scheme,
                    logical_input=str(args.length),
                    error=error,
                )
            _run_standalone_precision_remainder(
                args=args,
                results=results,
                session=session,
                wavelet=wavelet,
                mode=mode,
                scheme=scheme,
                prefix=prefix,
                x1=x1,
                approximation_ref=approximation_ref,
                detail_ref=detail_ref,
                approximation_path=approximation_path,
                detail_path=detail_path,
                forward_prefix=forward_prefix,
            )


def _finish_precision_operation(session: TtnnSession) -> None:
    session.ttnn.synchronize_device(session.device)
    session.drain_profiler()


def run_ttnn_precision(
    args: argparse.Namespace,
    schemes: dict[str, SchemeMetadata],
    results: CsvResults,
    metadata_path: Path,
    log_dir: Path,
) -> None:
    total = len(args.wavelets) * len(args.boundary_modes)
    with TtnnSession(log_dir) as session:
        update_device_metadata(
            metadata_path, session.architecture, session.device_information
        )
        for index, (wavelet, mode) in enumerate(
            (
                (wavelet, mode)
                for wavelet in args.wavelets
                for mode in args.boundary_modes
            ),
            start=1,
        ):
            progress("precision ttnn-wavelet", index, total)
            scheme = schemes[wavelet]
            try:
                x1 = signal_1d(args.length, args.seed, wavelet, mode) * args.input_scale
                approximation_ref, detail_ref = (
                    np.asarray(value, dtype=np.float32)
                    for value in pywt.dwt(x1, wavelet, mode=mode)
                )
            except Exception as error:
                _write_precision_case_errors(
                    results,
                    wavelet=wavelet,
                    mode=mode,
                    backend="ttnn-wavelet",
                    scheme=scheme,
                    length=args.length,
                    height=args.height,
                    width=args.width,
                    error=error,
                )
                continue
            forward_outputs: tuple[Any, Any] | None = None
            try:
                input_tensor = session.to_device_1d(x1)
                forward_outputs = session.ttnn.dwt(
                    input_tensor, wavelet, boundary_mode=mode
                )
                _finish_precision_operation(session)
                for name, tensor, reference in zip(
                    ("approximation", "detail"),
                    forward_outputs,
                    (approximation_ref, detail_ref),
                ):
                    actual = session.to_numpy(tensor, reference.shape)
                    _write_precision_metric(
                        results,
                        dimension=1,
                        transform="lwt",
                        wavelet=wavelet,
                        mode=mode,
                        backend="ttnn-wavelet",
                        comparison="forward-vs-pywavelets",
                        band=name,
                        scheme=scheme,
                        logical_input=str(args.length),
                        logical_output=str(reference.size),
                        actual=actual,
                        reference=reference,
                    )
            except Exception as error:
                _write_precision_error(
                    results,
                    dimension=1,
                    transform="lwt",
                    wavelet=wavelet,
                    mode=mode,
                    backend="ttnn-wavelet",
                    comparison="forward-vs-pywavelets",
                    scheme=scheme,
                    logical_input=str(args.length),
                    error=error,
                )

            try:
                approximation_tensor = session.to_device_1d(approximation_ref)
                detail_tensor = session.to_device_1d(detail_ref)
                reconstructed = session.ttnn.idwt(
                    approximation_tensor,
                    detail_tensor,
                    wavelet,
                    args.length,
                    boundary_mode=mode,
                )
                _finish_precision_operation(session)
                actual = session.to_numpy(reconstructed, x1.shape)
                _write_precision_metric(
                    results,
                    dimension=1,
                    transform="ilwt",
                    wavelet=wavelet,
                    mode=mode,
                    backend="ttnn-wavelet",
                    comparison="inverse-reference-coefficients",
                    band="reconstructed",
                    scheme=scheme,
                    logical_input=str(approximation_ref.size),
                    logical_output=str(args.length),
                    actual=actual,
                    reference=x1,
                )
            except Exception as error:
                _write_precision_error(
                    results,
                    dimension=1,
                    transform="ilwt",
                    wavelet=wavelet,
                    mode=mode,
                    backend="ttnn-wavelet",
                    comparison="inverse-reference-coefficients",
                    scheme=scheme,
                    logical_input=str(approximation_ref.size),
                    error=error,
                )

            try:
                if forward_outputs is None:
                    raise RuntimeError(
                        "Roundtrip unavailable because forward operation failed"
                    )
                reconstructed = session.ttnn.idwt(
                    *forward_outputs,
                    wavelet,
                    args.length,
                    boundary_mode=mode,
                )
                _finish_precision_operation(session)
                actual = session.to_numpy(reconstructed, x1.shape)
                _write_precision_metric(
                    results,
                    dimension=1,
                    transform="ilwt",
                    wavelet=wavelet,
                    mode=mode,
                    backend="ttnn-wavelet",
                    comparison="roundtrip",
                    band="reconstructed",
                    scheme=scheme,
                    logical_input=str(approximation_ref.size),
                    logical_output=str(args.length),
                    actual=actual,
                    reference=x1,
                )
            except Exception as error:
                _write_precision_error(
                    results,
                    dimension=1,
                    transform="ilwt",
                    wavelet=wavelet,
                    mode=mode,
                    backend="ttnn-wavelet",
                    comparison="roundtrip",
                    scheme=scheme,
                    logical_input=str(approximation_ref.size),
                    error=error,
                )

            shape = (args.height, args.width)
            x2 = signal_2d(*shape, args.seed, wavelet, mode) * args.input_scale
            bands_ref = pywt_bands_2d(x2, wavelet, mode)
            forward_bands: tuple[Any, Any, Any, Any] | None = None
            try:
                input_tensor_2d = session.to_device_2d(x2)
                forward_bands = session.ttnn.dwt_2d(
                    input_tensor_2d, wavelet, boundary_mode=mode
                )
                _finish_precision_operation(session)
                for name, tensor, reference in zip(
                    ("LL", "LH", "HL", "HH"), forward_bands, bands_ref
                ):
                    actual = session.to_numpy(tensor, reference.shape)
                    _write_precision_metric(
                        results,
                        dimension=2,
                        transform="lwt_2d",
                        wavelet=wavelet,
                        mode=mode,
                        backend="ttnn-wavelet",
                        comparison="forward-vs-pywavelets",
                        band=name,
                        scheme=scheme,
                        logical_input=f"{shape[0]}x{shape[1]}",
                        logical_output=f"{reference.shape[0]}x{reference.shape[1]}",
                        actual=actual,
                        reference=reference,
                    )
            except Exception as error:
                _write_precision_error(
                    results,
                    dimension=2,
                    transform="lwt_2d",
                    wavelet=wavelet,
                    mode=mode,
                    backend="ttnn-wavelet",
                    comparison="forward-vs-pywavelets",
                    scheme=scheme,
                    logical_input=f"{shape[0]}x{shape[1]}",
                    error=error,
                )

            try:
                reference_band_tensors = tuple(
                    session.to_device_2d(band) for band in bands_ref
                )
                reconstructed_2d = session.ttnn.idwt_2d(
                    *reference_band_tensors,
                    wavelet,
                    shape,
                    boundary_mode=mode,
                )
                _finish_precision_operation(session)
                actual = session.to_numpy(reconstructed_2d, shape)
                _write_precision_metric(
                    results,
                    dimension=2,
                    transform="ilwt_2d",
                    wavelet=wavelet,
                    mode=mode,
                    backend="ttnn-wavelet",
                    comparison="inverse-reference-coefficients",
                    band="reconstructed",
                    scheme=scheme,
                    logical_input=f"{bands_ref[0].shape[0]}x{bands_ref[0].shape[1]}",
                    logical_output=f"{shape[0]}x{shape[1]}",
                    actual=actual,
                    reference=x2,
                )
            except Exception as error:
                _write_precision_error(
                    results,
                    dimension=2,
                    transform="ilwt_2d",
                    wavelet=wavelet,
                    mode=mode,
                    backend="ttnn-wavelet",
                    comparison="inverse-reference-coefficients",
                    scheme=scheme,
                    logical_input=f"{bands_ref[0].shape[0]}x{bands_ref[0].shape[1]}",
                    error=error,
                )

            _run_ttnn_precision_2d_roundtrip(
                session=session,
                forward_bands=forward_bands,
                wavelet=wavelet,
                shape=shape,
                mode=mode,
                results=results,
                scheme=scheme,
                bands_ref=bands_ref,
                reference=x2,
            )


@dataclass
class TtnnPerformanceCase:
    logical_input: str
    outputs: tuple[Any, ...]
    reconstructed: Any
    layout: str
    operations: list[tuple[str, Callable[[], Any], str, str]]


def prepare_ttnn_performance_case(
    session: TtnnSession,
    args: argparse.Namespace,
    wavelet: str,
    mode: str,
    dimension: int,
    logical: int | tuple[int, int],
) -> TtnnPerformanceCase:
    if dimension == 1:
        length = int(logical)
        values = signal_1d(length, args.seed, wavelet, mode, "performance")
        input_tensor = session.to_device_1d(values)
        outputs = tuple(session.ttnn.dwt(input_tensor, wavelet, boundary_mode=mode))
        reconstructed = session.ttnn.idwt(*outputs, wavelet, length, boundary_mode=mode)
        session.ttnn.synchronize_device(session.device)
        session.drain_profiler()
        coefficient_length = pywt.dwt_coeff_len(
            length, pywt.Wavelet(wavelet).dec_len, mode
        )
        return TtnnPerformanceCase(
            logical_input=str(length),
            outputs=outputs,
            reconstructed=reconstructed,
            layout="row-major",
            operations=[
                (
                    "lwt",
                    lambda: session.ttnn.dwt(
                        input_tensor,
                        wavelet,
                        boundary_mode=mode,
                        output_tensors=outputs,
                    ),
                    ttnn_padded_shapes(input_tensor),
                    str(coefficient_length),
                ),
                (
                    "ilwt",
                    lambda: session.ttnn.idwt(
                        *outputs,
                        wavelet,
                        length,
                        boundary_mode=mode,
                        output_tensor=reconstructed,
                    ),
                    ttnn_padded_shapes(*outputs),
                    str(length),
                ),
            ],
        )

    height, width = logical
    values = signal_2d(height, width, args.seed, wavelet, mode, "performance")
    input_tensor = session.to_device_2d(values)
    outputs = tuple(session.ttnn.dwt_2d(input_tensor, wavelet, boundary_mode=mode))
    reconstructed = session.ttnn.idwt_2d(
        *outputs, wavelet, (height, width), boundary_mode=mode
    )
    session.ttnn.synchronize_device(session.device)
    session.drain_profiler()
    band_shape = tuple(outputs[0].shape)
    logical_input = f"{height}x{width}"
    return TtnnPerformanceCase(
        logical_input=logical_input,
        outputs=outputs,
        reconstructed=reconstructed,
        layout="tile",
        operations=[
            (
                "lwt_2d",
                lambda: session.ttnn.dwt_2d(
                    input_tensor,
                    wavelet,
                    boundary_mode=mode,
                    output_tensors=outputs,
                ),
                ttnn_padded_shapes(input_tensor),
                f"{band_shape[-2]}x{band_shape[-1]}",
            ),
            (
                "ilwt_2d",
                lambda: session.ttnn.idwt_2d(
                    *outputs,
                    wavelet,
                    (height, width),
                    boundary_mode=mode,
                    output_tensor=reconstructed,
                ),
                ttnn_padded_shapes(*outputs),
                logical_input,
            ),
        ],
    )


def _write_ttnn_setup_errors(
    results: CsvResults,
    *,
    dimension: int,
    wavelet: str,
    mode: str,
    scheme: SchemeMetadata,
    logical: int | tuple[int, int],
    architecture: str,
    error: BaseException,
) -> None:
    logical_input = str(logical) if dimension == 1 else f"{logical[0]}x{logical[1]}"
    transforms = ("lwt", "ilwt") if dimension == 1 else ("lwt_2d", "ilwt_2d")
    for transform in transforms:
        case_id = _performance_id(
            "ttnn-wavelet", wavelet, mode, dimension, transform, logical_input
        )
        if case_id in results.completed:
            continue
        results.write(
            error_row(
                case_id,
                {
                    "dimension": f"{dimension}d",
                    "transform": transform,
                    "wavelet": wavelet,
                    "boundary_mode": mode,
                    "category": scheme.category,
                    "backend": "ttnn-wavelet",
                    "architecture": architecture,
                    "logical_input": logical_input,
                },
                error,
            )
        )


def run_ttnn_performance(
    args: argparse.Namespace,
    schemes: dict[str, SchemeMetadata],
    results: CsvResults,
    dimension: int,
    inputs: Sequence[int | tuple[int, int]],
    metadata_path: Path,
    log_dir: Path,
) -> None:
    total = len(args.wavelets) * len(args.boundary_modes) * len(inputs)
    iteration = 0
    with TtnnSession(log_dir) as session:
        update_device_metadata(
            metadata_path, session.architecture, session.device_information
        )
        for wavelet in args.wavelets:
            scheme = schemes[wavelet]
            for mode in args.boundary_modes:
                for logical in inputs:
                    iteration += 1
                    progress(
                        f"performance-{dimension}d ttnn-wavelet",
                        iteration,
                        total,
                        f"{wavelet}/{mode} input={logical}",
                    )
                    try:
                        case = prepare_ttnn_performance_case(
                            session, args, wavelet, mode, dimension, logical
                        )
                    except Exception as error:
                        _write_ttnn_setup_errors(
                            results,
                            dimension=dimension,
                            wavelet=wavelet,
                            mode=mode,
                            scheme=scheme,
                            logical=logical,
                            architecture=session.architecture,
                            error=error,
                        )
                        continue
                    logical_input = case.logical_input
                    outputs = case.outputs
                    reconstructed = case.reconstructed
                    layout = case.layout
                    operations = case.operations
                    for (
                        transform,
                        operation,
                        physical_input,
                        logical_output,
                    ) in operations:
                        case_id = _performance_id(
                            "ttnn-wavelet",
                            wavelet,
                            mode,
                            dimension,
                            transform,
                            logical_input,
                        )
                        if case_id in results.completed:
                            continue
                        base = {
                            "dimension": f"{dimension}d",
                            "transform": transform,
                            "wavelet": wavelet,
                            "boundary_mode": mode,
                            "category": scheme.category,
                            "backend": "ttnn-wavelet",
                            "architecture": session.architecture,
                            "logical_input": logical_input,
                        }
                        try:
                            print(
                                f"[ttnn-wavelet timing] {case_id}",
                                file=PROGRESS_STREAM,
                                flush=True,
                            )
                            timing = session.measure(
                                operation, args.warmup_runs, args.repeats
                            )
                            physical_output = (
                                ttnn_padded_shapes(*outputs)
                                if transform in ("lwt", "lwt_2d")
                                else ttnn_padded_shapes(reconstructed)
                            )
                            row = make_performance_row(
                                backend="ttnn-wavelet",
                                dimension=dimension,
                                transform=transform,
                                wavelet=wavelet,
                                mode=mode,
                                scheme=scheme,
                                logical_input=logical_input,
                                physical_input=physical_input,
                                logical_output=logical_output,
                                physical_output=physical_output,
                                architecture=session.architecture,
                                device_model=session.device_information,
                                layout=layout,
                                core_count=(
                                    "unavailable through public TTNN API; "
                                    "factory uses min(worker-grid, work-items)"
                                ),
                                planner_decision={
                                    "provenance": "TTNN public operation API",
                                    "core_policy": "min(worker-grid, work-items)",
                                    "standalone_2d_core_cap": (
                                        args.cores if dimension == 2 else None
                                    ),
                                    "comparability": (
                                        "standalone is equivalent only when its reported active core count "
                                        "matches TTNN factory policy"
                                    ),
                                },
                                memory_config=(
                                    "input/output: TTNN DRAM memory config; "
                                    "temporary workspace: program-local L1 circular buffers"
                                ),
                                warmup_runs=args.warmup_runs,
                                repeats=args.repeats,
                                timing_mechanism=TIMING_METHOD,
                                primary_metric="device_time_ms",
                                samples=timing["device_time_ms"],
                                diagnostics=timing,
                            )
                            row["program_config_sizes_bytes"] = (
                                "unavailable through public TTNN Python API"
                            )
                            row["program_config_max_bytes"] = (
                                "unavailable through public TTNN Python API"
                            )
                            results.write(row)
                        except Exception as error:
                            results.write(error_row(case_id, base, error))


def summarize_performance(path: Path) -> dict[str, int]:
    latest: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            latest[row["case_id"]] = row
    return {
        "rows": len(latest),
        "ok": sum(row["status"] == "ok" for row in latest.values()),
        "errors": sum(row["status"] != "ok" for row in latest.values()),
    }


def preflight_has_only_known_numerical_risks(path: Path) -> bool:
    """Allow performance after documented FP32 conditioning failures only.

    Precision rows remain failed in the CSV. Compile, dispatch, shape, finite,
    and all unexpected precision failures still stop a long benchmark.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row["status"] != "ok" for row in rows):
        return False
    return all(
        row["passed"] == "True" or row["wavelet"] in KNOWN_NUMERICAL_RISK_SCHEMES
        for row in rows
    )


def _preflight_args(
    args: argparse.Namespace, output_dir: Path, backend: str
) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=output_dir,
        overwrite=args.overwrite,
        resume=args.resume,
        seed=args.seed,
        wavelets=list(args.wavelets),
        boundary_modes=list(args.boundary_modes),
        backends=[backend],
        length=args.preflight_length,
        height=args.preflight_height,
        width=args.preflight_width,
        input_scale=args.preflight_input_scale,
        cores=args.cores,
    )


def run_backend_preflight(
    args: argparse.Namespace,
    schemes: dict[str, SchemeMetadata],
    output_dir: Path,
    backend: str,
) -> None:
    preflight_dir = output_dir / "preflight" / backend
    preflight_args = _preflight_args(args, preflight_dir, backend)
    preflight_metadata = create_metadata(
        "preflight", preflight_args, preflight_args.wavelets, schemes
    )
    preflight_current = (
        args.resume
        and (
            (preflight_dir / "PASSED").exists()
            or (preflight_dir / "ACCEPTED_KNOWN_NUMERICAL_RISKS").exists()
        )
        and metadata_matches(preflight_dir / "metadata.json", preflight_metadata)
    )
    if preflight_current:
        return

    preflight_passed = run_precision_command(preflight_args, schemes, "preflight")
    preflight_csv = preflight_dir / "precision.csv"
    if not preflight_passed and not preflight_has_only_known_numerical_risks(
        preflight_csv
    ):
        raise RuntimeError(
            f"Mandatory {backend} preflight failed; inspect {preflight_csv}"
        )
    if not preflight_passed:
        (preflight_dir / "ACCEPTED_KNOWN_NUMERICAL_RISKS").write_text(
            "compile/dispatch/shape/finite checks passed; raw known FP32 "
            "conditioning failures remain in precision.csv\n",
            encoding="utf-8",
        )
        print(
            f"Mandatory {backend} preflight has known FP32 conditioning "
            "failures; compile/dispatch checks passed and performance will "
            f"continue. Raw failures remain in {preflight_csv}",
            file=sys.stderr,
        )


def run_performance_command(
    args: argparse.Namespace,
    schemes: dict[str, SchemeMetadata],
    dimension: int,
) -> bool:
    output_dir = args.output_dir.resolve()
    prepare_output_dir(output_dir, args.overwrite, args.resume)
    metadata_path = output_dir / "metadata.json"
    verify_or_write_metadata(
        metadata_path,
        create_metadata(f"performance-{dimension}d", args, args.wavelets, schemes),
        args.resume,
    )
    if dimension == 1:
        inputs: Sequence[int | tuple[int, int]] = args.lengths or list(
            range(args.length_start, args.length_stop + 1, args.length_step)
        )
    else:
        inputs = args.shapes or [(1000, width) for width in range(100, 1001, 10)]
    results_path = output_dir / f"performance_{dimension}d.csv"
    with CsvResults(results_path, PERFORMANCE_FIELDS, args.resume) as results:
        if "pywavelets" in args.backends:
            run_pywavelets_performance(args, schemes, results, dimension, inputs)
        if "tt-wavelet" in args.backends:
            run_backend_preflight(args, schemes, output_dir, backend="tt-wavelet")
            run_standalone_performance(
                args, schemes, results, dimension, inputs, output_dir, metadata_path
            )
        if "ttnn-wavelet" in args.backends:
            run_backend_preflight(args, schemes, output_dir, backend="ttnn-wavelet")
            run_ttnn_performance(
                args,
                schemes,
                results,
                dimension,
                inputs,
                metadata_path,
                output_dir / "logs",
            )
    summary = summarize_performance(results_path)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return summary["errors"] == 0


def parse_shape(value: str) -> tuple[int, int]:
    try:
        height, width = value.lower().split("x", maxsplit=1)
        shape = (int(height), int(width))
    except (ValueError, AttributeError) as error:
        raise argparse.ArgumentTypeError(f"Expected HxW, got {value!r}") from error
    if shape[0] <= 0 or shape[1] <= 0:
        raise argparse.ArgumentTypeError("Shape dimensions must be positive")
    return shape


def _latest_csv_rows(path: Path) -> list[dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            latest[row["case_id"]] = row
    return [row for row in latest.values() if row["status"] == "ok"]


def _logical_input_sort_key(logical_input: str) -> tuple[int, int]:
    if "x" not in logical_input:
        return (0, int(logical_input))
    height, width = logical_input.split("x", maxsplit=1)
    return (int(height), int(width))


def _plot_positions(
    dimension: str, group: Sequence[dict[str, str]]
) -> tuple[dict[str, int], str, bool]:
    logical_inputs = sorted(
        {row["logical_input"] for row in group}, key=_logical_input_sort_key
    )
    if dimension != "2d":
        return {value: int(value) for value in logical_inputs}, "signal length", False
    shapes = [tuple(map(int, value.split("x", maxsplit=1))) for value in logical_inputs]
    heights = {height for height, _ in shapes}
    if len(heights) == 1:
        return (
            {value: width for value, (_, width) in zip(logical_inputs, shapes)},
            f"width (height={next(iter(heights))})",
            False,
        )
    return (
        {value: index for index, value in enumerate(logical_inputs)},
        "input shape (H×W)",
        True,
    )


def generate_plots(csv_paths: Sequence[Path], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for path in csv_paths for row in _latest_csv_rows(path)]
    if not rows:
        raise RuntimeError("No successful performance rows to plot")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        base_groups.setdefault(
            (row["dimension"], row["transform"], row["wavelet"]), []
        ).append(row)
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for (dimension, transform, wavelet), base_group in base_groups.items():
        hardware_architectures = sorted(
            {
                row["architecture"]
                for row in base_group
                if row["architecture"] not in ("", "CPU")
            }
        )
        if not hardware_architectures:
            grouped[(dimension, transform, wavelet, "CPU")] = base_group
            continue
        for architecture in hardware_architectures:
            grouped[(dimension, transform, wavelet, architecture)] = [
                row
                for row in base_group
                if row["architecture"] in ("CPU", architecture)
            ]

    speedup_fields = (
        "dimension",
        "transform",
        "wavelet",
        "architecture",
        "boundary_mode",
        "logical_input",
        "comparison",
        "speedup",
    )
    with (output_dir / "speedups.csv").open(
        "w", newline="", encoding="utf-8"
    ) as speedup_handle:
        speedup_writer = csv.DictWriter(speedup_handle, fieldnames=speedup_fields)
        speedup_writer.writeheader()
        for (dimension, transform, wavelet, architecture), group in sorted(
            grouped.items()
        ):
            x_positions, x_label, categorical_x = _plot_positions(dimension, group)
            figure, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
            for mode, axis in zip(BOUNDARY_MODES, axes.flat):
                mode_rows = [row for row in group if row["boundary_mode"] == mode]
                by_backend: dict[str, dict[str, float]] = {}
                for row in mode_rows:
                    logical_input = row["logical_input"]
                    by_backend.setdefault(row["backend"], {})[logical_input] = float(
                        row["median_ms"]
                    )
                for backend, points in sorted(by_backend.items()):
                    x_values = sorted(points, key=x_positions.__getitem__)
                    axis.plot(
                        [x_positions[value] for value in x_values],
                        [points[value] for value in x_values],
                        marker=".",
                        label=backend,
                    )
                axis.set_title(mode)
                axis.set_ylabel("milliseconds (CPU API / TT device)")
                axis.grid(True, linestyle=":", alpha=0.5)
                if by_backend:
                    axis.legend(fontsize=8)

                comparisons = (
                    ("tt-wavelet/PyWavelets", "pywavelets", "tt-wavelet"),
                    ("TTNN/PyWavelets", "pywavelets", "ttnn-wavelet"),
                    ("TTNN/standalone", "tt-wavelet", "ttnn-wavelet"),
                )
                for label, baseline, candidate in comparisons:
                    common = sorted(
                        set(by_backend.get(baseline, {}))
                        & set(by_backend.get(candidate, {})),
                        key=x_positions.__getitem__,
                    )
                    values = [
                        by_backend[baseline][value] / by_backend[candidate][value]
                        for value in common
                    ]
                    for x_value, value in zip(common, values):
                        speedup_writer.writerow(
                            {
                                "dimension": dimension,
                                "transform": transform,
                                "wavelet": wavelet,
                                "architecture": architecture,
                                "boundary_mode": mode,
                                "logical_input": x_value,
                                "comparison": label,
                                "speedup": value,
                            }
                        )
            for axis in axes[-1, :]:
                axis.set_xlabel(x_label)
            if categorical_x:
                labels = sorted(x_positions, key=x_positions.__getitem__)
                positions = [x_positions[label] for label in labels]
                for axis in axes[-1, :]:
                    axis.set_xticks(positions, labels, rotation=35, ha="right")
            figure.suptitle(
                f"{dimension} {transform} {wavelet} — raw time; PyWavelets CPU API, TT device profiler — {architecture}"
            )
            figure.tight_layout(rect=(0, 0, 1, 0.98))
            safe_architecture = "".join(
                character if character.isalnum() else "_" for character in architecture
            )
            stem = f"{dimension}_{transform}_{wavelet.replace('.', '_')}_{safe_architecture}"
            figure.savefig(output_dir / f"{stem}_milliseconds.png", dpi=180)
            plt.close(figure)


def add_common_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wavelets", nargs="*", default=[])
    parser.add_argument(
        "--boundary-modes",
        nargs="+",
        choices=BOUNDARY_MODES,
        default=list(BOUNDARY_MODES),
    )
    parser.add_argument(
        "--backends", nargs="+", choices=ALL_BACKENDS, default=list(ALL_BACKENDS)
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=0,
        help="standalone 2D core cap; 0 uses all available cores, matching TTNN source policy",
    )


def add_output_arguments(parser: argparse.ArgumentParser, default: str) -> None:
    parser.add_argument("--output-dir", type=Path, default=Path(default))
    state = parser.add_mutually_exclusive_group()
    state.add_argument("--overwrite", action="store_true")
    state.add_argument("--resume", action="store_true")


def add_performance_arguments(parser: argparse.ArgumentParser, default: str) -> None:
    add_common_selection_arguments(parser)
    add_output_arguments(parser, default)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="measured repeats per performance point (default: %(default)s)",
    )
    parser.add_argument("--preflight-length", type=int, default=257)
    parser.add_argument("--preflight-height", type=int, default=35)
    parser.add_argument("--preflight-width", type=int, default=37)
    parser.add_argument(
        "--preflight-input-scale",
        type=float,
        default=1.0e-6,
        help="normalized finite-FP32 coif17 preflight scale; precision policy unchanged",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    precision_parser = commands.add_parser(
        "precision", help="Validate every transform against PyWavelets"
    )
    add_common_selection_arguments(precision_parser)
    add_output_arguments(precision_parser, "benchmarks/precision")
    precision_parser.add_argument("--length", type=int, default=1024)
    precision_parser.add_argument("--height", type=int, default=64)
    precision_parser.add_argument("--width", type=int, default=64)
    precision_parser.add_argument("--input-scale", type=float, default=1.0)

    preflight_parser = commands.add_parser(
        "preflight", help="Compile/dispatch/sanity preflight"
    )
    add_common_selection_arguments(preflight_parser)
    add_output_arguments(preflight_parser, "benchmarks/preflight")
    preflight_parser.add_argument("--length", type=int, default=257)
    preflight_parser.add_argument("--height", type=int, default=35)
    preflight_parser.add_argument("--width", type=int, default=37)
    preflight_parser.add_argument(
        "--input-scale",
        type=float,
        default=1.0e-6,
        help="normalized finite-FP32 coif17 preflight scale; precision policy unchanged",
    )

    performance_1d = commands.add_parser(
        "performance-1d", help="Run reproducible 1D timing sweep"
    )
    add_performance_arguments(performance_1d, "benchmarks/performance-1d")
    performance_1d.add_argument("--lengths", nargs="*", type=int)
    performance_1d.add_argument("--length-start", type=int, default=100_000)
    performance_1d.add_argument("--length-stop", type=int, default=1_000_000)
    performance_1d.add_argument("--length-step", type=int, default=10_000)

    performance_2d = commands.add_parser(
        "performance-2d", help="Run explicit HxW timing sweep"
    )
    add_performance_arguments(performance_2d, "benchmarks/performance-2d")
    performance_2d.add_argument("--shapes", nargs="*", type=parse_shape)

    plot_parser = commands.add_parser(
        "plot", help="Regenerate plots from performance CSV"
    )
    plot_parser.add_argument("--csv", nargs="+", type=Path, required=True)
    plot_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def validate_args(args: argparse.Namespace, schemes: dict[str, SchemeMetadata]) -> None:
    if args.command == "plot":
        return
    if not args.wavelets:
        if args.command in ("performance-1d", "performance-2d", "preflight"):
            args.wavelets = select_performance_wavelets(schemes, args.seed)
        else:
            args.wavelets = sorted(schemes)
    unknown = sorted(set(args.wavelets) - set(schemes))
    if unknown:
        raise ValueError(f"Unknown wavelets: {unknown}")
    args.wavelets = list(dict.fromkeys(args.wavelets))
    if (
        args.command in ("performance-1d", "performance-2d", "preflight")
        and "coif17" not in args.wavelets
    ):
        args.wavelets.append("coif17")
    if hasattr(args, "repeats") and args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if hasattr(args, "warmup_runs") and args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative")
    if hasattr(args, "input_scale") and (
        not math.isfinite(args.input_scale) or args.input_scale <= 0.0
    ):
        raise ValueError("--input-scale must be finite and positive")
    if hasattr(args, "preflight_input_scale") and (
        not math.isfinite(args.preflight_input_scale)
        or args.preflight_input_scale <= 0.0
    ):
        raise ValueError("--preflight-input-scale must be finite and positive")
    if args.cores < 0:
        raise ValueError("--cores must be non-negative")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "plot":
            generate_plots(
                [path.resolve() for path in args.csv], args.output_dir.resolve()
            )
            return 0
        schemes = load_scheme_metadata()
        validate_args(args, schemes)
        if args.command in ("preflight", "performance-1d", "performance-2d"):
            print(
                json.dumps(
                    {
                        "seed": args.seed,
                        "selected_wavelets": args.wavelets,
                        "boundary_modes": args.boundary_modes,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
        if args.command == "precision":
            return 0 if run_precision_command(args, schemes) else 2
        if args.command == "preflight":
            return 0 if run_precision_command(args, schemes, "preflight") else 2
        if args.command == "performance-1d":
            return 0 if run_performance_command(args, schemes, 1) else 2
        if args.command == "performance-2d":
            return 0 if run_performance_command(args, schemes, 2) else 2
        parser.error(f"Unhandled command: {args.command}")
    except Exception as error:
        print(f"ERROR [{classify_exception(error)}]: {error}", file=sys.stderr)
        if os.environ.get("TT_WAVELET_BENCHMARK_TRACEBACK") == "1":
            traceback.print_exc()
        return 2
    return 2


def _run_ttnn_precision_2d_roundtrip(
    *,
    session: TtnnSession,
    forward_bands: tuple[Any, Any, Any, Any] | None,
    wavelet: str,
    shape: tuple[int, int],
    mode: str,
    results: CsvResults,
    scheme: SchemeMetadata,
    bands_ref: tuple[np.ndarray, ...],
    reference: np.ndarray,
) -> None:
    try:
        if forward_bands is None:
            raise RuntimeError("Roundtrip unavailable because forward operation failed")
        reconstructed_2d = session.ttnn.idwt_2d(
            *forward_bands,
            wavelet,
            shape,
            boundary_mode=mode,
        )
        _finish_precision_operation(session)
        actual = session.to_numpy(reconstructed_2d, shape)
        _write_precision_metric(
            results,
            dimension=2,
            transform="ilwt_2d",
            wavelet=wavelet,
            mode=mode,
            backend="ttnn-wavelet",
            comparison="roundtrip",
            band="reconstructed",
            scheme=scheme,
            logical_input=f"{bands_ref[0].shape[0]}x{bands_ref[0].shape[1]}",
            logical_output=f"{shape[0]}x{shape[1]}",
            actual=actual,
            reference=reference,
        )
    except Exception as error:
        _write_precision_error(
            results,
            dimension=2,
            transform="ilwt_2d",
            wavelet=wavelet,
            mode=mode,
            backend="ttnn-wavelet",
            comparison="roundtrip",
            scheme=scheme,
            logical_input=f"{bands_ref[0].shape[0]}x{bands_ref[0].shape[1]}",
            error=error,
        )


def _performance_id(
    backend: str,
    wavelet: str,
    mode: str,
    dimension: int,
    transform: str,
    logical_input: str,
) -> str:
    return "/".join((backend, wavelet, mode, f"{dimension}d", transform, logical_input))


def _add_diagnostic_statistics(
    row: dict[str, Any], prefix: str, values: Sequence[float]
) -> None:
    row[f"{prefix}_samples_ms"] = list(values)
    row[f"{prefix}_median_ms"] = statistics.median(values) if values else ""


def make_performance_row(
    *,
    backend: str,
    dimension: int,
    transform: str,
    wavelet: str,
    mode: str,
    scheme: SchemeMetadata,
    logical_input: str,
    physical_input: str,
    logical_output: str,
    physical_output: str,
    architecture: str,
    device_model: str,
    layout: str,
    core_count: Any,
    planner_decision: Any,
    memory_config: str,
    warmup_runs: int,
    repeats: int,
    timing_mechanism: str,
    primary_metric: str,
    samples: Sequence[float],
    diagnostics: dict[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    case_id = _performance_id(
        backend, wavelet, mode, dimension, transform, logical_input
    )
    row = {
        "case_id": case_id,
        "dimension": f"{dimension}d",
        "transform": transform,
        "wavelet": wavelet,
        "boundary_mode": mode,
        "category": scheme.category,
        "backend": backend,
        "architecture": architecture,
        "device_model": device_model,
        "logical_input": logical_input,
        "physical_input": physical_input,
        "logical_output": logical_output,
        "physical_output": physical_output,
        "layout": layout,
        "core_count": core_count,
        "planner_decision": planner_decision,
        "memory_config": memory_config,
        "program_config_sizes_bytes": "",
        "program_config_max_bytes": "",
        "warmup_count": warmup_runs,
        "repeat_count": repeats,
        "timing_mechanism": timing_mechanism,
        "primary_metric": primary_metric,
        **timing_statistics(samples),
        "status": "ok",
        "error_type": "",
        "error_message": "",
    }
    for prefix in ("enqueue_or_dispatch", "sync_wait", "host_api_total"):
        _add_diagnostic_statistics(
            row, prefix, (diagnostics or {}).get(f"{prefix}_ms", [])
        )
    return row


def run_pywavelets_performance(
    args: argparse.Namespace,
    schemes: dict[str, SchemeMetadata],
    results: CsvResults,
    dimension: int,
    inputs: Sequence[int | tuple[int, int]],
) -> None:
    total = len(args.wavelets) * len(args.boundary_modes) * len(inputs)
    iteration = 0
    for wavelet in args.wavelets:
        scheme = schemes[wavelet]
        for mode in args.boundary_modes:
            for logical in inputs:
                iteration += 1
                progress(
                    f"performance-{dimension}d pywavelets",
                    iteration,
                    total,
                    f"{wavelet}/{mode} input={logical}",
                )
                if dimension == 1:
                    length = int(logical)
                    values = signal_1d(length, args.seed, wavelet, mode, "performance")
                    approximation, detail = pywt.dwt(values, wavelet, mode=mode)
                    operations = {
                        "lwt": lambda: pywt.dwt(values, wavelet, mode=mode),
                        "ilwt": lambda: pywt.idwt(
                            approximation, detail, wavelet, mode=mode
                        ),
                    }
                    logical_input = str(length)
                    outputs = {"lwt": str(len(approximation)), "ilwt": str(length)}
                else:
                    height, width = logical  # type: ignore[misc]
                    values = signal_2d(
                        height, width, args.seed, wavelet, mode, "performance"
                    )
                    ll, details = pywt.dwt2(values, wavelet, mode=mode)
                    operations = {
                        "lwt_2d": lambda: pywt.dwt2(values, wavelet, mode=mode),
                        "ilwt_2d": lambda: pywt.idwt2(
                            (ll, details), wavelet, mode=mode
                        ),
                    }
                    logical_input = f"{height}x{width}"
                    outputs = {
                        "lwt_2d": f"{ll.shape[0]}x{ll.shape[1]}",
                        "ilwt_2d": logical_input,
                    }
                for transform, operation in operations.items():
                    case_id = _performance_id(
                        "pywavelets", wavelet, mode, dimension, transform, logical_input
                    )
                    if case_id in results.completed:
                        continue
                    base = {
                        "dimension": f"{dimension}d",
                        "transform": transform,
                        "wavelet": wavelet,
                        "boundary_mode": mode,
                        "category": scheme.category,
                        "backend": "pywavelets",
                        "logical_input": logical_input,
                    }
                    try:
                        for _ in range(args.warmup_runs):
                            operation()
                        samples: list[float] = []
                        for _ in range(args.repeats):
                            start = time.perf_counter()
                            operation()
                            samples.append((time.perf_counter() - start) * 1.0e3)
                        results.write(
                            make_performance_row(
                                backend="pywavelets",
                                dimension=dimension,
                                transform=transform,
                                wavelet=wavelet,
                                mode=mode,
                                scheme=scheme,
                                logical_input=logical_input,
                                physical_input=logical_input,
                                logical_output=outputs[transform],
                                physical_output=outputs[transform],
                                architecture="CPU",
                                device_model="host CPU",
                                layout="NumPy contiguous row-major",
                                core_count="host-library-managed",
                                planner_decision="PyWavelets native implementation",
                                memory_config="host memory",
                                warmup_runs=args.warmup_runs,
                                repeats=args.repeats,
                                timing_mechanism="python_perf_counter_cpu_api",
                                primary_metric="cpu_api_time_ms",
                                samples=samples,
                            )
                        )
                    except Exception as error:
                        results.write(error_row(case_id, base, error))


def _standalone_performance_row(
    response: dict[str, Any],
    *,
    dimension: int,
    transform: str,
    wavelet: str,
    mode: str,
    scheme: SchemeMetadata,
    logical_input: str,
) -> dict[str, Any]:
    device_samples = response.get("device_time_ms", [])
    if not device_samples:
        raise RuntimeError("Standalone response omitted device profiler samples")
    logical_output_value = response.get(
        "logical_output_size", response.get("logical_output_shape", "")
    )
    physical_input_value = response.get(
        "physical_input_size", response.get("physical_input_shape", "")
    )
    physical_output_value = response.get(
        "physical_output_size", response.get("physical_output_shape", "")
    )
    planner = {
        key: response[key]
        for key in (
            "chunk_count",
            "route_count",
            "planner_groups_per_chunk",
            "requested_core_limit",
        )
        if key in response
    }
    row = make_performance_row(
        backend="tt-wavelet",
        dimension=dimension,
        transform=transform,
        wavelet=wavelet,
        mode=mode,
        scheme=scheme,
        logical_input=logical_input,
        physical_input=json.dumps(physical_input_value, separators=(",", ":")),
        logical_output=json.dumps(logical_output_value, separators=(",", ":")),
        physical_output=json.dumps(physical_output_value, separators=(",", ":")),
        architecture=str(response.get("architecture", "unknown")),
        device_model="standalone MeshDevice unit mesh device 0",
        layout=str(response.get("layout", "unknown")),
        core_count=response.get("core_count", ""),
        planner_decision=planner,
        memory_config=str(response.get("memory_config", "unknown")),
        warmup_runs=int(response.get("warmup_count", 0)),
        repeats=int(response.get("repeat_count", len(device_samples))),
        timing_mechanism=str(response.get("timing_mechanism", TIMING_METHOD)),
        primary_metric="device_time_ms",
        samples=device_samples,
        diagnostics={
            "enqueue_or_dispatch_ms": response.get("enqueue_or_dispatch_ms", []),
            "sync_wait_ms": response.get("sync_wait_ms", []),
            "host_api_total_ms": response.get("host_api_total_ms", []),
        },
    )
    row["program_config_sizes_bytes"] = response.get("program_config_sizes_bytes", [])
    row["program_config_max_bytes"] = response.get("program_config_max_bytes", "")
    return row


def run_standalone_performance(
    args: argparse.Namespace,
    schemes: dict[str, SchemeMetadata],
    results: CsvResults,
    dimension: int,
    inputs: Sequence[int | tuple[int, int]],
    output_dir: Path,
    metadata_path: Path,
) -> None:
    total = len(args.wavelets) * len(args.boundary_modes) * len(inputs)
    iteration = 0
    with tempfile.TemporaryDirectory(
        prefix="ttwv-performance-"
    ) as temporary_name, StandaloneSession(output_dir / "logs") as session:
        temporary = Path(temporary_name)
        architecture_recorded = False
        for wavelet in args.wavelets:
            scheme = schemes[wavelet]
            for mode in args.boundary_modes:
                for logical in inputs:
                    iteration += 1
                    progress(
                        f"performance-{dimension}d tt-wavelet",
                        iteration,
                        total,
                        f"{wavelet}/{mode} input={logical}",
                    )
                    prefix = temporary / f"case-{iteration}"
                    requests: list[tuple[str, dict[str, Any]]] = []
                    if dimension == 1:
                        length = int(logical)
                        values = signal_1d(
                            length, args.seed, wavelet, mode, "performance"
                        )
                        approximation, detail = (
                            np.asarray(value, dtype=np.float32)
                            for value in pywt.dwt(values, wavelet, mode=mode)
                        )
                        input_path = prefix.with_suffix(".input.f32")
                        approximation_path = prefix.with_suffix(".approximation.f32")
                        detail_path = prefix.with_suffix(".detail.f32")
                        write_f32(input_path, values)
                        write_f32(approximation_path, approximation)
                        write_f32(detail_path, detail)
                        logical_input = str(length)
                        common = {
                            "dimension": 1,
                            "wavelet": wavelet,
                            "boundary_mode": mode,
                            "length": length,
                            "capture_outputs": False,
                            "warmup_runs": args.warmup_runs,
                            "repeats": args.repeats,
                        }
                        requests = [
                            (
                                "lwt",
                                {
                                    **common,
                                    "case_id": f"runner/{wavelet}/{mode}/1d/lwt/{length}",
                                    "transform": "lwt",
                                    "input_paths": [str(input_path)],
                                },
                            ),
                            (
                                "ilwt",
                                {
                                    **common,
                                    "case_id": f"runner/{wavelet}/{mode}/1d/ilwt/{length}",
                                    "transform": "ilwt",
                                    "coefficient_length": int(approximation.size),
                                    "input_paths": [
                                        str(approximation_path),
                                        str(detail_path),
                                    ],
                                },
                            ),
                        ]
                    else:
                        height, width = logical  # type: ignore[misc]
                        values = signal_2d(
                            height, width, args.seed, wavelet, mode, "performance"
                        )
                        bands = pywt_bands_2d(values, wavelet, mode)
                        input_path = prefix.with_suffix(".input.f32")
                        write_f32(input_path, values)
                        band_paths: list[str] = []
                        for name, band in zip(("LL", "LH", "HL", "HH"), bands):
                            path = Path(f"{prefix}.{name}.f32")
                            write_f32(path, band)
                            band_paths.append(str(path))
                        logical_input = f"{height}x{width}"
                        common = {
                            "dimension": 2,
                            "wavelet": wavelet,
                            "boundary_mode": mode,
                            "height": height,
                            "width": width,
                            "cores": args.cores,
                            "capture_outputs": False,
                            "warmup_runs": args.warmup_runs,
                            "repeats": args.repeats,
                        }
                        requests = [
                            (
                                "lwt_2d",
                                {
                                    **common,
                                    "case_id": f"runner/{wavelet}/{mode}/2d/lwt/{logical_input}",
                                    "transform": "lwt_2d",
                                    "input_paths": [str(input_path)],
                                },
                            ),
                            (
                                "ilwt_2d",
                                {
                                    **common,
                                    "case_id": f"runner/{wavelet}/{mode}/2d/ilwt/{logical_input}",
                                    "transform": "ilwt_2d",
                                    "input_paths": band_paths,
                                },
                            ),
                        ]
                    for transform, request in requests:
                        case_id = _performance_id(
                            "tt-wavelet",
                            wavelet,
                            mode,
                            dimension,
                            transform,
                            logical_input,
                        )
                        if case_id in results.completed:
                            continue
                        base = {
                            "dimension": f"{dimension}d",
                            "transform": transform,
                            "wavelet": wavelet,
                            "boundary_mode": mode,
                            "category": scheme.category,
                            "backend": "tt-wavelet",
                            "logical_input": logical_input,
                        }
                        try:
                            print(
                                f"[tt-wavelet timing] {case_id}",
                                file=PROGRESS_STREAM,
                                flush=True,
                            )
                            response = session.run(request)
                            if response.get("status") != "ok":
                                raise _runner_error(response)
                            results.write(
                                _standalone_performance_row(
                                    response,
                                    dimension=dimension,
                                    transform=transform,
                                    wavelet=wavelet,
                                    mode=mode,
                                    scheme=scheme,
                                    logical_input=logical_input,
                                )
                            )
                            if not architecture_recorded:
                                update_device_metadata(
                                    metadata_path,
                                    str(response.get("architecture", "unknown")),
                                    "standalone MeshDevice unit mesh device 0",
                                )
                                architecture_recorded = True
                        except TimeoutError:
                            raise
                        except Exception as error:
                            results.write(error_row(case_id, base, error))


def summarize_precision(path: Path) -> dict[str, int]:
    summary = {"rows": 0, "passed": 0, "failed": 0, "errors": 0}
    latest: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            latest[row["case_id"]] = row
    for row in latest.values():
        summary["rows"] += 1
        if row["status"] != "ok":
            summary["errors"] += 1
        elif row["passed"].lower() == "true":
            summary["passed"] += 1
        else:
            summary["failed"] += 1
    return summary


def run_precision_command(
    args: argparse.Namespace,
    schemes: dict[str, SchemeMetadata],
    command_name: str = "precision",
) -> bool:
    output_dir = args.output_dir.resolve()
    prepare_output_dir(output_dir, args.overwrite, args.resume)
    metadata_path = output_dir / "metadata.json"
    verify_or_write_metadata(
        metadata_path,
        create_metadata(command_name, args, args.wavelets, schemes),
        args.resume,
    )
    results_path = output_dir / "precision.csv"
    with CsvResults(
        results_path,
        PRECISION_FIELDS,
        args.resume,
        require_passed_for_resume=True,
    ) as results:
        results.precision_input_scale = args.input_scale
        for backend in args.backends:
            if backend == "pywavelets":
                run_pywavelets_precision(args, schemes, results)
            elif backend == "tt-wavelet":
                run_standalone_precision(
                    args, schemes, results, output_dir, metadata_path
                )
            elif backend == "ttnn-wavelet":
                run_ttnn_precision(
                    args, schemes, results, metadata_path, output_dir / "logs"
                )
    summary = summarize_precision(results_path)
    if command_name == "preflight":
        # Default normalized input keeps ill-conditioned coif17 lifting finite
        # in FP32. It changes no tolerance and strict failures still stop runs.
        success = summary["errors"] == 0 and summary["failed"] == 0
        summary["strict_precision_failures"] = summary["failed"]
    else:
        success = summary["errors"] == 0 and summary["failed"] == 0
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    marker = output_dir / "PASSED"
    if success:
        marker.write_text(
            (
                "compile/dispatch/shape/finite/strict-precision preflight passed\n"
                if command_name == "preflight"
                else "all precision rows passed\n"
            ),
            encoding="utf-8",
        )
    elif marker.exists():
        marker.unlink()
    print(json.dumps(summary, sort_keys=True))
    return success


def _run_standalone_precision_remainder(
    *,
    args: argparse.Namespace,
    results: CsvResults,
    session: StandaloneSession,
    wavelet: str,
    mode: str,
    scheme: SchemeMetadata,
    prefix: Path,
    x1: np.ndarray,
    approximation_ref: np.ndarray,
    detail_ref: np.ndarray,
    approximation_path: Path,
    detail_path: Path,
    forward_prefix: Path,
) -> None:
    inverse_prefix = Path(str(prefix) + ".inverse1d")
    inverse_request = {
        "case_id": f"runner/{wavelet}/{mode}/1d/ilwt-reference",
        "dimension": 1,
        "transform": "ilwt",
        "wavelet": wavelet,
        "boundary_mode": mode,
        "length": args.length,
        "coefficient_length": int(approximation_ref.size),
        "input_paths": [str(approximation_path), str(detail_path)],
        "capture_outputs": True,
        "output_prefix": str(inverse_prefix),
        "warmup_runs": 0,
        "repeats": 1,
    }
    try:
        response = session.run(inverse_request)
        if response.get("status") != "ok":
            raise _runner_error(response)
        actual = read_f32(Path(f"{inverse_prefix}.reconstructed.f32"), x1.shape)
        _write_precision_metric(
            results,
            dimension=1,
            transform="ilwt",
            wavelet=wavelet,
            mode=mode,
            backend="tt-wavelet",
            comparison="inverse-reference-coefficients",
            band="reconstructed",
            scheme=scheme,
            logical_input=str(approximation_ref.size),
            logical_output=str(args.length),
            actual=actual,
            reference=x1,
        )
    except Exception as error:
        _write_precision_error(
            results,
            dimension=1,
            transform="ilwt",
            wavelet=wavelet,
            mode=mode,
            backend="tt-wavelet",
            comparison="inverse-reference-coefficients",
            scheme=scheme,
            logical_input=str(approximation_ref.size),
            error=error,
        )

    roundtrip_prefix = Path(str(prefix) + ".roundtrip1d")
    roundtrip_request = {
        **inverse_request,
        "case_id": f"runner/{wavelet}/{mode}/1d/ilwt-roundtrip",
        "input_paths": [
            f"{forward_prefix}.approximation.f32",
            f"{forward_prefix}.detail.f32",
        ],
        "output_prefix": str(roundtrip_prefix),
    }
    try:
        response = session.run(roundtrip_request)
        if response.get("status") != "ok":
            raise _runner_error(response)
        actual = read_f32(Path(f"{roundtrip_prefix}.reconstructed.f32"), x1.shape)
        _write_precision_metric(
            results,
            dimension=1,
            transform="ilwt",
            wavelet=wavelet,
            mode=mode,
            backend="tt-wavelet",
            comparison="roundtrip",
            band="reconstructed",
            scheme=scheme,
            logical_input=str(approximation_ref.size),
            logical_output=str(args.length),
            actual=actual,
            reference=x1,
        )
    except Exception as error:
        _write_precision_error(
            results,
            dimension=1,
            transform="ilwt",
            wavelet=wavelet,
            mode=mode,
            backend="tt-wavelet",
            comparison="roundtrip",
            scheme=scheme,
            logical_input=str(approximation_ref.size),
            error=error,
        )

    shape = (args.height, args.width)
    x2 = signal_2d(*shape, args.seed, wavelet, mode) * args.input_scale
    bands_ref = pywt_bands_2d(x2, wavelet, mode)
    input_2d = prefix.with_suffix(".input2d.f32")
    write_f32(input_2d, x2)
    band_paths: list[str] = []
    for name, band in zip(("LL", "LH", "HL", "HH"), bands_ref):
        path = Path(f"{prefix}.{name}-ref.f32")
        write_f32(path, band)
        band_paths.append(str(path))

    forward_2d_prefix = Path(str(prefix) + ".forward2d")
    forward_2d_request = {
        "case_id": f"runner/{wavelet}/{mode}/2d/lwt",
        "dimension": 2,
        "transform": "lwt_2d",
        "wavelet": wavelet,
        "boundary_mode": mode,
        "height": shape[0],
        "width": shape[1],
        "cores": args.cores,
        "input_paths": [str(input_2d)],
        "capture_outputs": True,
        "output_prefix": str(forward_2d_prefix),
        "warmup_runs": 0,
        "repeats": 1,
    }
    try:
        response = session.run(forward_2d_request)
        if response.get("status") != "ok":
            raise _runner_error(response)
        for name, reference in zip(("LL", "LH", "HL", "HH"), bands_ref):
            actual = read_f32(Path(f"{forward_2d_prefix}.{name}.f32"), reference.shape)
            _write_precision_metric(
                results,
                dimension=2,
                transform="lwt_2d",
                wavelet=wavelet,
                mode=mode,
                backend="tt-wavelet",
                comparison="forward-vs-pywavelets",
                band=name,
                scheme=scheme,
                logical_input=f"{shape[0]}x{shape[1]}",
                logical_output=f"{reference.shape[0]}x{reference.shape[1]}",
                actual=actual,
                reference=reference,
            )
    except Exception as error:
        _write_precision_error(
            results,
            dimension=2,
            transform="lwt_2d",
            wavelet=wavelet,
            mode=mode,
            backend="tt-wavelet",
            comparison="forward-vs-pywavelets",
            scheme=scheme,
            logical_input=f"{shape[0]}x{shape[1]}",
            error=error,
        )

    inverse_2d_prefix = Path(str(prefix) + ".inverse2d")
    inverse_2d_request = {
        "case_id": f"runner/{wavelet}/{mode}/2d/ilwt-reference",
        "dimension": 2,
        "transform": "ilwt_2d",
        "wavelet": wavelet,
        "boundary_mode": mode,
        "height": shape[0],
        "width": shape[1],
        "cores": args.cores,
        "input_paths": band_paths,
        "capture_outputs": True,
        "output_prefix": str(inverse_2d_prefix),
        "warmup_runs": 0,
        "repeats": 1,
    }
    try:
        response = session.run(inverse_2d_request)
        if response.get("status") != "ok":
            raise _runner_error(response)
        actual = read_f32(Path(f"{inverse_2d_prefix}.reconstructed.f32"), shape)
        _write_precision_metric(
            results,
            dimension=2,
            transform="ilwt_2d",
            wavelet=wavelet,
            mode=mode,
            backend="tt-wavelet",
            comparison="inverse-reference-coefficients",
            band="reconstructed",
            scheme=scheme,
            logical_input=f"{bands_ref[0].shape[0]}x{bands_ref[0].shape[1]}",
            logical_output=f"{shape[0]}x{shape[1]}",
            actual=actual,
            reference=x2,
        )
    except Exception as error:
        _write_precision_error(
            results,
            dimension=2,
            transform="ilwt_2d",
            wavelet=wavelet,
            mode=mode,
            backend="tt-wavelet",
            comparison="inverse-reference-coefficients",
            scheme=scheme,
            logical_input=f"{bands_ref[0].shape[0]}x{bands_ref[0].shape[1]}",
            error=error,
        )

    roundtrip_2d_prefix = Path(str(prefix) + ".roundtrip2d")
    roundtrip_2d_request = {
        **inverse_2d_request,
        "case_id": f"runner/{wavelet}/{mode}/2d/ilwt-roundtrip",
        "input_paths": [
            f"{forward_2d_prefix}.{name}.f32" for name in ("LL", "LH", "HL", "HH")
        ],
        "output_prefix": str(roundtrip_2d_prefix),
    }
    try:
        response = session.run(roundtrip_2d_request)
        if response.get("status") != "ok":
            raise _runner_error(response)
        actual = read_f32(Path(f"{roundtrip_2d_prefix}.reconstructed.f32"), shape)
        _write_precision_metric(
            results,
            dimension=2,
            transform="ilwt_2d",
            wavelet=wavelet,
            mode=mode,
            backend="tt-wavelet",
            comparison="roundtrip",
            band="reconstructed",
            scheme=scheme,
            logical_input=f"{bands_ref[0].shape[0]}x{bands_ref[0].shape[1]}",
            logical_output=f"{shape[0]}x{shape[1]}",
            actual=actual,
            reference=x2,
        )
    except Exception as error:
        _write_precision_error(
            results,
            dimension=2,
            transform="ilwt_2d",
            wavelet=wavelet,
            mode=mode,
            backend="tt-wavelet",
            comparison="roundtrip",
            scheme=scheme,
            logical_input=f"{bands_ref[0].shape[0]}x{bands_ref[0].shape[1]}",
            error=error,
        )


if __name__ == "__main__":
    raise SystemExit(main())
