#!/usr/bin/env python3
"""Produce a fail-closed GPU-NAM production consumer receipt.

The existing GPU-NAM diagnostics are the executable product path: they load a
real ``.nam`` model, prepare the public Pulp GPU provider, compare the selected
GPU output with the stateful CPU oracle, and account for CPU fallback.  This
wrapper turns that output into one immutable receipt that can be consumed by
the unified control/docs review.  It deliberately runs a typed corruption
control as well; a receipt is not positive unless the control fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "pulp.gpu-nam.production-receipt.v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
MAX_ALLOWED_ERROR = 1.0e-3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_value(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def source_identity(root: Path) -> tuple[str, bool]:
    """Return an exact, clean consumer identity; never infer it from a path."""
    revision = git_value(root, "rev-parse", "--verify", "HEAD")
    status = git_value(root, "status", "--porcelain", "--untracked-files=all")
    return revision, bool(GIT_SHA.fullmatch(revision)) and status == ""


def verify_sdk_provenance(prefix: Path, provenance: dict[str, Any]) -> list[str]:
    """Validate the release marker and every integrity member it names."""
    if not isinstance(provenance, dict):
        return ["sdk provenance root must be a JSON object"]
    failures: list[str] = []
    required = {
        "schema": "pulp.sdk-provenance.v1",
        "kind": "release",
        "profile": "official-release",
        "distribution_eligible": True,
        "source_git_dirty": False,
        "build_type": "Release",
    }
    for key, expected in required.items():
        if provenance.get(key) != expected:
            failures.append(f"sdk provenance {key}={provenance.get(key)!r}, expected {expected!r}")
    source_sha = provenance.get("source_git_sha")
    if not isinstance(source_sha, str) or not GIT_SHA.fullmatch(source_sha):
        failures.append("sdk provenance source_git_sha is not a full lowercase commit SHA")
    platform_name = provenance.get("platform")
    if not isinstance(platform_name, str) or not re.fullmatch(r"(?:darwin|linux|windows)-(?:arm64|x64)", platform_name):
        failures.append("sdk provenance platform is missing or invalid")
    windows = isinstance(platform_name, str) and platform_name.startswith("windows-")
    coherence_members = {"include/pulp/view/widget_bridge.hpp", "lib/pulp-view-script.lib" if windows else "lib/libpulp-view-script.a"}
    gpu_members = {"lib/cmake/Pulp/PulpConfig.cmake", "lib/pulp-gpu-audio.lib" if windows else "lib/libpulp-gpu-audio.a"}
    gpu_audio = provenance.get("gpu_audio")
    if not isinstance(gpu_audio, dict) or gpu_audio.get("schema") != "pulp.sdk-gpu-audio-capabilities.v1":
        failures.append("sdk GPU capability receipt missing or malformed")
        gpu_audio = {}
    capabilities = gpu_audio.get("capabilities")
    if not isinstance(capabilities, dict) or any(capabilities.get(key) is not True for key in ("shared_provider", "shared_convolver", "exact_provider_proof")):
        failures.append("sdk authenticated shared GPU capabilities missing or malformed")
    gpu_files = gpu_audio.get("files")
    if not isinstance(gpu_files, dict) or set(gpu_files) != gpu_members:
        failures.append("sdk GPU capability integrity members incomplete")
        gpu_files = {}
    integrity = provenance.get("integrity")
    if not isinstance(integrity, dict) or integrity.get("schema") != "pulp.sdk-integrity.v1" or integrity.get("algorithm") != "sha256":
        failures.append("sdk provenance is missing pulp.sdk-integrity.v1")
    else:
        files = integrity.get("files")
        if not isinstance(files, dict) or not files:
            failures.append("sdk integrity files are missing")
        else:
            if set(files) != coherence_members:
                failures.append("sdk coherence integrity members incomplete")
            for relative in set(files) & set(gpu_files):
                if files[relative] != gpu_files[relative]:
                    failures.append(f"sdk integrity disagreement: {relative}")
            for relative, expected_hash in {**files, **gpu_files}.items():
                path = Path(relative)
                if path.is_absolute() or ".." in path.parts or not isinstance(expected_hash, str) or not SHA256.fullmatch(expected_hash):
                    failures.append(f"sdk integrity member is unsafe: {relative!r}")
                    continue
                member = prefix / path
                try:
                    inside_prefix = member.resolve().is_relative_to(prefix.resolve())
                except (OSError, ValueError):
                    inside_prefix = False
                if member.is_symlink() or not inside_prefix:
                    failures.append(f"sdk integrity member escapes prefix: {relative}")
                elif not member.is_file():
                    failures.append(f"sdk integrity member is missing: {relative}")
                elif sha256(member) != expected_hash:
                    failures.append(f"sdk integrity mismatch: {relative}")
    return failures


def parse_nonnegative_int(fields: dict[str, str], key: str) -> int | None:
    value = fields.get(key)
    if value is None or not re.fullmatch(r"[0-9]+", value):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def validate_execution(fields: dict[str, str], blocks: int) -> list[str]:
    failures: list[str] = []
    dropped = parse_nonnegative_int(fields, "input_dropped")
    primed = parse_nonnegative_int(fields, "fallback_primed")
    completions = parse_nonnegative_int(fields, "gpu_inner_completions")
    if dropped != 0:
        failures.append("input_dropped must be zero")
    if primed != blocks:
        failures.append("fallback_primed must equal requested blocks")
    if completions is None or completions <= 0:
        failures.append("gpu_inner_completions must be positive")
    if not fields.get("backend", "").strip() or fields.get("backend") == "unavailable":
        failures.append("provider backend identity must be present")
    try:
        error = float(fields.get("max_error", "nan"))
    except ValueError:
        error = float("nan")
    if not (error >= 0.0 and error <= MAX_ALLOWED_ERROR):
        failures.append(f"max_error must be finite and <= {MAX_ALLOWED_ERROR:g}")
    return failures


def parse_output(stdout: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # ``backend`` is the only value that can contain spaces.
        match = re.match(r"provider_available=(\d+)\s+backend=(.*)$", line)
        if match:
            values["provider_available"] = match.group(1)
            values["backend"] = match.group(2).strip()
        for key, value in re.findall(r"([A-Za-z][A-Za-z0-9_]*)=([^\s]+)", line):
            if key != "backend" or "backend" not in values:
                values[key] = value
    return values


def run_diagnostic(executable: Path, model: Path, blocks: int, inject: bool) -> dict[str, Any]:
    command = [
        str(executable),
        f"--model-path={model}",
        "--block-size=32",
        "--lead-blocks=4",
        f"--blocks={blocks}",
    ]
    if inject:
        command.append("--inject-direct-output-error")
    completed = subprocess.run(command, text=True, capture_output=True, timeout=180)
    parsed = parse_output(completed.stdout + "\n" + completed.stderr)
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "fields": parsed,
    }


def device_identity() -> dict[str, str]:
    """Best-effort host labels; absence never turns into a GPU claim."""
    result = {"os": platform.platform(), "machine": platform.machine()}
    if sys.platform == "darwin":
        for key, command in (
            ("hardware_model", ["sysctl", "-n", "hw.model"]),
            ("cpu_brand", ["sysctl", "-n", "machdep.cpu.brand_string"]),
        ):
            try:
                value = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
            except (OSError, subprocess.CalledProcessError):
                value = "unavailable"
            result[key] = value or "unavailable"
    return result


def model_identity(model: Path) -> dict[str, Any]:
    try:
        document = json.loads(model.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"model is not valid JSON: {exc}") from exc
    architecture = document.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        raise ValueError("model has no architecture")
    return {
        "path": str(model),
        "sha256": sha256(model),
        "architecture": architecture,
        "sample_rate_hz": document.get("sample_rate"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--sdk-prefix", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--blocks", type=int, default=96)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    diagnostic = args.diagnostic.resolve()
    model = args.model.resolve()
    sdk = args.sdk_prefix.resolve()
    output = args.output.resolve()
    if args.blocks < 1 or args.blocks > 1_000_000:
        parser.error("--blocks must be in 1..1000000")
    if not diagnostic.is_file() or not os.access(diagnostic, os.X_OK):
        parser.error("--diagnostic must name an executable")
    if not model.is_file():
        parser.error("--model must name an existing .nam file")
    provenance_path = sdk / "sdk-provenance.json"
    config_path = sdk / "lib/cmake/Pulp/PulpConfig.cmake"
    if not provenance_path.is_file() or not config_path.is_file():
        parser.error("--sdk-prefix must contain sdk-provenance.json and PulpConfig.cmake")
    try:
        loaded_provenance = json.loads(provenance_path.read_text())
    except (OSError, ValueError) as exc:
        parser.error(f"invalid SDK provenance: {exc}")
    provenance = loaded_provenance if isinstance(loaded_provenance, dict) else {}
    source = (args.source_root or diagnostic.parent.parent).resolve()
    source_revision, source_clean = source_identity(source)
    provenance_failures = verify_sdk_provenance(sdk, provenance)
    if not source_clean:
        provenance_failures.append("consumer source is not an exact clean Git checkout")
    try:
        model_info = model_identity(model)
    except ValueError as exc:
        parser.error(str(exc))

    positive = run_diagnostic(diagnostic, model, args.blocks, False)
    negative = run_diagnostic(diagnostic, model, args.blocks, True)
    positive_fields = positive["fields"]
    negative_fields = negative["fields"]
    execution_failures = validate_execution(positive_fields, args.blocks)
    gpu_completions = parse_nonnegative_int(positive_fields, "gpu_inner_completions") or 0
    cpu_fallback = parse_nonnegative_int(positive_fields, "cpu_fallback") or 0
    transport_misses = parse_nonnegative_int(positive_fields, "transport_misses") or 0
    input_dropped = parse_nonnegative_int(positive_fields, "input_dropped")
    parity_failures = parse_nonnegative_int(positive_fields, "parity_failures")
    try:
        max_error = float(positive_fields.get("max_error", "nan"))
    except ValueError:
        max_error = float("nan")
    positive_pass = (
        positive["exit_code"] == 0
        and positive_fields.get("diagnostic_status") == "passed"
        and positive_fields.get("provider_available") == "1"
        and positive_fields.get("direct_model_parity_failures") == "0"
        and positive_fields.get("parity_failures") == "0"
        and not execution_failures
        and not provenance_failures
    )
    negative_pass = (
        negative["exit_code"] != 0
        and negative_fields.get("diagnostic_status") == "failed"
        and (parse_nonnegative_int(negative_fields, "direct_model_parity_failures") or 0) > 0
        and negative_fields.get("parity_failures") == "0"
    )

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if positive_pass and negative_pass else "fail_closed",
        "consumer": {
            "repository": "https://github.com/danielraffel/pulp-gpu-nam",
            "source_root": str(source),
            "source_revision": source_revision,
            "source_clean": source_clean,
            "diagnostic": str(diagnostic),
            "diagnostic_sha256": sha256(diagnostic),
        },
        "sdk": {
            "prefix": str(sdk),
            "version": provenance.get("sdk_version", "unavailable"),
            "source_git_sha": provenance.get("source_git_sha", "unavailable"),
            "source_git_ref": provenance.get("source_git_ref", "unavailable"),
            "provenance_sha256": sha256(provenance_path),
            "distribution_eligible": provenance.get("distribution_eligible") is True,
            "provenance_failures": provenance_failures,
            "gpu_capabilities": (provenance["gpu_audio"].get("capabilities", {})
                                 if isinstance(provenance.get("gpu_audio"), dict) and isinstance(provenance["gpu_audio"].get("capabilities"), dict) else {}),
        },
        "model": model_info,
        "provider": {
            "available": positive_fields.get("provider_available") == "1",
            "identity": positive_fields.get("backend", "unavailable"),
            "device": device_identity(),
        },
        "execution": {
            "block_size": 32,
            "lead_blocks": 4,
            "requested_blocks": args.blocks,
            "gpu_inner_completions": gpu_completions,
            "cpu_fallback": cpu_fallback,
            "transport_misses": transport_misses,
            "input_dropped": input_dropped if input_dropped is not None else "unavailable",
            "parity_failures": parity_failures if parity_failures is not None else "unavailable",
            "max_error": max_error if math.isfinite(max_error) else None,
            "observed_path": "gpu" if positive_pass else "unknown",
            "fallback_policy": "cpu",
            "fallback_primed": parse_nonnegative_int(positive_fields, "fallback_primed") == args.blocks,
            "validation_failures": execution_failures,
        },
        "unified_controls": {
            "source": "GpuNamProcessor::define_parameters",
            "cli": "gpu-nam-gpu-cpu-diagnostic",
            "mcp": "unsupported_surface: no MCP executor in downstream plugin",
            "controls": [
                {"id": 1, "name": "Input", "unit": "dB"},
                {"id": 2, "name": "Output", "unit": "dB"},
                {"id": 3, "name": "Mix", "unit": "%"},
                {"id": 4, "name": "Engine", "unit": "CPU|GPU|Auto"},
                {"id": 5, "name": "Bypass", "unit": "bool"},
                {"id": 6, "name": "Gate", "unit": "dB"},
                {"id": 7, "name": "Bass", "unit": "0..10"},
                {"id": 8, "name": "Middle", "unit": "0..10"},
                {"id": 9, "name": "Treble", "unit": "0..10"},
                {"id": 10, "name": "Noise Gate", "unit": "bool"},
                {"id": 11, "name": "EQ", "unit": "bool"},
                {"id": 12, "name": "Output Mode", "unit": "Raw|Normalized|Calibrated"},
                {"id": 13, "name": "Cal Level", "unit": "dB"},
                {"id": 14, "name": "Slim", "unit": "0..1"},
            ],
        },
        "tests": {
            "positive": {
                "status": "passed" if positive_pass else "failed",
                "exit_code": positive["exit_code"],
                "command": positive["command"],
            },
            "typed_negative_direct_output_corruption": {
                "status": "passed" if negative_pass else "failed",
                "exit_code": negative["exit_code"],
                "command": negative["command"],
                "expected": "diagnostic_status=failed with direct_model_parity_failures>0 and parity_failures=0",
            },
        },
        "docs": {
            "consumer_receipt": "docs/production-gpu-nam-receipt.md",
            "control_contract": "docs/nam-support.md",
            "pulp_gpu_audio_contract": "https://github.com/Generous-Corp/pulp/blob/main/docs/guides/gpu-audio-sdk.md",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": receipt["status"], "output": str(output)}, sort_keys=True))
    return 0 if receipt["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
