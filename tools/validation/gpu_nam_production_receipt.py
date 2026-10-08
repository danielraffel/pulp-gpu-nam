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
        provenance = json.loads(provenance_path.read_text())
    except (OSError, ValueError) as exc:
        parser.error(f"invalid SDK provenance: {exc}")
    if provenance.get("distribution_eligible") is not True:
        parser.error("SDK provenance is not distribution eligible")

    source = (args.source_root or diagnostic.parent.parent).resolve()
    try:
        model_info = model_identity(model)
    except ValueError as exc:
        parser.error(str(exc))

    positive = run_diagnostic(diagnostic, model, args.blocks, False)
    negative = run_diagnostic(diagnostic, model, args.blocks, True)
    positive_fields = positive["fields"]
    negative_fields = negative["fields"]
    positive_pass = (
        positive["exit_code"] == 0
        and positive_fields.get("diagnostic_status") == "passed"
        and positive_fields.get("provider_available") == "1"
        and positive_fields.get("direct_model_parity_failures") == "0"
        and positive_fields.get("parity_failures") == "0"
        and int(positive_fields.get("gpu_inner_completions", "0")) > 0
    )
    negative_pass = (
        negative["exit_code"] != 0
        and negative_fields.get("diagnostic_status") == "failed"
        and int(negative_fields.get("direct_model_parity_failures", "0")) > 0
        and negative_fields.get("parity_failures") == "0"
    )

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if positive_pass and negative_pass else "fail_closed",
        "consumer": {
            "repository": "https://github.com/danielraffel/pulp-gpu-nam",
            "source_root": str(source),
            "source_revision": git_value(source, "rev-parse", "HEAD"),
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
            "gpu_capabilities": provenance.get("gpu_audio", {}).get("capabilities", {}),
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
            "gpu_inner_completions": int(positive_fields.get("gpu_inner_completions", "0")),
            "cpu_fallback": int(positive_fields.get("cpu_fallback", "0")),
            "transport_misses": int(positive_fields.get("transport_misses", "0")),
            "input_dropped": int(positive_fields.get("input_dropped", "0")),
            "parity_failures": int(positive_fields.get("parity_failures", "-1")),
            "max_error": float(positive_fields.get("max_error", "nan")),
            "observed_path": "gpu" if positive_pass else "unknown",
            "fallback_policy": "cpu",
            "fallback_primed": int(positive_fields.get("fallback_primed", "0")) == args.blocks,
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
