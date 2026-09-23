#!/usr/bin/env python3
"""Build and exercise GPU NAM against an installed Pulp GPU SDK.

This is deliberately a consumer-side proof.  It proves that GPU NAM's public
targets and existing lifecycle/fallback tests work with the installed SDK.  It
does not infer private shared-memory-provider adoption from target presence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, cwd: Path, log: Path) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True,
                          check=False)
    output = proc.stdout + proc.stderr
    log.write_text(output, encoding="utf-8")
    return {
        "command": [str(part) for part in command],
        "command_line": shlex.join(str(part) for part in command),
        "exit_code": proc.returncode,
        "duration_seconds": round(time.time() - started, 3),
        "log": str(log),
        "log_sha256": sha256(log),
    }


def git_value(source: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=source, text=True,
                          capture_output=True, check=False)
    if proc.returncode:
        return ""
    return proc.stdout.strip()


def find_pulp_config(prefix: Path) -> Path | None:
    matches = sorted(prefix.rglob("PulpConfig.cmake"))
    return matches[0] if matches else None


def executable_record(build_dir: Path, name: str) -> dict[str, Any]:
    candidates = [build_dir / "src" / name,
                  build_dir / "src" / "Release" / name]
    path = next((candidate for candidate in candidates if candidate.is_file()),
                candidates[0])
    record: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        record["sha256"] = sha256(path)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-prefix", type=Path, required=True)
    parser.add_argument("--source", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cmake", default="cmake")
    args = parser.parse_args()

    source = args.source.resolve()
    prefix = args.sdk_prefix.resolve()
    if not (source / "CMakeLists.txt").is_file():
        parser.error(f"not a GPU NAM source tree: {source}")
    if not prefix.is_dir():
        parser.error(f"SDK prefix does not exist: {prefix}")

    build_dir = (args.build_dir or
                 Path(tempfile.mkdtemp(prefix="gpu-nam-installed-sdk-"))).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    log_dir = build_dir / "receipt-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    configure = run([
        args.cmake, "-S", str(source), "-B", str(build_dir),
        "-DGPU_NAM_USE_INSTALLED_PULP=ON",
        "-DGPU_NAM_BUILD_TESTS=ON",
        "-DGPU_NAM_BUILD_GPU_AUDIO_CAPABILITY_PROBE=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_PREFIX_PATH={prefix}",
    ], cwd=source, log=log_dir / "configure.log")

    build = run([
        args.cmake, "--build", str(build_dir), "--config", "Release",
        "--target", "gpu-nam-plugin-test",
        "gpu-nam-gpu-audio-capability-probe",
    ], cwd=source, log=log_dir / "build.log")

    # Follow compilation with ctest so the receipt records actual test
    # execution rather than compilation alone.
    ctest = run([
        "ctest", "--test-dir", str(build_dir), "-C", "Release",
        "-R", "^(gpu-nam-plugin-test|gpu-nam-gpu-audio-capability-probe)$",
        "--output-on-failure",
    ], cwd=source, log=log_dir / "ctest.log")

    config = find_pulp_config(prefix)
    model = source / "models" / "example.nam"
    executables = {
        name: executable_record(build_dir, name)
        for name in ("gpu-nam-plugin-test", "gpu-nam-gpu-audio-capability-probe")
    }
    source_status = git_value(source, "status", "--porcelain")
    executable_build_ok = all(item["exists"] for item in executables.values())
    receipt = {
        "schema": "pulp.gpu-nam.installed-sdk-receipt.v1",
        "source": {
            "path": str(source),
            "git_commit": git_value(source, "rev-parse", "HEAD"),
            "git_origin": git_value(source, "remote", "get-url", "origin"),
            "git_status": source_status,
        },
        "pulp_sdk": {
            "prefix": str(prefix),
            "config": str(config) if config else None,
            "config_sha256": sha256(config) if config else None,
            "provider_identity": "reported only by the public capability probe;"
                                 " shared provider adoption is not inferred",
        },
        "model": executable_record(model),
        "build": {
            "directory": str(build_dir),
            "configure": configure,
            "build": build,
            "ctest": ctest,
        },
        "executables": executables,
        "result": {
            "installed_sdk_consumer": all(
                step["exit_code"] == 0 for step in (configure, build, ctest)
            ) and source_status == "" and config is not None and model.is_file()
            and executable_build_ok,
            "shared_provider_adoption": "unproven",
        },
    }
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "receipt": str(output),
        "installed_sdk_consumer": receipt["result"]["installed_sdk_consumer"],
        "shared_provider_adoption": receipt["result"]["shared_provider_adoption"],
    }, sort_keys=True))
    return 0 if receipt["result"]["installed_sdk_consumer"] else 1


if __name__ == "__main__":
    sys.exit(main())
