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
import re
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any


GPU_PLUGIN_TEST_NAMES = (
    "GPU NAM GPU engine reproduces the CPU engine",
    "GPU NAM GPU engine keeps stereo channels independent on the shared device",
    "GPU NAM switches Engine CPU->GPU->CPU live at fixed latency",
)
CAPABILITY_TEST_NAME = "gpu-nam-gpu-audio-capability-probe"
GPU_SKIP_MARKERS = (
    "gpu engine unavailable",
    "no gpu device; skipping",
)


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
    """Return only the canonical installed-SDK config.

    Recursive discovery can accidentally bind a receipt to a nested staging
    tree or a different SDK copied below the requested prefix.  The install
    contract is the CMake package path below the prefix, so require that exact
    path instead.
    """
    candidate = prefix / "lib" / "cmake" / "Pulp" / "PulpConfig.cmake"
    return candidate if candidate.is_file() else None


def file_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        record["sha256"] = sha256(path)
    return record


def executable_record(build_dir: Path, name: str) -> dict[str, Any]:
    candidates = [build_dir / "src" / name,
                  build_dir / "src" / "Release" / name]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    # A reused multi-config/single-config tree may contain two artifacts.  Do
    # not silently hash whichever happens to sort first.
    path = existing[0] if len(existing) == 1 else candidates[0]
    record = file_record(path)
    record["ambiguous"] = len(existing) > 1
    if len(existing) > 1:
        record["candidates"] = [str(candidate) for candidate in existing]
    return record


def ctest_inventory(build_dir: Path, log: Path) -> dict[str, Any]:
    return run([
        "ctest", "--test-dir", str(build_dir), "-C", "Release",
        "--show-only=json-v1",
    ], cwd=build_dir, log=log)


def inventory_names(inventory_log: Path) -> set[str]:
    try:
        data = json.loads(inventory_log.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    return {
        str(test.get("name"))
        for test in data.get("tests", [])
        if isinstance(test, dict) and test.get("name")
    }


def ctest_regex() -> str:
    names = (*GPU_PLUGIN_TEST_NAMES, CAPABILITY_TEST_NAME)
    return "^(" + "|".join(re.escape(name) for name in names) + ")$"


def ctest_has_gpu_skip(log: Path) -> bool:
    try:
        output = log.read_text(encoding="utf-8").lower()
    except OSError:
        return True
    return any(marker in output for marker in GPU_SKIP_MARKERS)


def ctest_selected_names_present(log: Path) -> bool:
    try:
        output = log.read_text(encoding="utf-8")
    except OSError:
        return False
    return all(name in output for name in (*GPU_PLUGIN_TEST_NAMES,
                                           CAPABILITY_TEST_NAME))


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
    if args.build_dir and build_dir.exists() and any(build_dir.iterdir()):
        parser.error(
            f"refusing reused non-empty build directory (receipt provenance): {build_dir}"
        )
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
    inventory = ctest_inventory(build_dir, log_dir / "ctest-inventory.json")
    inventory_names_seen = inventory_names(log_dir / "ctest-inventory.json")
    expected_names = set((*GPU_PLUGIN_TEST_NAMES, CAPABILITY_TEST_NAME))
    inventory_complete = expected_names.issubset(inventory_names_seen)

    ctest = run([
        "ctest", "--test-dir", str(build_dir), "-C", "Release",
        "-R", ctest_regex(), "--verbose", "--output-on-failure",
    ], cwd=source, log=log_dir / "ctest.log")

    config = find_pulp_config(prefix)
    model = source / "src" / "models" / "example.nam"
    executables = {
        name: executable_record(build_dir, name)
        for name in ("gpu-nam-plugin-test", "gpu-nam-gpu-audio-capability-probe")
    }
    source_status = git_value(source, "status", "--porcelain")
    executable_build_ok = all(
        item["exists"] and not item.get("ambiguous", False)
        for item in executables.values()
    )
    tests_executed = (
        inventory["exit_code"] == 0
        and inventory_complete
        and ctest["exit_code"] == 0
        and ctest_selected_names_present(log_dir / "ctest.log")
        and not ctest_has_gpu_skip(log_dir / "ctest.log")
    )
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
        "model": file_record(model),
        "build": {
            "directory": str(build_dir),
            "configure": configure,
            "build": build,
            "ctest_inventory": inventory,
            "ctest": ctest,
            "ctest_inventory_names": sorted(inventory_names_seen),
            "ctest_inventory_complete": inventory_complete,
            "ctest_selected_names_present": ctest_selected_names_present(
                log_dir / "ctest.log"
            ),
            "ctest_gpu_skip_detected": ctest_has_gpu_skip(log_dir / "ctest.log"),
        },
        "executables": executables,
        "result": {
            "installed_sdk_consumer": all(
                step["exit_code"] == 0 for step in (configure, build)
            ) and tests_executed and source_status == "" and config is not None and model.is_file()
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
