#!/usr/bin/env python3
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tools/validation/installed_sdk_consumer"

def run(*args, env):
    result = subprocess.run(args, text=True, capture_output=True, env=env)
    if result.returncode:
        raise AssertionError(f"{args} returned {result.returncode}\n{result.stdout}\n{result.stderr}")

def main():
    pulp = os.environ.get("GPU_NAM_PULP_SDK")
    vellum = os.environ.get("GPU_NAM_VELLUM_SDK")
    if not pulp or not vellum:
        raise SystemExit("GPU_NAM_PULP_SDK and GPU_NAM_VELLUM_SDK are required")
    with tempfile.TemporaryDirectory() as directory:
        build = str(Path(directory) / "build")
        env = {**os.environ, "CMAKE_PREFIX_PATH": os.pathsep.join((pulp, vellum))}
        run("cmake", "-S", str(FIXTURE), "-B", build, "-DCMAKE_BUILD_TYPE=Release", env=env)
        run("cmake", "--build", build, "--parallel", "2", env=env)

if __name__ == "__main__":
    main()
