#!/usr/bin/env python3
"""Check the CPU engine's rendered output against stored per-platform hashes.

Runs gpu-nam-golden-render, hashes every output file with SHA-256 and compares
the hashes with render-hashes.json. Any changed sample fails the check, so a
speed change must prove it leaves the output untouched.

Hashes are stored per platform (OS, CPU architecture). The same source can
round differently on different platforms: arm64 compilers contract a*b+c into a
fused multiply-add while x86-64 without FMA does not, and the exact tanh mode
calls the platform's libm. So each platform is compared only with itself, and
the report notes where platforms happen to agree.

  render_hashes.py --render build-cpu/gpu-nam-golden-render            # check
  render_hashes.py --render build-cpu/gpu-nam-golden-render --update   # record
"""
import argparse
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_GOLDEN = HERE / "render-hashes.json"
DEFAULT_MODELS = HERE.parent.parent / "src" / "models"


def platform_key():
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    return f"{sys.platform}-{machine}"


def render(binary, models):
    with tempfile.TemporaryDirectory() as out:
        subprocess.run([str(binary), str(models), out], check=True)
        files = sorted(pathlib.Path(out).glob("*.f32"))
        return {f.stem: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render", required=True, help="path to gpu-nam-golden-render")
    ap.add_argument("--models", default=str(DEFAULT_MODELS))
    ap.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    ap.add_argument("--platform", default=platform_key())
    ap.add_argument("--update", action="store_true", help="record this platform's hashes")
    a = ap.parse_args()

    got = render(a.render, a.models)
    if not got:
        print("error: the render produced no files", file=sys.stderr)
        return 2
    golden_path = pathlib.Path(a.golden)
    golden = json.loads(golden_path.read_text()) if golden_path.exists() else {}

    if a.update:
        golden[a.platform] = got
        golden_path.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
        print(f"recorded {len(got)} hashes for {a.platform} in {golden_path}")
        return 0

    want = golden.get(a.platform)
    if want is None:
        print(f"error: no stored hashes for {a.platform}. Record them on an unchanged main with --update.",
              file=sys.stderr)
        print(json.dumps({a.platform: got}, indent=2, sort_keys=True))
        return 1

    changed = sorted(k for k in want if got.get(k) != want[k])
    missing = sorted(k for k in want if k not in got)
    extra = sorted(k for k in got if k not in want)
    for other, hashes in sorted(golden.items()):
        if other != a.platform:
            same = sum(1 for k, v in got.items() if hashes.get(k) == v)
            print(f"note: {same}/{len(got)} renders are byte-identical to {other}")
    if changed or missing or extra:
        for k in changed:
            print(f"CHANGED  {k}" + ("" if k in got else " (not rendered)"))
        for k in extra:
            print(f"NEW      {k} (no stored hash)")
        print(f"FAIL: {len(changed)} of {len(want)} renders differ from the stored hashes for {a.platform}")
        return 1
    print(f"OK: {len(got)} renders byte-identical to the stored hashes for {a.platform}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
