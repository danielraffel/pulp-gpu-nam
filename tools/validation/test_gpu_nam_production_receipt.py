#!/usr/bin/env python3
"""Unit tests for the fail-closed production receipt wrapper."""

import json
import hashlib
import os
import stat
import subprocess
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "gpu_nam_production_receipt.py"


def write_fake(path: Path, *, input_dropped: int = 0, fallback_primed: int = 2,
               max_error: str = "0", gpu_inner_completions: int = 2) -> None:
    path.write_text(
        f"""#!/usr/bin/env python3
import sys
inject = '--inject-direct-output-error' in sys.argv
if inject:
    print('provider_available=1 backend=Fake GPU')
    print('direct_model_parity_failures=1 parity_failures=0')
    print('diagnostic_status=failed')
    raise SystemExit(3)
print('provider_available=1 backend=Fake GPU')
print('blocks=2 gpu_inner_completions={gpu_inner_completions} cpu_fallback=0 fallback_primed={fallback_primed} transport_misses=0 input_dropped={input_dropped} parity_failures=0 max_error={max_error}')
print('direct_model_parity_failures=0 diagnostic_status=passed')
"""
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def run_receipt(tmp_path: Path, diagnostic: Path, model: Path, sdk: Path, out: Path):
    return subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--diagnostic",
            str(diagnostic),
            "--model",
            str(model),
            "--sdk-prefix",
            str(sdk),
            "--source-root",
            str(tmp_path / "source"),
            "--blocks",
            "2",
            "--output",
            str(out),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def init_clean_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    marker = source / "source.txt"
    marker.write_text("clean\n")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(source), "add", "source.txt"], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "fixture"], check=True)
    return source


def write_release_sdk(tmp_path: Path, *, integrity: bool = True) -> Path:
    sdk = tmp_path / "sdk"
    config = sdk / "lib/cmake/Pulp/PulpConfig.cmake"
    config.parent.mkdir(parents=True)
    config.write_text("# test\n")
    members = {}
    for name in ("include/pulp/view/widget_bridge.hpp", "lib/libpulp-view-script.a", "lib/libpulp-gpu-audio.a"):
        member = sdk / name
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_text("fixture\n")
        members[name] = member
    marker = {
        "schema": "pulp.sdk-provenance.v1",
        "kind": "release",
        "profile": "official-release",
        "distribution_eligible": True,
        "sdk_version": "1.0.0",
        "source_git_ref": "v1.0.0",
        "source_git_sha": "a" * 40,
        "source_git_dirty": False,
        "platform": "darwin-arm64",
        "build_type": "Release",
        "gpu_audio": {
            "schema": "pulp.sdk-gpu-audio-capabilities.v1",
            "capabilities": {key: True for key in ("shared_provider", "shared_convolver", "exact_provider_proof")},
            "files": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in {"lib/cmake/Pulp/PulpConfig.cmake": config, "lib/libpulp-gpu-audio.a": members["lib/libpulp-gpu-audio.a"]}.items()},
        },
    }
    if integrity:
        marker["integrity"] = {
            "schema": "pulp.sdk-integrity.v1",
            "algorithm": "sha256",
            "files": {
                name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in members.items() if name != "lib/libpulp-gpu-audio.a"
            },
        }
    (sdk / "sdk-provenance.json").write_text(json.dumps(marker))
    return sdk


class ProductionReceiptTests(unittest.TestCase):
  def test_positive_and_typed_negative(self) -> None:
    with tempfile.TemporaryDirectory() as raw:
      tmp_path = Path(raw)
      init_clean_source(tmp_path)
      self._positive_and_typed_negative(tmp_path)

  def _positive_and_typed_negative(self, tmp_path: Path) -> None:
    diagnostic = tmp_path / "diagnostic.py"
    write_fake(diagnostic)
    model = tmp_path / "example.nam"
    model.write_text('{"architecture":"WaveNet","sample_rate":48000}')
    sdk = tmp_path / "sdk"
    sdk = write_release_sdk(tmp_path)
    out = tmp_path / "receipt.json"
    result = run_receipt(tmp_path, diagnostic, model, sdk, out)
    self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
    receipt = json.loads(out.read_text())
    self.assertEqual(receipt["schema"], "pulp.gpu-nam.production-receipt.v1")
    self.assertEqual(receipt["status"], "pass")
    self.assertEqual(receipt["provider"]["identity"], "Fake GPU")
    self.assertEqual(receipt["tests"]["positive"]["status"], "passed")
    self.assertEqual(receipt["tests"]["typed_negative_direct_output_corruption"]["status"], "passed")

  def test_execution_integrity_is_fail_closed(self) -> None:
    for kwargs, expected in (
        ({"input_dropped": 1}, "input_dropped"),
        ({"fallback_primed": 1}, "fallback_primed"),
        ({"max_error": "nan"}, "max_error"),
        ({"gpu_inner_completions": 0}, "gpu_inner_completions"),
    ):
      with self.subTest(expected=expected), tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        init_clean_source(tmp_path)
        diagnostic = tmp_path / "diagnostic.py"
        write_fake(diagnostic, **kwargs)
        model = tmp_path / "example.nam"
        model.write_text('{"architecture":"WaveNet","sample_rate":48000}')
        out = tmp_path / "receipt.json"
        result = run_receipt(tmp_path, diagnostic, model, write_release_sdk(tmp_path), out)
        self.assertEqual(result.returncode, 3, result.stderr + result.stdout)
        receipt = json.loads(out.read_text())
        self.assertEqual(receipt["status"], "fail_closed")
        self.assertTrue(any(expected in item for item in receipt["execution"]["validation_failures"]))

  def test_missing_provider_identity_is_fail_closed(self) -> None:
    with tempfile.TemporaryDirectory() as raw:
      tmp_path = Path(raw)
      init_clean_source(tmp_path)
      diagnostic = tmp_path / "diagnostic.py"
      write_fake(diagnostic)
      diagnostic.write_text(diagnostic.read_text().replace("backend=Fake GPU", "backend= "))
      model = tmp_path / "example.nam"
      model.write_text('{"architecture":"WaveNet","sample_rate":48000}')
      out = tmp_path / "receipt.json"
      result = run_receipt(tmp_path, diagnostic, model, write_release_sdk(tmp_path), out)
      self.assertEqual(result.returncode, 3)
      self.assertIn("provider backend identity", " ".join(json.loads(out.read_text())["execution"]["validation_failures"]))

  def test_sdk_integrity_and_source_cleanliness_are_fail_closed(self) -> None:
    with tempfile.TemporaryDirectory() as raw:
      tmp_path = Path(raw)
      source = init_clean_source(tmp_path)
      diagnostic = tmp_path / "diagnostic.py"
      write_fake(diagnostic)
      model = tmp_path / "example.nam"
      model.write_text('{"architecture":"WaveNet","sample_rate":48000}')
      out = tmp_path / "receipt.json"
      result = run_receipt(tmp_path, diagnostic, model, write_release_sdk(tmp_path, integrity=False), out)
      self.assertEqual(result.returncode, 3)
      self.assertTrue(any("integrity" in item for item in json.loads(out.read_text())["sdk"]["provenance_failures"]))

    with tempfile.TemporaryDirectory() as raw:
      tmp_path = Path(raw)
      init_clean_source(tmp_path)
      diagnostic = tmp_path / "diagnostic.py"
      write_fake(diagnostic)
      model = tmp_path / "example.nam"
      model.write_text('{"architecture":"WaveNet","sample_rate":48000}')
      sdk = write_release_sdk(tmp_path)
      (sdk / "lib/cmake/Pulp/PulpConfig.cmake").write_text("tampered\n")
      out = tmp_path / "receipt.json"
      result = run_receipt(tmp_path, diagnostic, model, sdk, out)
      self.assertEqual(result.returncode, 3)
      self.assertIn("integrity mismatch", " ".join(json.loads(out.read_text())["sdk"]["provenance_failures"]))

    with tempfile.TemporaryDirectory() as raw:
      tmp_path = Path(raw)
      source = init_clean_source(tmp_path)
      (source / "dirty.txt").write_text("uncommitted\n")
      diagnostic = tmp_path / "diagnostic.py"
      write_fake(diagnostic)
      model = tmp_path / "example.nam"
      model.write_text('{"architecture":"WaveNet","sample_rate":48000}')
      out = tmp_path / "receipt.json"
      result = run_receipt(tmp_path, diagnostic, model, write_release_sdk(tmp_path), out)
      self.assertEqual(result.returncode, 3)
      self.assertIn("consumer source", " ".join(json.loads(out.read_text())["sdk"]["provenance_failures"]))


  def test_malformed_and_incomplete_evidence_is_fail_closed(self) -> None:
    for mutation in ("primed", "negative_count", "gpu_audio", "capabilities", "integrity_subset"):
      with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        init_clean_source(root)
        diagnostic = root / "diagnostic.py"
        write_fake(diagnostic)
        if mutation == "primed":
          diagnostic.write_text(diagnostic.read_text().replace("fallback_primed=2", "fallback_primed=invalid"))
        if mutation == "negative_count":
          diagnostic.write_text(diagnostic.read_text().replace("direct_model_parity_failures=1", "direct_model_parity_failures=invalid"))
        model = root / "example.nam"
        model.write_text('{"architecture":"WaveNet","sample_rate":48000}')
        sdk = write_release_sdk(root)
        path = sdk / "sdk-provenance.json"
        marker = json.loads(path.read_text())
        if mutation == "gpu_audio": marker["gpu_audio"] = []
        if mutation == "capabilities": marker["gpu_audio"]["capabilities"] = []
        if mutation == "integrity_subset":
          marker["integrity"]["files"] = {"lib/cmake/Pulp/PulpConfig.cmake": hashlib.sha256((sdk / "lib/cmake/Pulp/PulpConfig.cmake").read_bytes()).hexdigest()}
        path.write_text(json.dumps(marker))
        out = root / "receipt.json"
        result = run_receipt(root, diagnostic, model, sdk, out)
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertEqual(json.loads(out.read_text())["status"], "fail_closed")

  def test_fail_closed_when_provider_is_unavailable(self) -> None:
    with tempfile.TemporaryDirectory() as raw:
      self._provider_unavailable(Path(raw))

  def _provider_unavailable(self, tmp_path: Path) -> None:
    init_clean_source(tmp_path)
    diagnostic = tmp_path / "diagnostic.py"
    diagnostic.write_text(
        "#!/usr/bin/env python3\nprint('provider_available=0 backend=unavailable')\nprint('diagnostic_status=provider_unavailable')\n"
    )
    diagnostic.chmod(diagnostic.stat().st_mode | stat.S_IXUSR)
    model = tmp_path / "example.nam"
    model.write_text('{"architecture":"WaveNet","sample_rate":48000}')
    sdk = tmp_path / "sdk"
    sdk = write_release_sdk(tmp_path)
    out = tmp_path / "receipt.json"
    result = subprocess.run(
        ["python3", str(SCRIPT), "--diagnostic", str(diagnostic), "--model", str(model),
         "--sdk-prefix", str(sdk), "--source-root", str(tmp_path / "source"), "--output", str(out)],
        text=True, capture_output=True, check=False,
    )
    self.assertEqual(result.returncode, 3)
    self.assertEqual(json.loads(out.read_text())["status"], "fail_closed")


if __name__ == "__main__":
    unittest.main()
