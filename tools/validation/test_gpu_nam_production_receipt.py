#!/usr/bin/env python3
"""Unit tests for the fail-closed production receipt wrapper."""

import json
import os
import stat
import subprocess
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "gpu_nam_production_receipt.py"


def write_fake(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import sys
inject = '--inject-direct-output-error' in sys.argv
if inject:
    print('provider_available=1 backend=Fake GPU')
    print('direct_model_parity_failures=1 parity_failures=0')
    print('diagnostic_status=failed')
    raise SystemExit(3)
print('provider_available=1 backend=Fake GPU')
print('blocks=2 gpu_inner_completions=2 cpu_fallback=0 fallback_primed=2 transport_misses=0 input_dropped=0 parity_failures=0 max_error=0')
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
            str(tmp_path),
            "--blocks",
            "2",
            "--output",
            str(out),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


class ProductionReceiptTests(unittest.TestCase):
  def test_positive_and_typed_negative(self) -> None:
    with tempfile.TemporaryDirectory() as raw:
      tmp_path = Path(raw)
      self._positive_and_typed_negative(tmp_path)

  def _positive_and_typed_negative(self, tmp_path: Path) -> None:
    diagnostic = tmp_path / "diagnostic.py"
    write_fake(diagnostic)
    model = tmp_path / "example.nam"
    model.write_text('{"architecture":"WaveNet","sample_rate":48000}')
    sdk = tmp_path / "sdk"
    (sdk / "lib/cmake/Pulp").mkdir(parents=True)
    (sdk / "lib/cmake/Pulp/PulpConfig.cmake").write_text("# test\n")
    (sdk / "sdk-provenance.json").write_text(
        json.dumps({
            "distribution_eligible": True,
            "sdk_version": "test",
            "source_git_sha": "a" * 40,
            "source_git_ref": "test",
            "gpu_audio": {"capabilities": {"shared_provider": True}},
        })
    )
    out = tmp_path / "receipt.json"
    result = run_receipt(tmp_path, diagnostic, model, sdk, out)
    self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
    receipt = json.loads(out.read_text())
    self.assertEqual(receipt["schema"], "pulp.gpu-nam.production-receipt.v1")
    self.assertEqual(receipt["status"], "pass")
    self.assertEqual(receipt["provider"]["identity"], "Fake GPU")
    self.assertEqual(receipt["tests"]["positive"]["status"], "passed")
    self.assertEqual(receipt["tests"]["typed_negative_direct_output_corruption"]["status"], "passed")


  def test_fail_closed_when_provider_is_unavailable(self) -> None:
    with tempfile.TemporaryDirectory() as raw:
      self._provider_unavailable(Path(raw))

  def _provider_unavailable(self, tmp_path: Path) -> None:
    diagnostic = tmp_path / "diagnostic.py"
    diagnostic.write_text(
        "#!/usr/bin/env python3\nprint('provider_available=0 backend=unavailable')\nprint('diagnostic_status=provider_unavailable')\n"
    )
    diagnostic.chmod(diagnostic.stat().st_mode | stat.S_IXUSR)
    model = tmp_path / "example.nam"
    model.write_text('{"architecture":"WaveNet","sample_rate":48000}')
    sdk = tmp_path / "sdk"
    (sdk / "lib/cmake/Pulp").mkdir(parents=True)
    (sdk / "lib/cmake/Pulp/PulpConfig.cmake").write_text("# test\n")
    (sdk / "sdk-provenance.json").write_text(json.dumps({"distribution_eligible": True}))
    out = tmp_path / "receipt.json"
    result = subprocess.run(
        ["python3", str(SCRIPT), "--diagnostic", str(diagnostic), "--model", str(model),
         "--sdk-prefix", str(sdk), "--output", str(out)],
        text=True, capture_output=True, check=False,
    )
    self.assertEqual(result.returncode, 3)
    self.assertEqual(json.loads(out.read_text())["status"], "fail_closed")


if __name__ == "__main__":
    unittest.main()
