#!/usr/bin/env python3
"""Fake-driver tests for the installed-SDK receipt contract.

These tests deliberately replace CMake and CTest with tiny scripts.  They
exercise receipt validation without compiling Pulp, Dawn, or the plugin.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("installed_sdk_gpu_audio_receipt.py")
TEST_NAMES = [
    "GPU NAM GPU engine reproduces the CPU engine",
    "GPU NAM GPU engine keeps stereo channels independent on the shared device",
    "GPU NAM switches Engine CPU->GPU->CPU live at fixed latency",
    "gpu-nam-gpu-audio-capability-probe",
]


class InstalledSdkReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.source = root / "source"
        (self.source / "src" / "models").mkdir(parents=True)
        (self.source / "CMakeLists.txt").write_text("project(fake)\n")
        (self.source / "src" / "models" / "example.nam").write_text("model\n")
        self.sdk = root / "sdk"
        (self.sdk / "lib" / "cmake" / "Pulp").mkdir(parents=True)
        (self.sdk / "lib" / "cmake" / "Pulp" / "PulpConfig.cmake").write_text(
            "# fake installed SDK\n"
        )
        self.bin = root / "bin"
        self.bin.mkdir()
        self._write_cmake()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_cmake(self) -> None:
        path = self.bin / "cmake"
        path.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, sys\n"
            "args = sys.argv[1:]\n"
            "if '--build' in args:\n"
            "  build = pathlib.Path(args[args.index('--build') + 1])\n"
            "  (build / 'src').mkdir(parents=True, exist_ok=True)\n"
            "  for name in ('gpu-nam-plugin-test', 'gpu-nam-gpu-audio-capability-probe'):\n"
            "    (build / 'src' / name).write_text('fake executable')\n"
            "print('fake cmake')\n"
        )
        path.chmod(0o755)

    def _write_ctest(self, *, skip: bool = False, empty: bool = False) -> None:
        path = self.bin / "ctest"
        names = [] if empty else TEST_NAMES
        output = ""
        if not empty:
            output = "\n".join(
                f"{index}/4 Test #{index}: {name} Passed"
                for index, name in enumerate(names, 1)
            )
            if skip:
                output += "\nGPU engine unavailable — skipping GPU-vs-CPU test"
        path.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            f"names = {names!r}\n"
            "if '--show-only=json-v1' in sys.argv:\n"
            "  print(json.dumps({'tests': [{'name': name} for name in names]}))\n"
            "else:\n"
            f"  print({output!r})\n"
            "  print('100% tests passed, 4 tests passed out of 4')\n"
        )
        path.chmod(0o755)

    def _run(self, *, ctest_mode: str = "good", config: bool = True) -> tuple[int, dict]:
        if ctest_mode == "skip":
            self._write_ctest(skip=True)
        elif ctest_mode == "empty":
            self._write_ctest(empty=True)
        else:
            self._write_ctest()
        if not config:
            (self.sdk / "lib" / "cmake" / "Pulp" / "PulpConfig.cmake").unlink()
        root = Path(self.temp.name)
        output = root / "receipt.json"
        build = root / "build"
        env = dict(os.environ)
        env["PATH"] = f"{self.bin}{os.pathsep}{env['PATH']}"
        proc = subprocess.run(
            [
                "python3", str(SCRIPT), "--sdk-prefix", str(self.sdk),
                "--source", str(self.source), "--build-dir", str(build),
                "--output", str(output), "--cmake", str(self.bin / "cmake"),
            ],
            text=True, capture_output=True, env=env, check=False,
        )
        return proc.returncode, json.loads(output.read_text())

    def test_good_receipt_records_model_and_exact_tests(self) -> None:
        rc, receipt = self._run()
        self.assertEqual(rc, 0)
        self.assertTrue(receipt["result"]["installed_sdk_consumer"])
        self.assertTrue(receipt["model"]["exists"])
        self.assertEqual(receipt["build"]["ctest_inventory_names"], sorted(TEST_NAMES))

    def test_empty_ctest_inventory_fails_closed(self) -> None:
        rc, receipt = self._run(ctest_mode="empty")
        self.assertNotEqual(rc, 0)
        self.assertFalse(receipt["result"]["installed_sdk_consumer"])
        self.assertFalse(receipt["build"]["ctest_inventory_complete"])

    def test_gpu_skip_with_green_ctest_fails_closed(self) -> None:
        rc, receipt = self._run(ctest_mode="skip")
        self.assertNotEqual(rc, 0)
        self.assertTrue(receipt["build"]["ctest_gpu_skip_detected"])

    def test_missing_canonical_sdk_config_fails_closed(self) -> None:
        rc, receipt = self._run(config=False)
        self.assertNotEqual(rc, 0)
        self.assertIsNone(receipt["pulp_sdk"]["config"])


if __name__ == "__main__":
    unittest.main()
