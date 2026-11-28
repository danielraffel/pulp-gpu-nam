#!/usr/bin/env python3
"""Contract for the macOS CI workflow: its status context name and token scope."""

import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MACOS = (ROOT / ".github" / "workflows" / "macos.yml").read_text(encoding="utf-8")


class MacosWorkflowTests(unittest.TestCase):
    def test_single_job_is_named_macos(self):
        # A ruleset requires the job name as the status context.
        jobs = MACOS.split("\njobs:\n", 1)[1]
        self.assertEqual(re.findall(r"(?m)^  ([A-Za-z0-9_-]+):\s*$", jobs), ["macos"])

    def test_untrusted_pull_requests_get_read_only_token(self):
        self.assertNotIn("pull_request_target", MACOS)
        self.assertIn("permissions:\n  contents: read\n", MACOS)


if __name__ == "__main__":
    unittest.main()
