from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "manifests"
SCRIPT = ROOT / "scripts" / "build_data_quality_report.py"


class CliSmokeTests(unittest.TestCase):
    def test_help(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--release-manifest", result.stdout)

    def test_fixture_report_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_json = Path(temp_dir) / "summary.json"
            output_markdown = Path(temp_dir) / "report.md"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--source-manifest",
                    str(FIXTURES / "source.csv"),
                    "--release-manifest",
                    str(FIXTURES / "release.csv"),
                    "--train-manifest",
                    str(FIXTURES / "train.csv"),
                    "--test-manifest",
                    str(FIXTURES / "test.csv"),
                    "--event-column",
                    "event_id",
                    "--expected-source-events",
                    "5",
                    "--expected-release-events",
                    "4",
                    "--output-json",
                    str(output_json),
                    "--output-markdown",
                    str(output_markdown),
                    "--fail-on-issues",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            report = json.loads(output_json.read_text())

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["split_audit"]["event_overlap_count"], 0)
        self.assertEqual(report["error_count"], 0)


if __name__ == "__main__":
    unittest.main()
