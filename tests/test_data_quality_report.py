from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.build_data_quality_report import build_quality_report, render_markdown


FIXTURES = Path(__file__).parent / "fixtures" / "manifests"


class DataQualityReportTests(unittest.TestCase):
    def test_clean_fixture_report(self) -> None:
        report = build_quality_report(
            source_manifest=FIXTURES / "source.csv",
            release_manifest=FIXTURES / "release.csv",
            train_manifest=FIXTURES / "train.csv",
            test_manifest=FIXTURES / "test.csv",
            event_column="event_id",
            required_columns=("latitude", "longitude", "event_time"),
            expected_source_events=5,
            expected_release_events=4,
        )

        self.assertEqual(report["error_count"], 0)
        self.assertEqual(report["filtering"]["removed_unique_events"], 1)
        self.assertEqual(report["split_audit"]["event_overlap_count"], 0)
        self.assertEqual(
            report["manifests"]["release"]["sensor_row_counts"],
            {"EMIT": 1, "L89": 1, "S2": 3, "S5P": 1},
        )
        self.assertIn("No manifest-level validation issues", render_markdown(report))

    def test_overlap_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            overlap_path = Path(temp_dir) / "overlap.csv"
            with (FIXTURES / "release.csv").open(newline="", encoding="utf-8") as src:
                rows = list(csv.DictReader(src))
                fields = list(rows[0])
            with overlap_path.open("w", newline="", encoding="utf-8") as dst:
                writer = csv.DictWriter(dst, fieldnames=fields)
                writer.writeheader()
                writer.writerow(rows[1])
                writer.writerow(rows[2])

            report = build_quality_report(
                release_manifest=FIXTURES / "release.csv",
                train_manifest=FIXTURES / "train.csv",
                test_manifest=overlap_path,
                event_column="event_id",
            )

        self.assertEqual(report["split_audit"]["event_overlap_count"], 1)
        self.assertTrue(any(issue["code"] == "event_leakage" for issue in report["issues"]))


if __name__ == "__main__":
    unittest.main()
