#!/usr/bin/env python3
"""Build a manifest-level MethaneUnion data quality report.

The default audit reads CSV metadata only. File existence checks are opt-in so
the report can run without downloading the released dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EVENT_COLUMN_CANDIDATES = ("event_id", "plume_id")
AVAILABLE_SENSOR_COLUMN_CANDIDATES = ("available_sensor", "available_sensors")
LATITUDE_COLUMN_CANDIDATES = ("latitude", "latitude_center", "query_latitude")
LONGITUDE_COLUMN_CANDIDATES = ("longitude", "longitude_center", "query_longitude")
DATETIME_COLUMN_CANDIDATES = ("event_time", "datetime", "query_time")
SENSOR_PREFIXES = {
    "S2": ("s2_", "sentinel2_", "sentinel_2_"),
    "L89": ("l89_", "landsat_", "landsat8_", "landsat9_"),
    "EMIT": ("emit_",),
    "S5P": ("s5p_", "sentinel5p_", "sentinel_5p_"),
}


def _nonempty(value: object) -> bool:
    text = str(value or "").strip()
    return bool(text) and text.lower() not in {"nan", "none", "null"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        return fields, list(reader)


def _choose_column(
    fields: Iterable[str], explicit: str | None, candidates: Iterable[str]
) -> str | None:
    field_set = set(fields)
    if explicit:
        return explicit if explicit in field_set else None
    return next((name for name in candidates if name in field_set), None)


def _normalize_sensor(token: str) -> str | None:
    normalized = re.sub(r"[^A-Z0-9]", "", token.upper())
    aliases = {
        "S2": "S2",
        "SENTINEL2": "S2",
        "L8": "L89",
        "L9": "L89",
        "L89": "L89",
        "LANDSAT": "L89",
        "LANDSAT8": "L89",
        "LANDSAT9": "L89",
        "EMIT": "EMIT",
        "S5P": "S5P",
        "SENTINEL5P": "S5P",
    }
    return aliases.get(normalized)


def _sensors_for_row(
    row: dict[str, str], fields: list[str], available_sensor_column: str | None
) -> list[str]:
    sensors: set[str] = set()
    if available_sensor_column and _nonempty(row.get(available_sensor_column)):
        for token in re.findall(r"[A-Za-z0-9_-]+", row[available_sensor_column]):
            sensor = _normalize_sensor(token)
            if sensor:
                sensors.add(sensor)

    if not sensors:
        for field in fields:
            lowered = field.lower()
            if not lowered.endswith("_path") or not _nonempty(row.get(field)):
                continue
            for sensor, prefixes in SENSOR_PREFIXES.items():
                if lowered.startswith(prefixes):
                    sensors.add(sensor)
                    break

    return sorted(sensors)


def _valid_datetime(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    normalized = text.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(normalized)
        return True
    except ValueError:
        try:
            datetime.fromisoformat(normalized.split(" ", 1)[0])
            return True
        except ValueError:
            return False


def _path_audit(
    rows: list[dict[str, str]],
    fields: list[str],
    manifest_path: Path,
    data_root: Path | None,
) -> dict[str, Any]:
    path_columns = [field for field in fields if field.lower().endswith("_path")]
    values = {
        str(row.get(field) or "").strip()
        for row in rows
        for field in path_columns
        if _nonempty(row.get(field))
    }
    missing: list[str] = []
    for value in sorted(values):
        path = Path(value)
        if not path.is_absolute():
            path = (data_root or manifest_path.parent) / path
        if not path.exists():
            missing.append(value)
    return {
        "path_columns": path_columns,
        "unique_nonempty_paths": len(values),
        "missing_paths": len(missing),
        "missing_path_examples": missing[:20],
    }


def summarize_manifest(
    path: Path,
    *,
    event_column: str | None = None,
    available_sensor_column: str | None = None,
    required_columns: Iterable[str] = (),
    verify_files: bool = False,
    data_root: Path | None = None,
) -> tuple[dict[str, Any], set[str], list[dict[str, str]]]:
    fields, rows = _read_csv(path)
    issues: list[dict[str, str]] = []
    selected_event_column = _choose_column(
        fields, event_column, EVENT_COLUMN_CANDIDATES
    )
    selected_sensor_column = _choose_column(
        fields, available_sensor_column, AVAILABLE_SENSOR_COLUMN_CANDIDATES
    )
    latitude_column = _choose_column(fields, None, LATITUDE_COLUMN_CANDIDATES)
    longitude_column = _choose_column(fields, None, LONGITUDE_COLUMN_CANDIDATES)
    datetime_column = _choose_column(fields, None, DATETIME_COLUMN_CANDIDATES)

    if event_column and selected_event_column is None:
        issues.append(
            {
                "severity": "error",
                "code": "requested_event_column_missing",
                "detail": f"{event_column!r} is not present in {path}",
            }
        )
    elif selected_event_column is None:
        issues.append(
            {
                "severity": "error",
                "code": "event_column_not_detected",
                "detail": f"Expected one of {EVENT_COLUMN_CANDIDATES} in {path}",
            }
        )

    missing_required = sorted(set(required_columns) - set(fields))
    if missing_required:
        issues.append(
            {
                "severity": "error",
                "code": "required_columns_missing",
                "detail": ", ".join(missing_required),
            }
        )

    event_values = [
        str(row.get(selected_event_column, "")).strip()
        for row in rows
        if selected_event_column
    ]
    missing_event_rows = sum(not _nonempty(value) for value in event_values)
    events = {value for value in event_values if _nonempty(value)}
    event_frequency = Counter(value for value in event_values if _nonempty(value))
    events_with_multiple_rows = sum(count > 1 for count in event_frequency.values())
    if missing_event_rows:
        issues.append(
            {
                "severity": "error",
                "code": "missing_event_ids",
                "detail": f"{missing_event_rows} rows have no event identifier",
            }
        )

    sensor_counts: Counter[str] = Counter()
    availability_patterns: Counter[str] = Counter()
    for row in rows:
        sensors = _sensors_for_row(row, fields, selected_sensor_column)
        sensor_counts.update(sensors)
        availability_patterns["+".join(sensors) if sensors else "none"] += 1

    missing_coordinates = 0
    invalid_coordinates = 0
    if latitude_column and longitude_column:
        for row in rows:
            lat_raw = str(row.get(latitude_column, "")).strip()
            lon_raw = str(row.get(longitude_column, "")).strip()
            if not lat_raw or not lon_raw:
                missing_coordinates += 1
                continue
            try:
                lat = float(lat_raw)
                lon = float(lon_raw)
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    invalid_coordinates += 1
            except ValueError:
                invalid_coordinates += 1

    missing_datetimes = 0
    invalid_datetimes = 0
    if datetime_column:
        for row in rows:
            raw = str(row.get(datetime_column, "")).strip()
            if not raw:
                missing_datetimes += 1
            elif not _valid_datetime(raw):
                invalid_datetimes += 1

    if invalid_coordinates:
        issues.append(
            {
                "severity": "error",
                "code": "invalid_coordinates",
                "detail": f"{invalid_coordinates} rows contain invalid coordinates",
            }
        )
    if invalid_datetimes:
        issues.append(
            {
                "severity": "error",
                "code": "invalid_datetimes",
                "detail": f"{invalid_datetimes} rows contain invalid datetimes",
            }
        )

    summary: dict[str, Any] = {
        "path": str(path),
        "sha256": _sha256(path),
        "columns": fields,
        "row_count": len(rows),
        "event_column": selected_event_column,
        "unique_event_count": len(events) if selected_event_column else None,
        "missing_event_id_rows": missing_event_rows,
        "events_with_multiple_rows": events_with_multiple_rows,
        "available_sensor_column": selected_sensor_column,
        "sensor_row_counts": dict(sorted(sensor_counts.items())),
        "availability_patterns": dict(
            sorted(availability_patterns.items(), key=lambda item: (-item[1], item[0]))
        ),
        "latitude_column": latitude_column,
        "longitude_column": longitude_column,
        "missing_coordinate_rows": missing_coordinates,
        "invalid_coordinate_rows": invalid_coordinates,
        "datetime_column": datetime_column,
        "missing_datetime_rows": missing_datetimes,
        "invalid_datetime_rows": invalid_datetimes,
        "missing_required_columns": missing_required,
    }
    if verify_files:
        summary["file_audit"] = _path_audit(rows, fields, path, data_root)
        if summary["file_audit"]["missing_paths"]:
            issues.append(
                {
                    "severity": "error",
                    "code": "missing_files",
                    "detail": (
                        f"{summary['file_audit']['missing_paths']} referenced paths "
                        f"were not found for {path}"
                    ),
                }
            )

    return summary, events, issues


def build_quality_report(
    *,
    release_manifest: Path,
    source_manifest: Path | None = None,
    train_manifest: Path | None = None,
    test_manifest: Path | None = None,
    event_column: str | None = None,
    available_sensor_column: str | None = None,
    required_columns: Iterable[str] = (),
    verify_files: bool = False,
    data_root: Path | None = None,
    expected_source_events: int | None = None,
    expected_release_events: int | None = None,
) -> dict[str, Any]:
    paths = {
        "source": source_manifest,
        "release": release_manifest,
        "train": train_manifest,
        "test": test_manifest,
    }
    manifests: dict[str, Any] = {}
    event_sets: dict[str, set[str]] = {}
    issues: list[dict[str, str]] = []

    for name, path in paths.items():
        if path is None:
            continue
        summary, events, local_issues = summarize_manifest(
            path,
            event_column=event_column,
            available_sensor_column=available_sensor_column,
            required_columns=required_columns,
            verify_files=verify_files,
            data_root=data_root,
        )
        manifests[name] = summary
        event_sets[name] = events
        for issue in local_issues:
            issues.append({"manifest": name, **issue})

    expected_counts = {
        "source": expected_source_events,
        "release": expected_release_events,
    }
    for name, expected in expected_counts.items():
        if expected is None or name not in manifests:
            continue
        actual = manifests[name]["unique_event_count"]
        if actual != expected:
            issues.append(
                {
                    "manifest": name,
                    "severity": "error",
                    "code": "unexpected_event_count",
                    "detail": f"expected {expected} unique events, found {actual}",
                }
            )

    split_audit: dict[str, Any] | None = None
    if train_manifest is not None or test_manifest is not None:
        if train_manifest is None or test_manifest is None:
            issues.append(
                {
                    "manifest": "split",
                    "severity": "error",
                    "code": "incomplete_split_inputs",
                    "detail": "both --train-manifest and --test-manifest are required",
                }
            )
        elif manifests["train"]["event_column"] and manifests["test"]["event_column"]:
            overlap = sorted(event_sets["train"] & event_sets["test"])
            split_audit = {
                "train_unique_events": len(event_sets["train"]),
                "test_unique_events": len(event_sets["test"]),
                "event_overlap_count": len(overlap),
                "event_overlap_examples": overlap[:20],
            }
            if overlap:
                issues.append(
                    {
                        "manifest": "split",
                        "severity": "error",
                        "code": "event_leakage",
                        "detail": f"{len(overlap)} event IDs occur in both train and test",
                    }
                )

    report: dict[str, Any] = {
        "report_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "manifests": manifests,
        "split_audit": split_audit,
        "issues": issues,
        "error_count": sum(issue["severity"] == "error" for issue in issues),
    }
    if "source" in manifests and "release" in manifests:
        source_count = manifests["source"]["unique_event_count"]
        release_count = manifests["release"]["unique_event_count"]
        report["filtering"] = {
            "source_unique_events": source_count,
            "release_unique_events": release_count,
            "removed_unique_events": (
                source_count - release_count
                if source_count is not None and release_count is not None
                else None
            ),
        }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MethaneUnion Data Quality Report",
        "",
        f"Generated: `{report['generated_at_utc']}`  ",
        f"Git commit: `{report.get('git_commit') or 'unavailable'}`",
        "",
        "## Manifest summary",
        "",
        "| Manifest | Rows | Unique events | Multiple-row events | SHA-256 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name, summary in report["manifests"].items():
        lines.append(
            f"| {name} | {summary['row_count']} | "
            f"{summary['unique_event_count']} | {summary['events_with_multiple_rows']} | "
            f"`{summary['sha256'][:12]}…` |"
        )

    if report.get("filtering"):
        filtering = report["filtering"]
        lines.extend(
            [
                "",
                "## Filtering",
                "",
                f"- Source unique events: **{filtering['source_unique_events']}**",
                f"- Released unique events: **{filtering['release_unique_events']}**",
                f"- Removed unique events: **{filtering['removed_unique_events']}**",
            ]
        )

    release = report["manifests"].get("release")
    if release:
        lines.extend(
            [
                "",
                "## Released sensor availability",
                "",
                "| Sensor | Rows with sensor |",
                "| --- | ---: |",
            ]
        )
        for sensor, count in release["sensor_row_counts"].items():
            lines.append(f"| {sensor} | {count} |")
        lines.extend(
            [
                "",
                "| Availability pattern | Rows |",
                "| --- | ---: |",
            ]
        )
        for pattern, count in release["availability_patterns"].items():
            lines.append(f"| {pattern} | {count} |")

    if report.get("split_audit"):
        split = report["split_audit"]
        lines.extend(
            [
                "",
                "## Split validation",
                "",
                f"- Train unique events: **{split['train_unique_events']}**",
                f"- Test unique events: **{split['test_unique_events']}**",
                f"- Train/test event overlap: **{split['event_overlap_count']}**",
            ]
        )

    lines.extend(["", "## Findings", ""])
    if report["issues"]:
        for issue in report["issues"]:
            lines.append(
                f"- **{issue['severity'].upper()} — {issue['code']}** "
                f"({issue['manifest']}): {issue['detail']}"
            )
    else:
        lines.append("No manifest-level validation issues were detected.")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--test-manifest", type=Path)
    parser.add_argument("--event-column")
    parser.add_argument("--available-sensor-column")
    parser.add_argument("--required-column", action="append", default=[])
    parser.add_argument("--verify-files", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--expected-source-events", type=int)
    parser.add_argument("--expected-release-events", type=int)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--fail-on-issues", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_quality_report(
        source_manifest=args.source_manifest,
        release_manifest=args.release_manifest,
        train_manifest=args.train_manifest,
        test_manifest=args.test_manifest,
        event_column=args.event_column,
        available_sensor_column=args.available_sensor_column,
        required_columns=args.required_column,
        verify_files=args.verify_files,
        data_root=args.data_root,
        expected_source_events=args.expected_source_events,
        expected_release_events=args.expected_release_events,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(f"saved_json: {args.output_json}")
    print(f"saved_markdown: {args.output_markdown}")
    print(f"errors: {report['error_count']}")
    return 1 if args.fail_on_issues and report["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
