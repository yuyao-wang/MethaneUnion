#!/usr/bin/env python3
"""Audit final six-time artifacts by plume_id against files that exist on disk.

The key distinction in this audit is:
  * S2, L89, and EMIT final artifacts are six separate 512x512 TIFF files.
  * S5P final artifacts are one six-time 224x224 NPZ file, so S5P is reported
    separately and is not included in the strict 512-TIFF union.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_ROOT = REPO_ROOT / "Upgrade_data_pipeline" / "csv"

CM_CSV = CSV_ROOT / "carbon_mapper_plumes_20160101_20260530_with_plume_tif.csv"
S2_META_CSV = CSV_ROOT / "s2_6time_point_center_exact_v3_paths.csv"
L89_CSV = CSV_ROOT / "l89_6time_complete_paths.csv"
EMIT_CSV = Path(
    "/mnt/engg-niulab/yuyao/preprocessed_512/"
    "emit_32band/emit_6time_512_manifest.csv"
)
S5P_SAMPLE_CSV = CSV_ROOT / "s5p_6time_samples.pos_only_14757.csv"
S5P_META_CSV = CSV_ROOT / "s5p_6time_with_centers.csv"

S2_ROOT = Path(
    "/mnt/engg-niulab/yuyao/preprocessed_512/"
    "S2_6time_point_center_plus1000_v14"
)

TIMEPOINTS = ["t0", "prev1", "prev2", "prev3", "seasonal", "year"]
PREV_TIMEPOINTS = ["prev1", "prev2", "prev3", "seasonal", "year"]

OUT_SUMMARY = CSV_ROOT / "downloaded_6time_artifact_sensor_summary.csv"
OUT_YEAR = CSV_ROOT / "downloaded_6time_artifact_by_sensor_and_year.csv"
OUT_PREV = CSV_ROOT / "downloaded_6time_prev_day_stats.csv"
OUT_UNION = CSV_ROOT / "downloaded_6time_512_union_by_plume.csv"


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def as_bool(value: object) -> bool:
    return clean(value).lower() in {"1", "true", "yes"}


_EXISTS_CACHE: dict[str, bool] = {}


def is_nonempty_file(value: object) -> bool:
    text = clean(value)
    if not text:
        return False
    if text not in _EXISTS_CACHE:
        try:
            path = Path(text)
            _EXISTS_CACHE[text] = path.is_file() and path.stat().st_size > 0
        except OSError:
            _EXISTS_CACHE[text] = False
    return _EXISTS_CACHE[text]


def first_existing(row: pd.Series, columns: Iterable[str]) -> str:
    for column in columns:
        if column in row.index and is_nonempty_file(row[column]):
            return clean(row[column])
    return ""


def read_unique(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame["plume_id"] = frame["plume_id"].astype(str)
    return frame.drop_duplicates("plume_id", keep="first").copy()


def audit_s2() -> tuple[set[str], pd.DataFrame, dict[str, dict[str, str]]]:
    filenames = {
        "t0": "s2.tif",
        "prev1": "s2_-7.tif",
        "prev2": "s2_prev2.tif",
        "prev3": "s2_prev3.tif",
        "seasonal": "s2_-90.tif",
        "year": "s2_-360.tif",
    }
    ids_by_timepoint: dict[str, set[str]] = {}
    paths_by_timepoint: dict[str, dict[str, str]] = {}
    for timepoint in TIMEPOINTS:
        timepoint_root = S2_ROOT / timepoint
        ids: set[str] = set()
        paths: dict[str, str] = {}
        with os.scandir(timepoint_root) as entries:
            for entry in entries:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                image = Path(entry.path) / filenames[timepoint]
                if is_nonempty_file(image):
                    ids.add(entry.name)
                    paths[entry.name] = str(image)
        ids_by_timepoint[timepoint] = ids
        paths_by_timepoint[timepoint] = paths

    complete_ids = set.intersection(*(ids_by_timepoint[tp] for tp in TIMEPOINTS))
    metadata = read_unique(S2_META_CSV)
    return complete_ids, metadata, paths_by_timepoint


def audit_l89() -> tuple[set[str], pd.DataFrame, dict[str, dict[str, str]]]:
    frame = read_unique(L89_CSV)
    aliases = {
        "t0": ["t0_512_path", "l89_0_std_512"],
        "prev1": ["prev1_512_path", "l89_-7_std_512"],
        "prev2": ["prev2_512_path", "l89_prev2_std_512"],
        "prev3": ["prev3_512_path", "l89_prev3_std_512"],
        "seasonal": ["seasonal_512_path", "l89_-90_std_512"],
        "year": ["year_512_path", "l89_-360_std_512"],
    }
    paths_by_timepoint = {tp: {} for tp in TIMEPOINTS}
    complete_ids: set[str] = set()
    for _, row in frame.iterrows():
        plume_id = str(row["plume_id"])
        present = True
        for timepoint in TIMEPOINTS:
            path = first_existing(row, aliases[timepoint])
            if path:
                paths_by_timepoint[timepoint][plume_id] = path
            else:
                present = False
        if present:
            complete_ids.add(plume_id)
    return complete_ids, frame, paths_by_timepoint


def audit_emit() -> tuple[set[str], pd.DataFrame, dict[str, dict[str, str]]]:
    frame = read_unique(EMIT_CSV)
    paths_by_timepoint = {tp: {} for tp in TIMEPOINTS}
    complete_ids: set[str] = set()
    for _, row in frame.iterrows():
        plume_id = str(row["plume_id"])
        present = True
        for timepoint in TIMEPOINTS:
            path = first_existing(row, [f"{timepoint}_512_path"])
            if path:
                paths_by_timepoint[timepoint][plume_id] = path
            else:
                present = False
        if present:
            complete_ids.add(plume_id)
    return complete_ids, frame, paths_by_timepoint


def audit_s5p() -> tuple[set[str], pd.DataFrame]:
    samples = read_unique(S5P_SAMPLE_CSV)
    expected_channels = set(TIMEPOINTS)
    complete_ids: set[str] = set()
    for _, row in samples.iterrows():
        channels = {
            item.strip()
            for item in clean(row.get("channels")).split(",")
            if item.strip()
        }
        flags_ok = all(as_bool(row.get(f"has_{tp}", False)) for tp in TIMEPOINTS)
        if (
            is_nonempty_file(row.get("image_path"))
            and channels == expected_channels
            and flags_ok
        ):
            complete_ids.add(str(row["plume_id"]))
    metadata = read_unique(S5P_META_CSV)
    return complete_ids, metadata


def parse_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def time_series_with_fallback(
    frame: pd.DataFrame, primary: str, fallback: str | None = None
) -> pd.Series:
    if primary in frame.columns:
        values = frame[primary].copy()
    else:
        values = pd.Series("", index=frame.index, dtype=object)
    if fallback and fallback in frame.columns:
        missing = values.isna() | values.astype(str).str.strip().isin(["", "nan", "None"])
        values.loc[missing] = frame.loc[missing, fallback]
    return parse_time(values)


def prev_stats(
    sensor: str, complete_ids: set[str], metadata: pd.DataFrame
) -> list[dict[str, object]]:
    frame = metadata[metadata["plume_id"].isin(complete_ids)].copy()
    if sensor == "S5P":
        t0 = time_series_with_fallback(frame, "s5p_t0_image_time", "t0_image_time")
    else:
        t0 = time_series_with_fallback(frame, "t0_image_time")

    rows: list[dict[str, object]] = []
    for timepoint in PREV_TIMEPOINTS:
        if sensor == "S5P":
            previous = time_series_with_fallback(
                frame, f"s5p_{timepoint}_image_time", f"{timepoint}_image_time"
            )
        else:
            previous = time_series_with_fallback(frame, f"{timepoint}_image_time")
        delta = (t0 - previous).dt.total_seconds() / 86400.0
        valid = delta.dropna()
        rounded = valid.round().astype(int)
        mode = rounded.mode()
        rows.append(
            {
                "sensor": sensor,
                "timepoint": timepoint,
                "complete_artifact_plumes": len(complete_ids),
                "valid_time_pairs": int(valid.size),
                "missing_time_pairs": int(len(frame) - valid.size),
                "negative_delta_count": int((valid < 0).sum()),
                "zero_delta_count": int((valid == 0).sum()),
                "min_days": float(valid.min()) if len(valid) else np.nan,
                "p05_days": float(valid.quantile(0.05)) if len(valid) else np.nan,
                "p25_days": float(valid.quantile(0.25)) if len(valid) else np.nan,
                "median_days": float(valid.median()) if len(valid) else np.nan,
                "mean_days": float(valid.mean()) if len(valid) else np.nan,
                "p75_days": float(valid.quantile(0.75)) if len(valid) else np.nan,
                "p95_days": float(valid.quantile(0.95)) if len(valid) else np.nan,
                "max_days": float(valid.max()) if len(valid) else np.nan,
                "rounded_day_mode": int(mode.iloc[0]) if len(mode) else np.nan,
                "rounded_day_mode_count": (
                    int((rounded == mode.iloc[0]).sum()) if len(mode) else 0
                ),
            }
        )
    return rows


def main() -> int:
    print("Auditing S2 files...", flush=True)
    s2_ids, s2_meta, s2_paths = audit_s2()
    print(f"S2 complete 512 TIFF plumes: {len(s2_ids)}", flush=True)

    print("Auditing L89 files...", flush=True)
    l89_ids, l89_meta, l89_paths = audit_l89()
    print(f"L89 complete 512 TIFF plumes: {len(l89_ids)}", flush=True)

    print("Auditing EMIT files...", flush=True)
    emit_ids, emit_meta, emit_paths = audit_emit()
    print(f"EMIT complete 512 TIFF plumes: {len(emit_ids)}", flush=True)

    print("Auditing S5P files...", flush=True)
    s5p_ids, s5p_meta = audit_s5p()
    print(f"S5P complete 224 NPZ plumes: {len(s5p_ids)}", flush=True)

    cm = read_unique(CM_CSV)
    cm["event_time"] = parse_time(cm["datetime"])
    cm["event_year"] = cm["event_time"].dt.year.astype("Int64")
    cm_index = cm.set_index("plume_id")

    sensor_ids = {
        "S2": s2_ids,
        "L89": l89_ids,
        "EMIT": emit_ids,
        "S5P": s5p_ids,
    }
    strict_512_union = s2_ids | l89_ids | emit_ids
    s5p_joined_ids = s5p_ids & strict_512_union

    union = cm_index.loc[
        cm_index.index.intersection(strict_512_union),
        [
            "event_time",
            "event_year",
            "country",
            "region",
            "ipcc_sector",
            "instrument",
            "platform",
        ],
    ].copy()
    union = union.reset_index()
    for sensor, ids in sensor_ids.items():
        union[f"has_{sensor.lower()}_complete_artifact"] = union["plume_id"].isin(ids)
    union["strict_512_sensor_count"] = union[
        [
            "has_s2_complete_artifact",
            "has_l89_complete_artifact",
            "has_emit_complete_artifact",
        ]
    ].sum(axis=1)
    union["all_artifact_sensor_count"] = union[
        [
            "has_s2_complete_artifact",
            "has_l89_complete_artifact",
            "has_emit_complete_artifact",
            "has_s5p_complete_artifact",
        ]
    ].sum(axis=1)
    union = union.sort_values(["event_time", "plume_id"])
    union.to_csv(OUT_UNION, index=False)

    summary_specs = [
        ("S2", "6 x 512x512 TIFF", "strict_512", s2_ids, s2_paths),
        ("L89", "6 x 512x512 TIFF", "strict_512", l89_ids, l89_paths),
        ("EMIT", "6 x 512x512 TIFF", "strict_512", emit_ids, emit_paths),
        (
            "S5P_JOINED_TO_ANY_512",
            "1 x six-time 224x224 NPZ",
            "joined_to_strict_512_cohort",
            s5p_joined_ids,
            {
                tp: {pid: "packed_in_npz" for pid in s5p_joined_ids}
                for tp in TIMEPOINTS
            },
        ),
        (
            "S5P_STANDALONE_REFERENCE",
            "1 x six-time 224x224 NPZ",
            "not_used_as_512_cohort_count",
            s5p_ids,
            {
                tp: {pid: "packed_in_npz" for pid in s5p_ids}
                for tp in TIMEPOINTS
            },
        ),
        (
            "ANY_512_TIFF_SENSOR_UNION",
            "S2/L89/EMIT union",
            "strict_512_union",
            strict_512_union,
            None,
        ),
    ]
    summary_rows = []
    for sensor, artifact, category, ids, paths in summary_specs:
        years = cm_index.loc[cm_index.index.intersection(ids), "event_year"]
        row = {
            "sensor_or_scope": sensor,
            "final_artifact": artifact,
            "category": category,
            "complete_unique_plumes": len(ids),
            "2016_complete_unique_plumes": int(years.eq(2016).sum()),
            "2026_complete_unique_plumes": int(years.eq(2026).sum()),
            "not_found_in_cm_catalogue": len(ids - set(cm_index.index)),
        }
        for timepoint in TIMEPOINTS:
            row[f"{timepoint}_artifact_plumes"] = (
                len(paths[timepoint]) if paths is not None else np.nan
            )
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)

    years = sorted(int(y) for y in cm["event_year"].dropna().unique())
    year_rows = []
    for year in years:
        year_ids = set(cm.loc[cm["event_year"].eq(year), "plume_id"])
        year_rows.append(
            {
                "event_year": year,
                "s2_complete_512_plumes": len(s2_ids & year_ids),
                "l89_complete_512_plumes": len(l89_ids & year_ids),
                "emit_complete_512_plumes": len(emit_ids & year_ids),
                "s5p_joined_to_any_512_plumes": len(s5p_joined_ids & year_ids),
                "s5p_standalone_reference_plumes": len(s5p_ids & year_ids),
                "any_512_tiff_sensor_union_plumes": len(
                    strict_512_union & year_ids
                ),
            }
        )
    year_rows.append(
        {
            "event_year": "TOTAL",
            "s2_complete_512_plumes": len(s2_ids),
            "l89_complete_512_plumes": len(l89_ids),
            "emit_complete_512_plumes": len(emit_ids),
            "s5p_joined_to_any_512_plumes": len(s5p_joined_ids),
            "s5p_standalone_reference_plumes": len(s5p_ids),
            "any_512_tiff_sensor_union_plumes": len(strict_512_union),
        }
    )
    pd.DataFrame(year_rows).to_csv(OUT_YEAR, index=False)

    stats_rows = []
    stats_rows.extend(prev_stats("S2", s2_ids, s2_meta))
    stats_rows.extend(prev_stats("L89", l89_ids, l89_meta))
    stats_rows.extend(prev_stats("EMIT", emit_ids, emit_meta))
    stats_rows.extend(prev_stats("S5P_JOINED_TO_ANY_512", s5p_joined_ids, s5p_meta))
    stats = pd.DataFrame(stats_rows)
    numeric = [
        "min_days",
        "p05_days",
        "p25_days",
        "median_days",
        "mean_days",
        "p75_days",
        "p95_days",
        "max_days",
    ]
    stats[numeric] = stats[numeric].round(2)
    stats.to_csv(OUT_PREV, index=False)

    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(pd.DataFrame(year_rows).to_string(index=False))
    print(stats.to_string(index=False))
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_YEAR}")
    print(f"Wrote {OUT_PREV}")
    print(f"Wrote {OUT_UNION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
