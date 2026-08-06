#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


TIMEPOINTS = [
    ("t0", "has_t0", "t0_raw_path", "t0_512_path", "t0_image_time", "t0_overpass_key", "t0_cloud_cover"),
    ("prev1", "has_prev1", "prev1_raw_path", "prev1_512_path", "prev1_image_time", "prev1_overpass_key", "prev1_cloud_cover"),
    ("prev2", "has_prev2", "prev2_raw_path", "prev2_512_path", "prev2_image_time", "prev2_overpass_key", "prev2_cloud_cover"),
    ("prev3", "has_prev3", "prev3_raw_path", "prev3_512_path", "prev3_image_time", "prev3_overpass_key", "prev3_cloud_cover"),
    (
        "seasonal",
        "has_seasonal",
        "seasonal_raw_path",
        "seasonal_512_path",
        "seasonal_image_time",
        "seasonal_overpass_key",
        "seasonal_cloud_cover",
    ),
    ("year", "has_year", "year_raw_path", "year_512_path", "year_image_time", "year_overpass_key", "year_cloud_cover"),
]

WEATHER_COLUMNS = [
    "era5_cache_path",
    "era5_latitude",
    "era5_longitude",
    "u10",
    "v10",
    "wind_speed_10m",
    "wind_dir_10m",
    "u100",
    "v100",
    "wind_speed_100m",
    "wind_dir_100m",
    "boundary_layer_height",
    "surface_pressure",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_column_water_vapour",
    "total_cloud_cover",
    "total_precipitation",
]


def has_value(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null", "<na>"}


def iso_z(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def rounded_era5_hour(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return (ts + pd.Timedelta(minutes=30)).dt.floor("h").dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_l89_long(l89: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["plume_id", "event_time", "plume_latitude", "plume_longitude", "plume_bounds"]
    chunks = []
    for tp, flag_col, raw_col, path_col, time_col, overpass_col, cloud_col in TIMEPOINTS:
        cols = [c for c in base_cols + [flag_col, raw_col, path_col, time_col, overpass_col, cloud_col] if c in l89.columns]
        part = l89[cols].copy()
        rename = {
            flag_col: "l89_has_512",
            raw_col: "l89_raw_path",
            path_col: "l89_512_path",
            time_col: "l89_image_time",
            overpass_col: "l89_overpass_key",
            cloud_col: "l89_cloud_cover",
        }
        part = part.rename(columns={k: v for k, v in rename.items() if k in part.columns})
        part["sensor"] = "L89"
        part["timepoint"] = tp
        for col in ["l89_has_512", "l89_raw_path", "l89_512_path", "l89_image_time", "l89_overpass_key", "l89_cloud_cover"]:
            if col not in part.columns:
                part[col] = ""
        keep = (
            part["l89_has_512"].astype(str).eq("1")
            & part["l89_512_path"].fillna("").astype(str).str.strip().ne("")
            & part["l89_image_time"].fillna("").astype(str).str.strip().ne("")
        )
        chunks.append(part.loc[keep])
    out = pd.concat(chunks, ignore_index=True)
    out["l89_image_time_utc"] = iso_z(out["l89_image_time"])
    out["l89_expected_era5_time_utc"] = rounded_era5_hour(out["l89_image_time"])
    return out


def load_l89_era5_ok(path: Path) -> pd.DataFrame:
    era5 = pd.read_csv(path, dtype=str, low_memory=False)
    era5 = era5[(era5["sensor"].eq("L89")) & (era5["status"].eq("ok"))].copy()
    era5 = era5.drop_duplicates(["plume_id", "sensor", "timepoint"], keep="last")
    era5["era5_image_time_utc"] = iso_z(era5["image_time"])
    return era5


def merge_long(l89_long: pd.DataFrame, era5_ok: pd.DataFrame) -> pd.DataFrame:
    era_cols = [
        "plume_id",
        "sensor",
        "timepoint",
        "image_time",
        "era5_image_time_utc",
        "image_time_source",
        "era5_time_utc",
        "time_delta_minutes",
        "plume_latitude",
        "plume_longitude",
        *WEATHER_COLUMNS,
    ]
    era_cols = [c for c in era_cols if c in era5_ok.columns]
    merged = l89_long.merge(
        era5_ok[era_cols],
        on=["plume_id", "sensor", "timepoint"],
        how="left",
        suffixes=("", "_era5_source"),
    )
    merged["era5_has_ok"] = merged["era5_time_utc"].fillna("").astype(str).str.strip().ne("")
    merged["era5_image_time_match"] = merged["era5_has_ok"] & merged["era5_image_time_utc"].eq(merged["l89_image_time_utc"])
    merged["era5_hour_match"] = merged["era5_has_ok"] & merged["era5_time_utc"].eq(merged["l89_expected_era5_time_utc"])
    lat_cur = pd.to_numeric(merged["plume_latitude"], errors="coerce")
    lon_cur = pd.to_numeric(merged["plume_longitude"], errors="coerce")
    lat_era = pd.to_numeric(merged.get("plume_latitude_era5_source", ""), errors="coerce")
    lon_era = pd.to_numeric(merged.get("plume_longitude_era5_source", ""), errors="coerce")
    merged["era5_location_match"] = (
        merged["era5_has_ok"]
        & lat_cur.sub(lat_era).abs().le(1e-6)
        & lon_cur.sub(lon_era).abs().le(1e-6)
    )
    merged["era5_ready"] = (
        merged["era5_has_ok"]
        & merged["era5_image_time_match"]
        & merged["era5_hour_match"]
        & merged["era5_location_match"]
    )
    merged["era5_needs_build"] = ~merged["era5_ready"]
    return merged


def build_wide(l89: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    out = l89.copy()
    long_cols = [
        "plume_id",
        "timepoint",
        "era5_has_ok",
        "era5_ready",
        "era5_needs_build",
        "l89_expected_era5_time_utc",
        "era5_time_utc",
        "era5_image_time_match",
        "era5_hour_match",
        "era5_location_match",
        *WEATHER_COLUMNS,
    ]
    long_cols = [c for c in long_cols if c in merged.columns]
    for tp, *_ in TIMEPOINTS:
        part = merged.loc[merged["timepoint"].eq(tp), long_cols].drop(columns=["timepoint"])
        part = part.rename(columns={c: f"{tp}_{c}" for c in part.columns if c != "plume_id"})
        out = out.merge(part, on="plume_id", how="left")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge current L89 512-image table with L89 ERA5 weather rows.")
    parser.add_argument("--l89-csv", default="Upgrade_data_pipeline/csv/l89_6time_complete_paths.csv")
    parser.add_argument("--era5-csv", default="Upgrade_data_pipeline/csv/era5_image_time_weather.csv")
    parser.add_argument("--out-long", default="Upgrade_data_pipeline/csv/l89_512_era5_weather_long.csv")
    parser.add_argument("--out-wide", default="Upgrade_data_pipeline/csv/l89_6time_complete_paths_with_era5.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    l89 = pd.read_csv(args.l89_csv, dtype=str, low_memory=False)
    l89_long = build_l89_long(l89)
    era5_ok = load_l89_era5_ok(Path(args.era5_csv))
    merged = merge_long(l89_long, era5_ok)
    wide = build_wide(l89, merged)

    Path(args.out_long).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_long, index=False)
    wide.to_csv(args.out_wide, index=False)

    print(f"l89_512_records={len(merged)}")
    print(f"l89_512_unique_plumes={merged['plume_id'].nunique()}")
    print(f"era5_ready={int(merged['era5_ready'].sum())}")
    print(f"era5_needs_build={int(merged['era5_needs_build'].sum())}")
    print("needs_build_by_timepoint=")
    print(merged.loc[merged["era5_needs_build"], "timepoint"].value_counts().sort_index().to_string())
    print(f"wrote_long={args.out_long}")
    print(f"wrote_wide={args.out_wide}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
