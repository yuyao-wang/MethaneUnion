#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


TIMEPOINTS = ["t0", "prev1", "prev2", "prev3", "seasonal", "year"]

SOURCE_META = {
    "S2": {
        "source": "Copernicus CDSE catalogue+zipper",
        "product": "Sentinel-2 MSI L2A, productType=S2MSI2A, 20m JP2 bands B1-B9/B8A/B11/B12",
        "cloud_rule": "cloudCover <= 20",
        "year_offset_days": 360,
    },
    "L89": {
        "source": "Google Earth Engine export",
        "product": "LANDSAT/LC08/C02/T1_L2 + LANDSAT/LC09/C02/T1_L2, SR_B1-SR_B7 + QA bands",
        "cloud_rule": "CLOUD_COVER <= 20",
        "year_offset_days": 360,
    },
    "EMIT": {
        "source": "NASA Earthdata / earthaccess",
        "product": "EMITL2ARFL reflectance granule, SWIR methane subset selected from wavelengths",
        "cloud_rule": "",
        "year_offset_days": 180,
    },
    "S5P": {
        "source": "Copernicus CDSE catalogue+zipper",
        "product": "Sentinel-5P L2 CH4, productType=L2__CH4___, raw .nc",
        "cloud_rule": "",
        "year_offset_days": 360,
    },
}

TARGET_RAW_ROOT = Path("/mnt/engg-niulab/yuyao/sensors_raw_data")
TARGET_512_ROOT = Path("/mnt/engg-niulab/yuyao/preprocessed_512")


def has_value(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    s = str(v).strip()
    return s != "" and s.lower() not in {"nan", "none", "<na>"}


def svalue(row: pd.Series, col: str) -> str:
    if row is None or col not in row.index:
        return ""
    v = row.get(col)
    return str(v).strip() if has_value(v) else ""


def existing_path(path: str, check_files: bool) -> str:
    if not has_value(path):
        return ""
    if not check_files:
        return path
    return path if Path(path).exists() else ""


def load_by_pid(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame().set_index(pd.Index([], name="plume_id"))
    df = pd.read_csv(path, low_memory=False)
    if "plume_id" not in df.columns:
        return pd.DataFrame().set_index(pd.Index([], name="plume_id"))
    df["plume_id"] = df["plume_id"].astype(str)
    return df.drop_duplicates("plume_id", keep="first").set_index("plume_id", drop=False)


def target_raw_dir(sensor: str, timepoint: str, plume_id: str) -> str:
    return str(TARGET_RAW_ROOT / sensor / timepoint / plume_id)


def target_512_path(sensor: str, timepoint: str, plume_id: str) -> str:
    if sensor == "S2":
        name = {
            "t0": "s2_0_std_512.tif",
            "prev1": "s2_-7_std_512.tif",
            "prev2": "s2_prev2_std_512.tif",
            "prev3": "s2_prev3_std_512.tif",
            "seasonal": "s2_-90_std_512.tif",
            "year": "s2_-360_std_512.tif",
        }[timepoint]
        return str(TARGET_512_ROOT / "S2" / plume_id / name)
    if sensor == "L89":
        name = {
            "t0": "l89_0_std_512.tif",
            "prev1": "l89_-7_std_512.tif",
            "prev2": "l89_prev2_std_512.tif",
            "prev3": "l89_prev3_std_512.tif",
            "seasonal": "l89_-90_std_512.tif",
            "year": "l89_-360_std_512.tif",
        }[timepoint]
        return str(TARGET_512_ROOT / "L89" / plume_id / name)
    if sensor == "EMIT":
        name = {
            "t0": "emit_0_swir_512.tif",
            "prev1": "emit_prev1_swir_512.tif",
            "prev2": "emit_prev2_swir_512.tif",
            "prev3": "emit_prev3_swir_512.tif",
            "seasonal": "emit_-90_swir_512.tif",
            "year": "emit_-180_swir_512.tif",
        }[timepoint]
        return str(TARGET_512_ROOT / "EMIT" / plume_id / name)
    return ""


def get_lookup(lookup: pd.DataFrame, plume_id: str) -> Optional[pd.Series]:
    if lookup is None or lookup.empty or plume_id not in lookup.index:
        return None
    return lookup.loc[plume_id]


def existing_for(sensor: str, timepoint: str, plume_id: str, main: pd.Series,
                 l89: Optional[pd.Series], emit: Optional[pd.Series],
                 s5p: Optional[pd.Series], check_files: bool,
                 target_512_existing: set[str]) -> Dict[str, str]:
    out = {"existing_raw_path": "", "existing_512_path": "", "existing_metadata": ""}

    # Prefer niulab rebuilt/target 512 when present.
    target_512 = target_512_path(sensor, timepoint, plume_id)
    if target_512 and target_512 in target_512_existing:
        out["existing_512_path"] = target_512
        return out

    if sensor == "S2":
        main_cols = {
            "t0": "s2_0_std_512",
            "seasonal": "s2_-90_std_512",
            "year": "s2_-360_std_512",
        }
        if timepoint in main_cols:
            out["existing_512_path"] = existing_path(svalue(main, main_cols[timepoint]), check_files)
        # t0/prev1/seasonal/year raw exists on hardware /data2 through the old S2 csv.
        # The downloader should still skip these if 512 exists; raw is documented by source_manifest.
        return out

    if sensor == "L89" and l89 is not None:
        raw_cols = {
            "t0": "l89_path",
            "prev1": "l89_-7_path",
            "seasonal": "l89_pre_path",
            "year": "l89_pre_pre_path",
        }
        stage_cols = {
            "t0": "l89_0_std_512",
            "prev1": "l89_-7_std_512",
            "seasonal": "l89_-90_std_512",
            "year": "l89_-360_std_512",
        }
        if timepoint in raw_cols:
            out["existing_raw_path"] = existing_path(svalue(l89, raw_cols[timepoint]), check_files)
        if timepoint in stage_cols:
            out["existing_512_path"] = existing_path(svalue(l89, stage_cols[timepoint]), check_files)
        return out

    if sensor == "EMIT" and emit is not None:
        granule_cols = {
            "t0": "emit_granule_id",
            "seasonal": "emit_-90_granule_id",
            "year": "emit_-180_granule_id",
        }
        if timepoint in granule_cols:
            out["existing_metadata"] = svalue(emit, granule_cols[timepoint])
        # Existing WV3/L89 simulated 512 is not counted as existing for the new SWIR raw requirement.
        return out

    if sensor == "S5P" and s5p is not None:
        raw_cols = {
            "t0": "S5p_path",
            "seasonal": "s5p_minus90_path",
            "year": "s5p_minus360_path",
        }
        if timepoint in raw_cols:
            out["existing_raw_path"] = existing_path(svalue(s5p, raw_cols[timepoint]), check_files)
        return out

    return out


def sensor_has_t0(sensor: str, row: pd.Series) -> bool:
    col = f"{sensor.lower()}_has_t0"
    if col not in row.index:
        return False
    return str(row.get(col)).strip().lower() in {"true", "1", "yes"}


def planned_action(sensor: str, timepoint: str, row: pd.Series, existing: Dict[str, str]) -> str:
    if not sensor_has_t0(sensor, row):
        return "skip_no_sensor_t0"
    if existing.get("existing_512_path"):
        return "skip_existing_512"
    if sensor == "EMIT":
        # For EMIT, granule ids are metadata only; raw SWIR reflectance must exist/download.
        return "download"
    if existing.get("existing_raw_path"):
        return "skip_existing_raw"
    return "download"


def build(args: argparse.Namespace) -> pd.DataFrame:
    main = pd.read_csv(args.main_csv, low_memory=False)
    main["plume_id"] = main["plume_id"].astype(str)
    if args.only_any_t0:
        main = main[main["has_any_t0"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    if args.limit > 0:
        main = main.head(args.limit).copy()

    l89 = load_by_pid(args.l89_csv)
    emit = load_by_pid(args.emit_csv)
    s5p = load_by_pid(args.s5p_csv)

    rows = []
    for _, row in main.iterrows():
        plume_id = str(row["plume_id"])
        lookups = {
            "L89": get_lookup(l89, plume_id),
            "EMIT": get_lookup(emit, plume_id),
            "S5P": get_lookup(s5p, plume_id),
        }
        for sensor in args.sensors:
            for timepoint in TIMEPOINTS:
                meta = SOURCE_META[sensor]
                ex = existing_for(
                    sensor, timepoint, plume_id, row,
                    lookups.get("L89"), lookups.get("EMIT"), lookups.get("S5P"),
                    args.check_files, args.target_512_existing,
                )
                action = planned_action(sensor, timepoint, row, ex)
                target_raw = target_raw_dir(sensor, timepoint, plume_id)
                rows.append({
                    "plume_id": plume_id,
                    "sensor": sensor,
                    "timepoint": timepoint,
                    "action": action,
                    "event_time": row.get("datetime", ""),
                    "plume_latitude": row.get("plume_latitude", ""),
                    "plume_longitude": row.get("plume_longitude", ""),
                    "plume_bounds": row.get("plume_bounds", ""),
                    "sensor_has_t0": sensor_has_t0(sensor, row),
                    "t0_available_time": row.get(f"{sensor.lower()}_t0_time", ""),
                    "download_source": meta["source"],
                    "product_type": meta["product"],
                    "cloud_rule": meta["cloud_rule"],
                    "year_offset_days": meta["year_offset_days"],
                    "target_raw_dir": target_raw,
                    "target_512_path": target_512_path(sensor, timepoint, plume_id),
                    "existing_raw_path": ex.get("existing_raw_path", ""),
                    "existing_512_path": ex.get("existing_512_path", ""),
                    "existing_metadata": ex.get("existing_metadata", ""),
                    "source_manifest_note": source_manifest_note(sensor, timepoint),
                })
    return pd.DataFrame(rows)


def source_manifest_note(sensor: str, timepoint: str) -> str:
    if sensor == "S2":
        return "t0/prev1/seasonal/year legacy raw is in hardware /data2 CM_S2_L2A_gee90360.csv; prev2/prev3 need CDSE query"
    if sensor == "L89":
        return "legacy raw/512 comes from preprocess_dataset_L89/CM_L89_L2SR_std512.csv; prev2/prev3 need GEE query"
    if sensor == "EMIT":
        return "legacy granule ids come from preprocess_dataset_EMIT/merged_with_emit_tag.csv; SWIR raw .nc still required"
    if sensor == "S5P":
        return "legacy raw .nc comes from preprocess_dataset_s5p/s5p_all_OFFL_with_centers.csv; prev1/prev2/prev3 need CDSE query"
    return ""


def parse_sensors(raw: str) -> list[str]:
    sensors = [s.strip().upper() for s in raw.split(",") if s.strip()]
    valid = set(SOURCE_META)
    bad = [s for s in sensors if s not in valid]
    if bad:
        raise ValueError(f"unknown sensors: {bad}")
    return sensors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-csv", type=Path, default=Path("Upgrade_data_pipeline/csv/carbon_mapper_plumes_20160101_20260530_with_t0_flags.csv"))
    parser.add_argument("--l89-csv", type=Path, default=Path("preprocess_dataset_L89/CM_L89_L2SR_std512.csv"))
    parser.add_argument("--emit-csv", type=Path, default=Path("preprocess_dataset_EMIT/merged_with_emit_tag.csv"))
    parser.add_argument("--s5p-csv", type=Path, default=Path("preprocess_dataset_s5p/s5p_all_OFFL_with_centers.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("Upgrade_data_pipeline/csv/multisensor_6time_download_manifest.csv"))
    parser.add_argument("--sensors", type=parse_sensors, default=parse_sensors("S2,L89,EMIT,S5P"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--only-any-t0", action="store_true", default=True)
    args = parser.parse_args()

    # One directory scan is much faster than Path.exists() for every
    # plume/sensor/timepoint on SMB mounts.
    args.target_512_existing = {
        str(p)
        for p in TARGET_512_ROOT.glob("*/*/*.tif")
    }
    print(f"indexed_target_512_files: {len(args.target_512_existing)}")

    out = build(args)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"saved: {args.out_csv}")
    print(f"rows: {len(out)}")
    summary = out.groupby(["sensor", "timepoint", "action"]).size().reset_index(name="count")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
