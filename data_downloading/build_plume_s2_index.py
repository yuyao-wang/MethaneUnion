import os
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd

BASE_DIR = "/data2/yuyao/methane_emission/carbon_mapper_data/CM_S2_L2A"
MERGED_CSV = "/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file.csv"
OUT_CSV = "/data2/yuyao/methane_emission/carbon_mapper_data/csvs/plume_s2_index.csv"
RAW_S2_DIR = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/data_download/raw_data_dir_s2"


def safe_path(value: str) -> str:
    if isinstance(value, str) and value.strip():
        return value
    return ""


def _parse_iso_dt(value: str):
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _parse_ext_pos_list(text: str):
    if not text:
        return None
    parts = text.split()
    if len(parts) % 2 != 0:
        return None
    coords = []
    for i in range(0, len(parts), 2):
        try:
            lat = float(parts[i])
            lon = float(parts[i + 1])
        except ValueError:
            return None
        coords.append((lon, lat))
    return coords


def _point_in_polygon(lon: float, lat: float, polygon):
    if not polygon:
        return False
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > lat) != (y2 > lat)) and (
            lon < (x2 - x1) * (lat - y1) / (y2 - y1 + 1e-12) + x1
        ):
            inside = not inside
    return inside


def _read_safe_meta(safe_dir: Path):
    meta = safe_dir / "MTD_MSIL2A.xml"
    if not meta.exists():
        return None
    try:
        root = ET.parse(meta).getroot()
    except ET.ParseError:
        return None
    cloud = root.findtext(".//Cloud_Coverage_Assessment")
    sensing_time = root.findtext(".//DATATAKE_SENSING_START") or root.findtext(
        ".//PRODUCT_START_TIME"
    )
    ext_pos = root.findtext(".//EXT_POS_LIST")
    product_uri = root.findtext(".//PRODUCT_URI") or safe_dir.name
    return {
        "safe_name": product_uri,
        "safe_path": safe_dir.as_posix(),
        "sensing_time": sensing_time,
        "cloud_cover": float(cloud) if cloud else None,
        "polygon": _parse_ext_pos_list(ext_pos),
    }


def _build_safe_index():
    index = {}
    root = Path(RAW_S2_DIR)
    if not root.exists():
        return index
    for safe_dir in root.glob("*.SAFE"):
        meta = _read_safe_meta(safe_dir)
        if not meta:
            continue
        date_key = (meta["sensing_time"] or "")[:10]
        index.setdefault(date_key, []).append(meta)
    return index


def _match_safe(plume_lon, plume_lat, plume_dt, safe_index):
    if not plume_dt:
        return None
    date_key = plume_dt.date().isoformat()
    candidates = safe_index.get(date_key, [])
    matches = []
    for meta in candidates:
        if _point_in_polygon(plume_lon, plume_lat, meta["polygon"]):
            matches.append(meta)
    if not matches:
        return None
    if not plume_dt:
        return matches[0]
    best = None
    best_delta = None
    for meta in matches:
        st = _parse_iso_dt(meta["sensing_time"])
        if not st:
            continue
        delta = abs((st - plume_dt).total_seconds())
        if best is None or delta < best_delta:
            best = meta
            best_delta = delta
    return best or matches[0]


def build_index():
    df = pd.read_csv(MERGED_CSV)
    safe_index = _build_safe_index()
    out = df[
        [
            "plume_id",
            "plume_latitude",
            "plume_longitude",
            "datetime",
            "instrument",
            "plume_tif",
            "plume_png",
            "rgb_tif",
            "rgb_png",
        ]
    ].copy()

    out["plume_path"] = out["plume_tif"].map(safe_path)

    s2_paths = []
    s2_exists = []
    for plume_id in out["plume_id"].astype(str):
        s2_path = os.path.join(BASE_DIR, plume_id, "s2.tif")
        exists = os.path.exists(s2_path)
        s2_paths.append(s2_path if exists else "")
        s2_exists.append(exists)

    out["s2_path"] = s2_paths
    out["s2_exists"] = s2_exists
    out["s2_cloud_cover"] = pd.NA
    out["s2_safe_name"] = ""
    out["s2_safe_path"] = ""
    out["s2_sensing_time"] = ""

    for idx, row in out.iterrows():
        if not row["s2_exists"]:
            continue
        plume_dt = _parse_iso_dt(str(row["datetime"])) if pd.notna(row["datetime"]) else None
        plume_lon = float(row["plume_longitude"])
        plume_lat = float(row["plume_latitude"])
        meta = _match_safe(plume_lon, plume_lat, plume_dt, safe_index)
        if not meta:
            continue
        out.at[idx, "s2_cloud_cover"] = meta.get("cloud_cover")
        out.at[idx, "s2_safe_name"] = meta.get("safe_name") or ""
        out.at[idx, "s2_safe_path"] = meta.get("safe_path") or ""
        out.at[idx, "s2_sensing_time"] = meta.get("sensing_time") or ""

    out.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}")
    print(f"S2 available: {sum(s2_exists)}/{len(s2_exists)}")


if __name__ == "__main__":
    build_index()
