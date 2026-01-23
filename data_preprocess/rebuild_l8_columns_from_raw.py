"""
Rebuild the Landsat 8/9 metadata columns in merged_file_with_l8.csv by
reading the already-downloaded raw products and plume-level stacks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import tifffile


MAX_L8_PER_PLUME = 3


def parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not isinstance(value, str) or len(value) == 0:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def datetime_to_iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_landsat_mtl(mtl_path: str) -> Dict[str, Optional[float]]:
    result: Dict[str, Optional[float]] = {
        "acq_datetime_iso": None,
        "sun_azimuth": None,
        "sun_elevation": None,
        "image_quality_oli": None,
        "image_quality_tirs": None,
    }

    if not Path(mtl_path).is_file():
        return result

    meta: Dict[str, str] = {}
    with open(mtl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("GROUP") or line.startswith("END_") or line == "END":
                continue
            if "=" not in line:
                continue
            k, v = [x.strip() for x in line.split("=", 1)]
            if v.startswith('"') and v.endswith('"'):
                v = v[1:-1]
            meta[k] = v

    date_str = meta.get("DATE_ACQUIRED")
    time_str = meta.get("SCENE_CENTER_TIME")

    if date_str and time_str:
        t = time_str
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        dt_str = f"{date_str}T{t}"
        try:
            dt = datetime.fromisoformat(dt_str)
            result["acq_datetime_iso"] = datetime_to_iso_z(dt)
        except ValueError:
            result["acq_datetime_iso"] = None

    def _get_float(k: str) -> Optional[float]:
        v = meta.get(k)
        if v is None:
            return None
        try:
            return float(v)
        except ValueError:
            return None

    def _get_int(k: str) -> Optional[int]:
        v = meta.get(k)
        if v is None:
            return None
        try:
            return int(v)
        except ValueError:
            return None

    result["sun_azimuth"] = _get_float("SUN_AZIMUTH")
    result["sun_elevation"] = _get_float("SUN_ELEVATION")
    result["image_quality_oli"] = _get_int("IMAGE_QUALITY_OLI")
    result["image_quality_tirs"] = _get_int("IMAGE_QUALITY_TIRS")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct Landsat L2SP metadata columns using existing raw downloads."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../carbon_mapper_data/csvs/merged_file_with_l8.csv"),
        help="CSV that already contains the l8_* columns (they may be empty).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../carbon_mapper_data/csvs/merged_file_with_l8.csv"),
        help="Where to write the augmented CSV. Defaults to the input path.",
    )
    parser.add_argument(
        "--plume-dir",
        type=Path,
        default=Path("../carbonmapper_data_l89_l2sp"),
        help="Directory that holds per-plume L8 stack files (l8_<product>.tif).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("../data/raw_data_dir_L89_L2SP"),
        help="Directory that stores the original Landsat product folders with MTL metadata.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=MAX_L8_PER_PLUME,
        help="Maximum number of L8 scenes to write per plume.",
    )
    return parser.parse_args()


@dataclass
class SceneMetadata:
    scene_id: str
    datetime_iso: str
    tif_path: Path
    height: Optional[int]
    width: Optional[int]
    sun_azimuth: Optional[float]
    sun_elevation: Optional[float]
    image_quality_oli: Optional[int]
    image_quality_tirs: Optional[int]
    sort_key: float = float("inf")


def infer_datetime_from_product_id(product_id: str) -> Optional[datetime]:
    parts = product_id.split("_")
    if len(parts) < 4:
        return None
    date_token = parts[3]
    if len(date_token) < 8:
        return None
    try:
        return datetime.strptime(date_token[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def read_stack_shape(tif_path: Path) -> tuple[Optional[int], Optional[int]]:
    try:
        with tifffile.TiffFile(str(tif_path)) as tif:
            shape = tif.series[0].shape
    except Exception:
        return None, None
    if len(shape) == 3:
        _, height, width = shape
    elif len(shape) == 2:
        height, width = shape
    else:
        return None, None
    return int(height), int(width)


def iter_scene_stacks(plume_dir: Path) -> Iterable[Path]:
    if not plume_dir.is_dir():
        return []
    return sorted(plume_dir.glob("l8_*.tif"))


def build_scene_metadata(
    tif_path: Path,
    raw_dir: Path,
) -> SceneMetadata:
    product_id = tif_path.stem.replace("l8_", "")
    scene_dir = raw_dir / product_id
    mtl_path = scene_dir / f"{product_id}_MTL.txt"
    meta = parse_landsat_mtl(str(mtl_path))

    datetime_iso = meta.get("acq_datetime_iso")
    acq_dt = parse_iso_datetime(datetime_iso) if datetime_iso else None
    if acq_dt is None:
        inferred = infer_datetime_from_product_id(product_id)
        if inferred is not None:
            acq_dt = inferred
            datetime_iso = datetime_to_iso_z(inferred)

    height, width = read_stack_shape(tif_path)

    return SceneMetadata(
        scene_id=product_id,
        datetime_iso=datetime_iso or "",
        tif_path=tif_path.resolve(),
        height=height,
        width=width,
        sun_azimuth=meta.get("sun_azimuth"),
        sun_elevation=meta.get("sun_elevation"),
        image_quality_oli=meta.get("image_quality_oli"),
        image_quality_tirs=meta.get("image_quality_tirs"),
    )


def assign_sort_keys(
    scenes: List[SceneMetadata], event_dt: Optional[datetime]
) -> List[SceneMetadata]:
    for scene in scenes:
        scene_dt = parse_iso_datetime(scene.datetime_iso)
        if scene_dt and event_dt:
            scene.sort_key = abs((scene_dt - event_dt).total_seconds())
        elif scene_dt:
            scene.sort_key = scene_dt.timestamp()
        else:
            scene.sort_key = float("inf")
    return sorted(scenes, key=lambda s: (s.sort_key, s.datetime_iso))


def ensure_l8_columns(df: pd.DataFrame, max_scenes: int) -> None:
    for i in range(1, max_scenes + 1):
        for suffix in (
            "scene_id",
            "datetime",
            "tif",
            "height",
            "width",
            "sun_azimuth",
            "sun_elevation",
            "image_quality_oli",
            "image_quality_tirs",
        ):
            col = f"l8_{i}_{suffix}"
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].astype("object")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, low_memory=False)
    ensure_l8_columns(df, args.max_scenes)

    plume_dir_root = args.plume_dir.resolve()
    raw_dir = args.raw_dir.resolve()

    updated_rows = 0

    for idx, row in df.iterrows():
        plume_id = row.get("plume_id", "")
        if not isinstance(plume_id, str) or len(plume_id) == 0:
            continue

        plume_dir = plume_dir_root / plume_id
        tif_paths = list(iter_scene_stacks(plume_dir))
        if not tif_paths:
            continue

        scenes: List[SceneMetadata] = []
        for tif_path in tif_paths:
            scenes.append(build_scene_metadata(tif_path, raw_dir))

        event_dt = parse_iso_datetime(row.get("datetime"))
        ranked = assign_sort_keys(scenes, event_dt)

        for i in range(args.max_scenes):
            prefix = f"l8_{i+1}_"
            if i < len(ranked):
                info = ranked[i]
                df.at[idx, prefix + "scene_id"] = info.scene_id
                df.at[idx, prefix + "datetime"] = info.datetime_iso
                df.at[idx, prefix + "tif"] = str(info.tif_path)
                df.at[idx, prefix + "height"] = info.height if info.height is not None else ""
                df.at[idx, prefix + "width"] = info.width if info.width is not None else ""
                df.at[idx, prefix + "sun_azimuth"] = info.sun_azimuth if info.sun_azimuth is not None else ""
                df.at[idx, prefix + "sun_elevation"] = (
                    info.sun_elevation if info.sun_elevation is not None else ""
                )
                df.at[idx, prefix + "image_quality_oli"] = (
                    info.image_quality_oli if info.image_quality_oli is not None else ""
                )
                df.at[idx, prefix + "image_quality_tirs"] = (
                    info.image_quality_tirs if info.image_quality_tirs is not None else ""
                )
            else:
                df.at[idx, prefix + "scene_id"] = ""
                df.at[idx, prefix + "datetime"] = ""
                df.at[idx, prefix + "tif"] = ""
                df.at[idx, prefix + "height"] = ""
                df.at[idx, prefix + "width"] = ""
                df.at[idx, prefix + "sun_azimuth"] = ""
                df.at[idx, prefix + "sun_elevation"] = ""
                df.at[idx, prefix + "image_quality_oli"] = ""
                df.at[idx, prefix + "image_quality_tirs"] = ""

        updated_rows += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Updated L8 metadata for {updated_rows} plumes. Saved to {args.output}")


if __name__ == "__main__":
    main()
