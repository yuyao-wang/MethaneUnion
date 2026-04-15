import argparse
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from common import (
    EMIT_MASK_ROOT,
    RAW_MASK_ROOT,
    SENSOR_SPECS,
    SENSORS,
    binary_array,
    emit_mask_path,
    first_existing_path,
    has_value,
    raw_plume_path,
    read_csv_rows,
    read_first_band,
    write_binary_tif,
    write_csv_rows,
)


DEFAULT_OUT_ROOT = Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/finalDataset_query/raw512")


def source_mask_candidates(row: Dict[str, str], sensor: str, raw_mask_root: Path, emit_mask_root: Path) -> list[Tuple[Path, str]]:
    plume_id = str(row.get("plume_id", "")).strip()
    out: list[Tuple[Path, str]] = []
    raw = raw_plume_path(plume_id, raw_mask_root)
    if raw.exists():
        out.append((raw, "raw_plume"))

    if sensor == "emit":
        emit = emit_mask_path(plume_id, emit_mask_root)
        if emit.exists():
            out.append((emit, "existing_emit_mask"))

    col = SENSOR_SPECS[sensor]["source_mask_col"]
    value = row.get(col, "")
    if has_value(value):
        p = Path(str(value))
        if p.exists():
            out.append((p, f"existing_{sensor}_mask"))
    return out


def reproject_or_copy_binary(src_mask: Path, dst_image: Path, dst_mask: Path, source_kind: str, overwrite: bool) -> str:
    if dst_mask.exists() and not overwrite:
        return "exists"

    with rasterio.open(dst_image) as ref:
        profile_ref = ref.profile
        dst_shape = (ref.height, ref.width)
        dst_transform = ref.transform
        dst_crs = ref.crs

        with rasterio.open(src_mask) as src:
            src_arr = src.read(1).astype(np.float32)
            src_crs = src.crs
            src_transform = src.transform

            can_reproject = source_kind == "raw_plume" and src_crs is not None and dst_crs is not None
            if can_reproject:
                dst = np.zeros(dst_shape, dtype=np.float32)
                reproject(
                    source=src_arr,
                    destination=dst,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
                mask = binary_array(dst)
                status = "reprojected_raw_plume"
            else:
                mask = binary_array(src_arr)
                if mask.shape != dst_shape:
                    # Existing masks are expected to be 512x512. If a legacy mask has
                    # georeferencing, still try a nearest-neighbor reprojection.
                    if src_crs is not None and dst_crs is not None:
                        dst = np.zeros(dst_shape, dtype=np.float32)
                        reproject(
                            source=mask.astype(np.float32),
                            destination=dst,
                            src_transform=src_transform,
                            src_crs=src_crs,
                            dst_transform=dst_transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.nearest,
                        )
                        mask = binary_array(dst)
                        status = "reprojected_existing_mask"
                    else:
                        raise RuntimeError(f"Mask shape {mask.shape} does not match target image {dst_shape}: {src_mask}")
                else:
                    status = "copied_existing_mask_binary"

    write_binary_tif(dst_mask, mask, profile_ref=profile_ref)
    return status


def process_row(row: Dict[str, str], args) -> Tuple[Dict[str, object], Dict[str, int], list[Dict[str, object]]]:
    stats: Dict[str, int] = Counter()
    plume_id = str(row.get("plume_id", "")).strip()
    out_row: Dict[str, object] = {
        "plume_id": plume_id,
        "latitude": row.get("latitude", ""),
        "longitude": row.get("longitude", ""),
        "datetime": row.get("datetime", ""),
        "has_s5p": row.get("has_s5p", ""),
        "S5p_path": row.get("S5p_path", ""),
        "s5p_minus90_path": row.get("s5p_minus90_path", ""),
        "s5p_minus360_path": row.get("s5p_minus360_path", ""),
        "nearest_iy": row.get("nearest_iy", ""),
        "nearest_ix": row.get("nearest_ix", ""),
        "pos_centers": row.get("pos_centers", ""),
    }
    qa_rows: list[Dict[str, object]] = []

    for sensor in SENSORS:
        spec = SENSOR_SPECS[sensor]
        image_cols = spec["image_cols"]
        out_image_cols = spec["out_image_cols"]
        image_paths = [row.get(c, "") for c in image_cols]
        valid_images = [Path(str(p)) for p in image_paths if has_value(p) and Path(str(p)).exists()]

        out_row[f"has_{sensor}"] = bool(len(valid_images) == len(image_cols))
        for src_col, dst_col in zip(image_cols, out_image_cols):
            out_row[dst_col] = row.get(src_col, "")

        ref_image = first_existing_path(row, image_cols)
        mask_out_col = spec["out_mask_col"]
        out_mask = args.out_root / "masks" / plume_id / spec["out_mask_name"]
        out_row[mask_out_col] = ""

        qa = {
            "plume_id": plume_id,
            "sensor": sensor,
            "ref_image": str(ref_image or ""),
            "source_mask": "",
            "out_mask": str(out_mask),
            "status": "",
            "reason": "",
        }

        if ref_image is None:
            stats[f"{sensor}_skip_no_ref_image"] += 1
            qa["status"] = "skip"
            qa["reason"] = "no_ref_image"
            qa_rows.append(qa)
            continue

        candidates = source_mask_candidates(row, sensor, args.raw_mask_root, args.emit_mask_root)
        if not candidates:
            stats[f"{sensor}_skip_no_source_mask"] += 1
            qa["status"] = "skip"
            qa["reason"] = "missing_source_mask"
            qa_rows.append(qa)
            continue

        failures = []
        for src_mask, src_kind in candidates:
            qa["source_mask"] = str(src_mask)
            try:
                status = reproject_or_copy_binary(src_mask, ref_image, out_mask, src_kind, overwrite=args.overwrite)
                out_row[mask_out_col] = str(out_mask)
                stats[f"{sensor}_{status}"] += 1
                qa["status"] = "ok"
                qa["reason"] = status
                break
            except Exception as e:
                failures.append(f"{src_kind}:{str(e)[:180]}")
        else:
            stats[f"{sensor}_mask_failed"] += 1
            qa["status"] = "fail"
            qa["reason"] = " | ".join(failures)[:400]
        qa_rows.append(qa)

    return out_row, dict(stats), qa_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate sensor-aligned binary 512x512 masks and raw manifest.")
    p.add_argument("--master_csv", type=Path, default=Path("preprocess_dataset_multisensor/master_multisensor_outer_join.csv"))
    p.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--out_csv", type=Path, default=None)
    p.add_argument("--qa_csv", type=Path, default=None)
    p.add_argument("--raw_mask_root", type=Path, default=RAW_MASK_ROOT)
    p.add_argument("--emit_mask_root", type=Path, default=EMIT_MASK_ROOT)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_csv or (args.out_root / "manifest_raw512.csv")
    qa_csv = args.qa_csv or (args.out_root / "prepare_raw512_masks_log.csv")

    _, rows = read_csv_rows(args.master_csv)
    if args.limit > 0:
        rows = rows[: args.limit]

    out_rows = []
    qa_rows = []
    stats: Counter = Counter()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for i, (out_row, local_stats, local_qa) in enumerate(ex.map(lambda r: process_row(r, args), rows), start=1):
            out_rows.append(out_row)
            qa_rows.extend(local_qa)
            stats.update(local_stats)
            if i % 500 == 0 or i == len(rows):
                print(f"[progress] rows={i}/{len(rows)}", flush=True)

    fieldnames = [
        "plume_id",
        "latitude",
        "longitude",
        "datetime",
        "has_s2",
        "has_l89",
        "has_emit",
        "has_s5p",
        "s2_0_512_path",
        "s2_-90_512_path",
        "s2_-360_512_path",
        "s2_mask_512_path",
        "l89_0_512_path",
        "l89_-90_512_path",
        "l89_-360_512_path",
        "l89_mask_512_path",
        "emit_0_512_path",
        "emit_-90_512_path",
        "emit_-180_512_path",
        "emit_mask_512_path",
        "S5p_path",
        "s5p_minus90_path",
        "s5p_minus360_path",
        "nearest_iy",
        "nearest_ix",
        "pos_centers",
    ]
    write_csv_rows(out_csv, fieldnames, out_rows)
    write_csv_rows(qa_csv, ["plume_id", "sensor", "ref_image", "source_mask", "out_mask", "status", "reason"], qa_rows)

    print(f"saved_manifest: {out_csv}")
    print(f"saved_log: {qa_csv}")
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")


if __name__ == "__main__":
    main()
