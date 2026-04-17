import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Tuple

import rasterio

from common import SENSOR_SPECS, SENSORS, has_value, mask_stats, read_csv_rows, write_csv_rows


def image_stats(paths: list[str]) -> Dict[str, object]:
    out: Dict[str, object] = {
        "image_triplet_exists": False,
        "image_all_512": False,
        "image_shapes": "",
        "image_dtypes": "",
        "image_reason": "",
    }
    valid_paths = [Path(str(p)) for p in paths if has_value(p)]
    if len(valid_paths) != len(paths):
        out["image_reason"] = "missing_path_value"
        return out
    missing = [str(p) for p in valid_paths if not p.exists()]
    if missing:
        out["image_reason"] = "missing_file:" + "|".join(missing[:3])
        return out

    shapes = []
    dtypes = []
    try:
        for p in valid_paths:
            with rasterio.open(p) as ds:
                shapes.append(f"{ds.count}x{ds.height}x{ds.width}")
                dtypes.append(",".join(ds.dtypes))
        out["image_triplet_exists"] = True
        out["image_shapes"] = "|".join(shapes)
        out["image_dtypes"] = "|".join(dtypes)
        out["image_all_512"] = all(s.endswith("x512x512") for s in shapes)
        if not out["image_all_512"]:
            out["image_reason"] = "not_all_512"
    except Exception as e:
        out["image_reason"] = str(e)[:400]
    return out


def check_sensor(row: Dict[str, str], sensor: str) -> Dict[str, object]:
    spec = SENSOR_SPECS[sensor]
    img = image_stats([row.get(c, "") for c in spec["out_image_cols"]])
    mask_path = Path(str(row.get(spec["out_mask_col"], ""))) if has_value(row.get(spec["out_mask_col"], "")) else None
    m = mask_stats(mask_path) if mask_path is not None else mask_stats(Path("__missing__"))

    positive_pixels = m.get("mask_positive_pixels")
    try:
        positive_pixels_int = int(positive_pixels)
    except Exception:
        positive_pixels_int = 0

    mask_ok = (
        bool(m["mask_exists"])
        and int(m["mask_height"] or 0) == 512
        and int(m["mask_width"] or 0) == 512
        and bool(m["mask_is_binary"])
        and positive_pixels_int >= 0
    )

    # We allow zero-positive masks to pass as "structurally valid"; positive
    # count is stored for review because weak labels should not be derived from
    # mask overlap in this pipeline.
    sensor_ok = bool(img["image_triplet_exists"] and img["image_all_512"] and mask_ok)
    reason = "ok" if sensor_ok else []
    if not sensor_ok:
        parts = []
        if not img["image_triplet_exists"]:
            parts.append(f"image:{img['image_reason']}")
        elif not img["image_all_512"]:
            parts.append("image:not_all_512")
        if not m["mask_exists"]:
            parts.append("mask:missing")
        elif int(m["mask_height"] or 0) != 512 or int(m["mask_width"] or 0) != 512:
            parts.append("mask:not_512")
        elif not m["mask_is_binary"]:
            parts.append("mask:not_binary")
        reason = ";".join(parts)

    return {
        "plume_id": row.get("plume_id", ""),
        "sensor": sensor,
        "sensor_ok": sensor_ok,
        "reason": reason,
        "image_triplet_exists": img["image_triplet_exists"],
        "image_all_512": img["image_all_512"],
        "image_shapes": img["image_shapes"],
        "image_dtypes": img["image_dtypes"],
        "mask_path": str(mask_path or ""),
        **m,
    }


def _numeric(value: object) -> bool:
    if not has_value(value):
        return False
    try:
        float(str(value))
        return True
    except Exception:
        return False


def check_s5p(row: Dict[str, str]) -> Dict[str, object]:
    path_cols = ["S5p_path", "s5p_minus90_path", "s5p_minus360_path"]
    paths = [Path(str(row.get(c, ""))) for c in path_cols if has_value(row.get(c, ""))]
    missing_reasons = []
    if len(paths) != len(path_cols):
        missing_reasons.append("missing_path_value")
    else:
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            missing_reasons.append("missing_file:" + "|".join(missing[:3]))

    if not _numeric(row.get("nearest_iy", "")) or not _numeric(row.get("nearest_ix", "")):
        missing_reasons.append("invalid_nearest_ixiy")

    ok = not missing_reasons
    return {
        "plume_id": row.get("plume_id", ""),
        "sensor": "s5p",
        "sensor_ok": ok,
        "reason": "ok" if ok else ";".join(missing_reasons),
        "image_triplet_exists": ok,
        "image_all_512": "",
        "image_shapes": "netcdf_triplet" if ok else "",
        "image_dtypes": "",
        "mask_path": "",
        "mask_exists": "",
        "mask_height": "",
        "mask_width": "",
        "mask_dtype": "",
        "mask_is_binary": "",
        "mask_positive_pixels": "",
        "mask_positive_fraction": "",
    }


def check_row(row: Dict[str, str]) -> Tuple[Dict[str, str], list[Dict[str, object]], Dict[str, int]]:
    qa_rows = [check_sensor(row, sensor) for sensor in SENSORS]
    qa_rows.append(check_s5p(row))
    stats: Counter = Counter()
    ok_sensors = [q["sensor"] for q in qa_rows if q["sensor_ok"]]
    ok_raster_sensors = [q["sensor"] for q in qa_rows if q["sensor_ok"] and q["sensor"] in SENSORS]
    out = dict(row)
    out["valid_raster_sensors"] = ";".join(ok_raster_sensors)
    out["valid_s5p"] = "s5p" in ok_sensors
    out["valid_sensors"] = ";".join(ok_sensors)
    out["num_valid_sensors"] = len(ok_sensors)
    out["raw512_ok"] = bool(ok_raster_sensors)
    out["query_source_ok"] = bool(ok_sensors)
    stats["rows_seen"] += 1
    if ok_sensors:
        stats["rows_clean_kept"] += 1
    else:
        stats["rows_clean_dropped"] += 1
    for sensor in ok_sensors:
        stats[f"sensor_ok_{sensor}"] += 1
    return out, qa_rows, dict(stats)


def row_signature(row: Dict[str, object], fields: list[str]) -> Tuple[str, ...]:
    return tuple(str(row.get(f, "")) for f in fields)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QA raw512 image/mask manifest and write a clean manifest.")
    p.add_argument("--manifest_csv", type=Path, required=True)
    p.add_argument("--clean_csv", type=Path, required=True)
    p.add_argument("--qa_csv", type=Path, required=True)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--clean_keep_mode",
        type=str,
        default="query_source_ok",
        choices=["query_source_ok", "raster_only"],
        help=(
            "query_source_ok: keep rows with at least one valid sensor (including S5P-only); "
            "raster_only: keep rows with at least one valid raster sensor (legacy behavior)."
        ),
    )
    p.add_argument(
        "--merge_into_existing_clean",
        action="store_true",
        help=(
            "If clean_csv already exists, append only newly qualified rows and keep existing row order."
        ),
    )
    return p.parse_args()


def main() -> None:
    fieldnames, rows = read_csv_rows(args.manifest_csv)
    if args.limit > 0:
        rows = rows[: args.limit]

    clean_rows = []
    qa_rows = []
    stats: Counter = Counter()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for i, (out_row, row_qa, local_stats) in enumerate(ex.map(check_row, rows), start=1):
            stats.update(local_stats)
            qa_rows.extend(row_qa)
            keep = bool(out_row["query_source_ok"]) if args.clean_keep_mode == "query_source_ok" else bool(out_row["raw512_ok"])
            if keep:
                clean_rows.append(out_row)
            if i % 1000 == 0 or i == len(rows):
                print(f"[progress] qa_rows={i}/{len(rows)} clean={len(clean_rows)}", flush=True)

    if args.merge_into_existing_clean and args.clean_csv.exists():
        existing_fields, existing_rows = read_csv_rows(args.clean_csv)
        seen = {row_signature(r, fieldnames) for r in existing_rows}
        merged_rows = list(existing_rows)
        added = 0
        skipped = 0
        for r in clean_rows:
            sig = row_signature(r, fieldnames)
            if sig in seen:
                skipped += 1
                continue
            seen.add(sig)
            merged_rows.append(r)
            added += 1
        clean_rows = merged_rows
        stats["clean_existing_rows"] = len(existing_rows)
        stats["clean_added_rows"] = added
        stats["clean_skipped_existing"] = skipped
        clean_fields = list(existing_fields)
    else:
        clean_fields = list(fieldnames)

    for c in ["valid_raster_sensors", "valid_s5p", "valid_sensors", "num_valid_sensors", "raw512_ok", "query_source_ok"]:
        if c not in clean_fields:
            clean_fields.append(c)
    qa_fields = [
        "plume_id",
        "sensor",
        "sensor_ok",
        "reason",
        "image_triplet_exists",
        "image_all_512",
        "image_shapes",
        "image_dtypes",
        "mask_path",
        "mask_exists",
        "mask_height",
        "mask_width",
        "mask_dtype",
        "mask_is_binary",
        "mask_positive_pixels",
        "mask_positive_fraction",
    ]

    write_csv_rows(args.clean_csv, clean_fields, clean_rows)
    write_csv_rows(args.qa_csv, qa_fields, qa_rows)
    print(f"saved_clean_csv: {args.clean_csv}")
    print(f"saved_qa_csv: {args.qa_csv}")
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")


if __name__ == "__main__":
    args = parse_args()
    main()
