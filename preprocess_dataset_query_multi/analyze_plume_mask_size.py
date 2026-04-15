import argparse
import csv
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio

from common import has_value, read_csv_rows, write_csv_rows


GSD = {
    "s2": 10.0,
    "l89": 30.0,
    "emit": 60.0,
}

MASK_COLS = {
    "s2": "s2_mask_512_path",
    "l89": "l89_mask_512_path",
    "emit": "emit_mask_512_path",
}


def mask_metrics(row: Dict[str, str], sensor: str) -> Dict[str, object]:
    pid = row.get("plume_id", "")
    col = MASK_COLS[sensor]
    out: Dict[str, object] = {
        "plume_id": pid,
        "sensor": sensor,
        "mask_path": row.get(col, ""),
        "status": "missing",
        "positive_pixels": 0,
        "positive_fraction": 0.0,
        "bbox_width_px": "",
        "bbox_height_px": "",
        "bbox_max_px": "",
        "bbox_width_m": "",
        "bbox_height_m": "",
        "bbox_max_m": "",
        "sqrt_area_m": "",
        "area_m2": "",
    }
    if not has_value(row.get(col, "")):
        return out
    path = Path(str(row[col]))
    if not path.exists():
        out["status"] = "missing_file"
        return out
    try:
        with rasterio.open(path) as ds:
            arr = ds.read(1)
        mask = np.isfinite(arr) & (arr > 0)
        pos = int(mask.sum())
        out["positive_pixels"] = pos
        out["positive_fraction"] = pos / int(mask.size) if mask.size else 0.0
        if pos == 0:
            out["status"] = "empty"
            return out
        ys, xs = np.where(mask)
        width_px = int(xs.max() - xs.min() + 1)
        height_px = int(ys.max() - ys.min() + 1)
        max_px = max(width_px, height_px)
        gsd = GSD[sensor]
        area_m2 = pos * gsd * gsd
        out.update(
            {
                "status": "ok",
                "bbox_width_px": width_px,
                "bbox_height_px": height_px,
                "bbox_max_px": max_px,
                "bbox_width_m": width_px * gsd,
                "bbox_height_m": height_px * gsd,
                "bbox_max_m": max_px * gsd,
                "sqrt_area_m": float(np.sqrt(area_m2)),
                "area_m2": area_m2,
            }
        )
        return out
    except Exception as e:
        out["status"] = f"error:{str(e)[:160]}"
        return out


def process_row(row: Dict[str, str]) -> List[Dict[str, object]]:
    return [mask_metrics(row, sensor) for sensor in MASK_COLS]


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def write_summary(path: Path, rows: List[Dict[str, object]]) -> None:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        grouped[str(r["sensor"])].append(r)

    summary_rows = []
    for sensor, rs in sorted(grouped.items()):
        ok = [r for r in rs if r["status"] == "ok"]
        empty = [r for r in rs if r["status"] == "empty"]
        vals_bbox = [float(r["bbox_max_m"]) for r in ok]
        vals_sqrt = [float(r["sqrt_area_m"]) for r in ok]
        vals_pos = [float(r["positive_pixels"]) for r in ok]
        row = {
            "sensor": sensor,
            "rows": len(rs),
            "ok": len(ok),
            "empty": len(empty),
            "missing_or_error": len(rs) - len(ok) - len(empty),
        }
        for q in [50, 75, 90, 95, 99, 100]:
            row[f"bbox_max_m_p{q}"] = percentile(vals_bbox, q)
            row[f"sqrt_area_m_p{q}"] = percentile(vals_sqrt, q)
            row[f"positive_pixels_p{q}"] = percentile(vals_pos, q)
        summary_rows.append(row)

    fields = list(summary_rows[0].keys()) if summary_rows else ["sensor"]
    write_csv_rows(path, fields, summary_rows)

    print(f"saved_summary_csv: {path}")
    for row in summary_rows:
        print(
            f"{row['sensor']}: ok={row['ok']} empty={row['empty']} "
            f"bbox_max_m p50={row['bbox_max_m_p50']:.1f} "
            f"p90={row['bbox_max_m_p90']:.1f} "
            f"p95={row['bbox_max_m_p95']:.1f} "
            f"p99={row['bbox_max_m_p99']:.1f}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze binary plume mask sizes in raw512 manifest.")
    p.add_argument("--manifest_csv", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--summary_csv", type=Path, required=True)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--max_rows", type=int, default=0)
    return p.parse_args()


def main() -> None:
    _, rows = read_csv_rows(args.manifest_csv)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    out_rows: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for i, result in enumerate(ex.map(process_row, rows), start=1):
            out_rows.extend(result)
            if i % 1000 == 0 or i == len(rows):
                print(f"[progress] rows={i}/{len(rows)}", flush=True)

    fields = [
        "plume_id",
        "sensor",
        "mask_path",
        "status",
        "positive_pixels",
        "positive_fraction",
        "bbox_width_px",
        "bbox_height_px",
        "bbox_max_px",
        "bbox_width_m",
        "bbox_height_m",
        "bbox_max_m",
        "sqrt_area_m",
        "area_m2",
    ]
    write_csv_rows(args.out_csv, fields, out_rows)
    print(f"saved_out_csv: {args.out_csv}")
    write_summary(args.summary_csv, out_rows)


if __name__ == "__main__":
    args = parse_args()
    main()
