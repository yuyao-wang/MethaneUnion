import argparse
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import rasterio


SENSORS = ("s2", "l89", "emit")


def has_path(value: str) -> bool:
    s = str(value or "").strip()
    return s != "" and s.lower() != "nan"


def mask_has_positive(path: Path) -> bool | None:
    if not path.exists():
        return None
    with rasterio.open(path) as ds:
        arr = ds.read(1)
    return bool(arr.max() > 0)


def process_manifest(manifest_csv: Path, out_csv: Path, workers: int = 16) -> Counter:
    stats: Counter = Counter()
    out_rows: list[dict[str, str]] = []
    rows: list[dict[str, str]] = []
    unique_paths: set[Path] = set()

    with manifest_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise RuntimeError(f"No header found in {manifest_csv}")

        for row in reader:
            rows.append(row)
            for sensor in SENSORS:
                plume_col = f"{sensor}_plume_path"
                plume_path = str(row.get(plume_col, "")).strip()
                if has_path(plume_path):
                    unique_paths.add(Path(plume_path))

    print(f"[progress] loaded_rows={len(rows)} unique_mask_paths={len(unique_paths)}", flush=True)

    cache: dict[Path, bool | None] = {}
    path_list = sorted(unique_paths)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for i, (path, state) in enumerate(zip(path_list, ex.map(mask_has_positive, path_list, chunksize=64)), start=1):
            cache[path] = state
            if i % 5000 == 0 or i == len(path_list):
                print(f"[progress] mask_scan={i}/{len(path_list)}", flush=True)

    for i, row in enumerate(rows, start=1):
        stats["rows_seen"] += 1
        label = str(row.get("label", "")).strip()

        sensor_states: dict[str, bool] = {}
        for sensor in SENSORS:
            plume_col = f"{sensor}_plume_path"
            plume_path = str(row.get(plume_col, "")).strip()
            if not has_path(plume_path):
                continue
            state = cache.get(Path(plume_path))
            if state is None:
                stats[f"{sensor}_mask_missing_file"] += 1
                continue
            sensor_states[sensor] = state

        sensor_count = len(sensor_states)
        stats[f"rows_with_{sensor_count}_sensor_masks"] += 1

        if sensor_count == 0:
            out_rows.append(row)
            stats["rows_kept_no_sensor_masks"] += 1
        else:
            positives = sum(1 for v in sensor_states.values() if v)
            negatives = sensor_count - positives

            if sensor_count >= 2:
                if positives == sensor_count:
                    out_row = dict(row)
                    if label != "1":
                        out_row["label"] = "1"
                        stats["rows_relabel_to_1_multisensor_all_positive"] += 1
                    else:
                        stats["rows_already_1_multisensor_all_positive"] += 1
                    out_rows.append(out_row)
                    stats["rows_kept_multisensor_all_positive"] += 1
                elif positives > 0 and negatives > 0:
                    stats["rows_dropped_multisensor_conflict"] += 1
                else:
                    out_rows.append(row)
                    stats["rows_kept_multisensor_all_zero"] += 1
                    if label == "1":
                        stats["rows_multisensor_all_zero_label1_unmodified"] += 1
            else:
                out_rows.append(row)
                only_positive = next(iter(sensor_states.values()))
                if (label == "0" and only_positive) or (label == "1" and not only_positive):
                    stats["rows_single_sensor_label_conflict"] += 1
                    sensor = next(iter(sensor_states.keys()))
                    stats[f"rows_single_sensor_label_conflict_{sensor}"] += 1
                else:
                    stats["rows_single_sensor_label_consistent"] += 1

        if i % 10000 == 0 or i == len(rows):
            print(f"[progress] row_scan={i}/{len(rows)} kept={len(out_rows)}", flush=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    stats["rows_written"] = len(out_rows)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean manifest labels using S2/L89/EMIT mask consensus.")
    parser.add_argument("--manifest_csv", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stats = process_manifest(args.manifest_csv, args.out_csv, workers=max(1, int(args.workers)))
    print(f"manifest: {args.manifest_csv}")
    print(f"out_csv: {args.out_csv}")
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")
