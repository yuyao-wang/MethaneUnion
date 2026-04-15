import argparse
import csv
import hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


SENSORS = ("s2", "l89", "emit")


def has_path(value: str) -> bool:
    s = str(value or "").strip()
    return s != "" and s.lower() != "nan"


def read_mask_state(path: Path) -> bool | None:
    if not path.exists():
        return None
    with rasterio.open(path) as ds:
        arr = ds.read(1)
    return bool(arr.max() > 0)


def write_binary_mask(path: Path) -> bool:
    with rasterio.open(path) as ds:
        arr = ds.read(1)
    binary = (arr > 0).astype(np.float32)
    changed = not np.array_equal(arr.astype(np.float32), binary)
    profile = {
        "driver": "GTiff",
        "height": binary.shape[0],
        "width": binary.shape[1],
        "count": 1,
        "dtype": "float32",
        "transform": from_origin(0, 0, 1, 1),
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": min(256, binary.shape[1]),
        "blockysize": min(256, binary.shape[0]),
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(binary, 1)
    return changed


def parse_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise RuntimeError(f"No header found in {csv_path}")
    return fieldnames, rows


def compute_mask_cache(rows: list[dict[str, str]], workers: int) -> dict[Path, bool | None]:
    unique_paths: set[Path] = set()
    for row in rows:
        for sensor in SENSORS:
            plume_path = str(row.get(f"{sensor}_plume_path", "")).strip()
            if has_path(plume_path):
                unique_paths.add(Path(plume_path))
    print(f"[progress] unique_mask_paths={len(unique_paths)}", flush=True)
    cache: dict[Path, bool | None] = {}
    paths = sorted(unique_paths)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for i, (path, state) in enumerate(zip(paths, ex.map(read_mask_state, paths, chunksize=64)), start=1):
            cache[path] = state
            if i % 5000 == 0 or i == len(paths):
                print(f"[progress] mask_state_scan={i}/{len(paths)}", flush=True)
    return cache


def clean_rows(rows: list[dict[str, str]], mask_cache: dict[Path, bool | None]) -> tuple[list[dict[str, str]], Counter]:
    stats: Counter = Counter()
    out_rows: list[dict[str, str]] = []

    for i, row in enumerate(rows, start=1):
        stats["rows_seen"] += 1
        label = str(row.get("label", "")).strip()
        sensor_states: dict[str, bool] = {}

        for sensor in SENSORS:
            plume_path = str(row.get(f"{sensor}_plume_path", "")).strip()
            if not has_path(plume_path):
                continue
            state = mask_cache.get(Path(plume_path))
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
                    out_row = dict(row)
                    if label != "0":
                        out_row["label"] = "0"
                        stats["rows_relabel_to_0_multisensor_all_zero"] += 1
                    out_rows.append(out_row)
                    stats["rows_kept_multisensor_all_zero"] += 1
            else:
                out_row = dict(row)
                sensor = next(iter(sensor_states.keys()))
                only_positive = next(iter(sensor_states.values()))
                corrected = "1" if only_positive else "0"
                if label != corrected:
                    out_row["label"] = corrected
                    stats["rows_single_sensor_label_conflict_fixed"] += 1
                    stats[f"rows_single_sensor_label_conflict_fixed_{sensor}"] += 1
                else:
                    stats["rows_single_sensor_label_consistent"] += 1
                out_rows.append(out_row)

        if i % 10000 == 0 or i == len(rows):
            print(f"[progress] clean_rows={i}/{len(rows)} kept={len(out_rows)}", flush=True)

    return out_rows, stats


def split_train_test(rows: list[dict[str, str]], train_ratio: float) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    threshold = max(0.0, min(1.0, float(train_ratio)))
    train_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []
    for row in rows:
        plume_id = str(row.get("plume_id", "")).strip()
        key = plume_id or str(row.get("id", "")).strip()
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        frac = int(digest[:8], 16) / 0xFFFFFFFF
        if frac < threshold:
            train_rows.append(row)
        else:
            test_rows.append(row)
    return train_rows, test_rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_labels(rows: list[dict[str, str]]) -> Counter:
    c = Counter()
    c["rows"] = len(rows)
    for row in rows:
        c[f"label_{str(row.get('label', '')).strip()}"] += 1
    return c


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Binarize EMIT plume masks and generate cleaned train/test CSVs.")
    p.add_argument("--input_csv", type=Path, required=True)
    p.add_argument("--train_out_csv", type=Path, required=True)
    p.add_argument("--test_out_csv", type=Path, required=True)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--train_ratio", type=float, default=0.8)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fieldnames, rows = parse_rows(args.input_csv)

    emit_paths = sorted(
        {
            Path(str(row.get("emit_plume_path", "")).strip())
            for row in rows
            if has_path(row.get("emit_plume_path", ""))
        }
    )
    print(f"[progress] emit_mask_paths={len(emit_paths)}", flush=True)
    emit_changed = 0
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        for i, changed in enumerate(ex.map(write_binary_mask, emit_paths, chunksize=32), start=1):
            if changed:
                emit_changed += 1
            if i % 1000 == 0 or i == len(emit_paths):
                print(f"[progress] emit_binarized={i}/{len(emit_paths)} changed={emit_changed}", flush=True)

    mask_cache = compute_mask_cache(rows, workers=max(1, int(args.workers)))
    cleaned_rows, stats = clean_rows(rows, mask_cache)
    train_rows, test_rows = split_train_test(cleaned_rows, train_ratio=float(args.train_ratio))

    write_csv(args.train_out_csv, fieldnames, train_rows)
    write_csv(args.test_out_csv, fieldnames, test_rows)

    train_stats = summarize_labels(train_rows)
    test_stats = summarize_labels(test_rows)

    print(f"input_csv: {args.input_csv}")
    print(f"train_out_csv: {args.train_out_csv}")
    print(f"test_out_csv: {args.test_out_csv}")
    print(f"emit_masks_changed: {emit_changed}")
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")
    for key in sorted(train_stats):
        print(f"train_{key}: {train_stats[key]}")
    for key in sorted(test_stats):
        print(f"test_{key}: {test_stats[key]}")
