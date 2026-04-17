import argparse
import csv
import json
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple


SENSOR_RULES = {
    "s2": {"cols": ["s2_0_path"], "train_max": date(2025, 5, 16), "test_min": date(2025, 5, 18)},
    "l89": {"cols": ["l89_0_path"], "train_max": date(2025, 5, 10), "test_min": date(2025, 5, 11)},
    "emit": {"cols": ["emit_0_path"], "train_max": date(2025, 2, 11), "test_min": date(2025, 2, 12)},
    # Backward-compatible: old manifests use s5p_0_path for NPZ path.
    "s5p": {"cols": ["s5p_npz_path", "s5p_0_path"], "train_max": date(2025, 7, 16), "test_min": date(2025, 7, 21)},
}
DEFAULT_CUTOFFS = {
    "s2": date(2025, 7, 27),
    "l89": date(2025, 7, 30),
    "emit": date(2025, 3, 8),
    "s5p": date(2025, 10, 19),
}


def has_value(v: object) -> bool:
    s = str(v or "").strip()
    return s != "" and s.lower() not in {"nan", "none"}


def parse_row_date(row: Dict[str, str]) -> date:
    raw = str(row.get("datetime", "") or "").strip()
    if not raw:
        raise ValueError("empty_datetime")
    s = raw.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s).date()
    except Exception:
        pass
    if "T" in s:
        return date.fromisoformat(s.split("T", 1)[0])
    return date.fromisoformat(s.split(" ", 1)[0])


def present_sensors(row: Dict[str, str]) -> List[str]:
    out: List[str] = []
    for sensor, cfg in SENSOR_RULES.items():
        cols = list(cfg.get("cols", []))
        if any(has_value(row.get(col, "")) for col in cols):
            out.append(sensor)
    return out


def sensor_side(sensor: str, d: date, mode: str, cutoffs: Dict[str, date]) -> str:
    if mode == "cutoff_no_gap":
        return "test" if d > cutoffs[sensor] else "train"
    rule = SENSOR_RULES[sensor]
    if d <= rule["train_max"]:
        return "train"
    if d >= rule["test_min"]:
        return "test"
    return "gap"


def write_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Temporal split for unified multisensor query manifest.")
    p.add_argument("--in_csv", type=Path, required=True)
    p.add_argument("--out_train_csv", type=Path, required=True)
    p.add_argument("--out_test_csv", type=Path, required=True)
    p.add_argument("--out_summary_json", type=Path, required=True)
    p.add_argument(
        "--mode",
        type=str,
        default="cutoff_no_gap",
        choices=["strict", "loose_test_priority", "cutoff_no_gap"],
        help=(
            "strict: all present sensors must be all-train or all-test; else gap. "
            "loose_test_priority: if any present sensor is test -> test; "
            "else if any present sensor is train -> train; else gap. "
            "cutoff_no_gap: per-sensor cutoff, if row_date > cutoff then test else train (no gap)."
        ),
    )
    p.add_argument("--cutoff_s2", type=date.fromisoformat, default=DEFAULT_CUTOFFS["s2"])
    p.add_argument("--cutoff_l89", type=date.fromisoformat, default=DEFAULT_CUTOFFS["l89"])
    p.add_argument("--cutoff_emit", type=date.fromisoformat, default=DEFAULT_CUTOFFS["emit"])
    p.add_argument("--cutoff_s5p", type=date.fromisoformat, default=DEFAULT_CUTOFFS["s5p"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cutoffs = {
        "s2": args.cutoff_s2,
        "l89": args.cutoff_l89,
        "emit": args.cutoff_emit,
        "s5p": args.cutoff_s5p,
    }

    with args.in_csv.open(newline="") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        if not fieldnames:
            raise RuntimeError(f"No header in {args.in_csv}")
        rows = list(r)

    train_rows: List[Dict[str, str]] = []
    test_rows: List[Dict[str, str]] = []
    gap_count = 0

    per_sensor = {s: defaultdict(int) for s in SENSOR_RULES}
    split_counts = defaultdict(int)
    leak_sets = {s: {"train": set(), "test": set()} for s in SENSOR_RULES}

    for row in rows:
        sensors = present_sensors(row)
        if not sensors:
            split_counts["gap_no_sensor"] += 1
            gap_count += 1
            continue

        try:
            d = parse_row_date(row)
        except Exception:
            split_counts["gap_bad_datetime"] += 1
            gap_count += 1
            continue

        sides = [sensor_side(s, d, args.mode, cutoffs) for s in sensors]
        if args.mode == "strict":
            if all(x == "train" for x in sides):
                split = "train"
                train_rows.append(row)
            elif all(x == "test" for x in sides):
                split = "test"
                test_rows.append(row)
            else:
                split = "gap"
                gap_count += 1
                split_counts["gap_mixed_or_window"] += 1
        elif args.mode == "loose_test_priority":
            has_test = any(x == "test" for x in sides)
            has_train = any(x == "train" for x in sides)
            if has_test:
                split = "test"
                test_rows.append(row)
            elif has_train:
                split = "train"
                train_rows.append(row)
            else:
                split = "gap"
                gap_count += 1
                split_counts["gap_mixed_or_window"] += 1
        else:
            has_test = any(x == "test" for x in sides)
            split = "test" if has_test else "train"
            if split == "test":
                test_rows.append(row)
            else:
                train_rows.append(row)

        for s in sensors:
            per_sensor[s]["total"] += 1
            if split == "train":
                per_sensor[s]["train"] += 1
            elif split == "test":
                per_sensor[s]["test"] += 1
            else:
                per_sensor[s]["gap"] += 1

        pid = str(row.get("plume_id", "") or "").strip()
        if pid:
            for s in sensors:
                if split == "train":
                    leak_sets[s]["train"].add(pid)
                elif split == "test":
                    leak_sets[s]["test"].add(pid)

    split_counts["train"] = len(train_rows)
    split_counts["test"] = len(test_rows)
    split_counts["gap"] = gap_count
    split_counts["total"] = len(rows)

    summary: Dict[str, object] = {
        "input_csv": str(args.in_csv),
        "mode": args.mode,
        "cutoffs": {k: v.isoformat() for k, v in cutoffs.items()},
        "counts": dict(split_counts),
        "sensor": {},
        "leak_plume_ids": {},
    }
    for s in SENSOR_RULES:
        st = per_sensor[s]
        total = int(st["total"])
        train = int(st["train"])
        test = int(st["test"])
        gap = int(st["gap"])
        usable = train + test
        summary["sensor"][s] = {
            "total": total,
            "train": train,
            "test": test,
            "gap": gap,
            "train_ratio_total": (train / total) if total else 0.0,
            "train_ratio_usable": (train / usable) if usable else 0.0,
        }
        summary["leak_plume_ids"][s] = len(leak_sets[s]["train"] & leak_sets[s]["test"])

    write_rows(args.out_train_csv, fieldnames, train_rows)
    write_rows(args.out_test_csv, fieldnames, test_rows)
    args.out_summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"saved_train: {args.out_train_csv} ({len(train_rows)})")
    print(f"saved_test: {args.out_test_csv} ({len(test_rows)})")
    print(f"gap_count: {gap_count}")
    print(f"saved_summary: {args.out_summary_json}")
    for s in ("s2", "l89", "emit", "s5p"):
        x = summary["sensor"][s]
        print(
            f"{s}: total={x['total']} train={x['train']} test={x['test']} gap={x['gap']} "
            f"train_ratio_total={x['train_ratio_total']:.6f} "
            f"train_ratio_usable={x['train_ratio_usable']:.6f} "
            f"leak={summary['leak_plume_ids'][s]}"
        )


if __name__ == "__main__":
    main()
