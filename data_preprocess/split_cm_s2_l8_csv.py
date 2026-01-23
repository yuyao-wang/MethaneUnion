"\"\"\"Create temporal and spatial train/test splits for CM S2/L8 crops.\"\"\""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data_csv"
DEFAULT_RATIO = 0.8
DEFAULT_PRECISION = 5
DEFAULT_SEED = 42


def chronological_split(df: pd.DataFrame, ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["image_timestamp"])
    data = data.sort_values("timestamp")
    cutoff = int(len(data) * ratio)
    cutoff = min(max(cutoff, 1), len(data) - 1) if len(data) > 1 else len(data)
    train = data.iloc[:cutoff].drop(columns="timestamp")
    test = data.iloc[cutoff:].drop(columns="timestamp")
    if test.empty:
        test = train.iloc[[-1]].copy()
        train = train.iloc[:-1]
    return train, test


def spatial_split(
    df: pd.DataFrame,
    ratio: float,
    precision: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = df.copy()
    data["site_id"] = (
        data["plume_latitude"].round(precision).astype(str)
        + "_"
        + data["plume_longitude"].round(precision).astype(str)
    )
    group_sizes = data.groupby("site_id").size().rename("count").reset_index()
    rng = np.random.default_rng(seed)
    site_ids = group_sizes["site_id"].to_numpy()
    rng.shuffle(site_ids)
    count_map: Dict[str, int] = dict(zip(group_sizes["site_id"], group_sizes["count"]))
    total = sum(count_map.values())
    threshold = ratio * total
    running = 0
    train_sites = []
    for site in site_ids:
        if running < threshold:
            train_sites.append(site)
            running += count_map[site]
        else:
            break
    if not train_sites:
        train_sites.append(site_ids[0])
    train_mask = data["site_id"].isin(train_sites)
    train = data[train_mask].drop(columns="site_id")
    test = data[~train_mask].drop(columns="site_id")
    if test.empty:
        last_site = train_sites.pop()
        move_mask = data["site_id"] == last_site
        test = data[move_mask].drop(columns="site_id")
        train = data[~move_mask].drop(columns="site_id")
    return train, test


def write_split(df: pd.DataFrame, base_name: str, suffix: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{base_name}_{suffix}.csv"
    df.to_csv(out_path, index=False)
    print(f"[INFO] wrote {len(df):5d} rows -> {out_path}")


def process_file(
    csv_name: str,
    args: argparse.Namespace,
) -> None:
    csv_path = DATA_DIR / csv_name
    if not csv_path.exists():
        print(f"[WARN] missing {csv_path}, skipping")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[WARN] {csv_path} is empty, skipping")
        return
    print(f"[INFO] splitting {csv_path} ({len(df)} rows)")
    time_train, time_test = chronological_split(df, args.train_ratio)
    space_train, space_test = spatial_split(df, args.train_ratio, args.precision, args.seed)
    base_name = csv_path.stem
    write_split(time_train, base_name, "time_train", DATA_DIR)
    write_split(time_test, base_name, "time_test", DATA_DIR)
    write_split(space_train, base_name, "space_train", DATA_DIR)
    write_split(space_test, base_name, "space_test", DATA_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate temporal and spatial train/test splits for CM S2/L8 datasets."
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_RATIO,
        help="Fraction of samples (or site pixels) to allocate to the training split.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=DEFAULT_PRECISION,
        help="Decimal precision used to group locations for spatial splits.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for shuffling site IDs during spatial split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("CM_s2_l8_s2.csv", "CM_s2_l8_l89.csv"):
        process_file(name, args)


if __name__ == "__main__":
    main()
