import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

EMIT_SIM_DIR = Path(
    "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/EMIT_simulated_WV3_L2A_60resolution_NOnorm"
)


def _read_with_selected_columns(csv_path: Path, selected_cols: List[str], rename_map: Dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    keep = [c for c in selected_cols if c in df.columns]
    if "plume_id" not in keep:
        keep = ["plume_id"] + keep
    df = df[keep].copy()
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    # Safety: ensure one row per plume_id to avoid cartesian product on merge
    df = df.drop_duplicates(subset=["plume_id"], keep="first")
    return df


def _coalesce(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series([pd.NA] * len(df), index=df.index)
    return df[existing].bfill(axis=1).iloc[:, 0]


def _valid_str_series(s: pd.Series) -> pd.Series:
    txt = s.astype("string").str.strip()
    return txt.notna() & txt.ne("") & txt.str.lower().ne("nan")


def _compose_emit_paths(df: pd.DataFrame, emit_dir: Path) -> pd.DataFrame:
    out = df.copy()
    plume_id_txt = out["plume_id"].astype("string")

    g0_ok = _valid_str_series(out["emit_granule_id"]) if "emit_granule_id" in out.columns else pd.Series(False, index=out.index)
    g90_ok = _valid_str_series(out["emit_-90_granule_id"]) if "emit_-90_granule_id" in out.columns else pd.Series(False, index=out.index)
    g180_ok = _valid_str_series(out["emit_-180_granule_id"]) if "emit_-180_granule_id" in out.columns else pd.Series(False, index=out.index)

    out["emit_0_simulated_512_path"] = pd.NA
    out["emit_-90_simulated_512_path"] = pd.NA
    out["emit_-180_simulated_512_path"] = pd.NA

    out.loc[g0_ok, "emit_0_simulated_512_path"] = plume_id_txt[g0_ok].map(lambda x: str(emit_dir / f"{x}_sim_WV3.tif"))
    out.loc[g90_ok, "emit_-90_simulated_512_path"] = plume_id_txt[g90_ok].map(lambda x: str(emit_dir / f"{x}_-90_sim_WV3.tif"))
    out.loc[g180_ok, "emit_-180_simulated_512_path"] = plume_id_txt[g180_ok].map(lambda x: str(emit_dir / f"{x}_-180_sim_WV3.tif"))
    return out


def build_master_csv(s2_csv: Path, l89_csv: Path, emit_csv: Path, s5p_csv: Path, output_csv: Path) -> pd.DataFrame:
    s2_cols = [
        "plume_id",
        "plume_latitude",
        "plume_longitude",
        "datetime",
        "s2_0_std_512",
        "s2_-7_std_512",
        "s2_-90_std_512",
        "s2_-360_std_512",
        "resized_512x512_path",
    ]
    s2_rename = {
        "plume_latitude": "s2_plume_latitude",
        "plume_longitude": "s2_plume_longitude",
        "datetime": "s2_datetime",
        "resized_512x512_path": "s2_plume_mask_512_path",
    }

    l89_cols = [
        "plume_id",
        "plume_latitude",
        "plume_longitude",
        "datetime",
        "l89_0_std_512",
        "l89_-7_std_512",
        "l89_-90_std_512",
        "l89_-360_std_512",
        "mask_path",
    ]
    l89_rename = {
        "plume_latitude": "l89_plume_latitude",
        "plume_longitude": "l89_plume_longitude",
        "datetime": "l89_datetime",
        "mask_path": "l89_mask_512_path",
    }

    emit_cols = [
        "plume_id",
        "plume_latitude",
        "plume_longitude",
        "datetime",
        "simulated_512_path",
        "emit_granule_id",
        "emit_-90_granule_id",
        "emit_-180_granule_id",
        "has_emit",
    ]
    emit_rename = {
        "plume_latitude": "emit_plume_latitude",
        "plume_longitude": "emit_plume_longitude",
        "datetime": "emit_datetime",
        "has_emit": "emit_has_emit_flag",
    }

    s5p_cols = [
        "plume_id",
        "plume_time",
        "lat",
        "lon",
        "S5p_path",
        "s5p_minus90_path",
        "s5p_minus360_path",
        "nearest_iy",
        "nearest_ix",
        "pos_centers",
        "ch4_var",
    ]
    s5p_rename = {
        "plume_time": "s5p_datetime",
        "lat": "s5p_latitude",
        "lon": "s5p_longitude",
    }

    s2_df = _read_with_selected_columns(s2_csv, s2_cols, s2_rename)
    l89_df = _read_with_selected_columns(l89_csv, l89_cols, l89_rename)
    emit_df = _read_with_selected_columns(emit_csv, emit_cols, emit_rename)
    emit_df = _compose_emit_paths(emit_df, EMIT_SIM_DIR)
    s5p_df = _read_with_selected_columns(s5p_csv, s5p_cols, s5p_rename)

    master = s2_df.merge(l89_df, on="plume_id", how="outer")
    master = master.merge(emit_df, on="plume_id", how="outer")
    master = master.merge(s5p_df, on="plume_id", how="outer")

    master["latitude"] = _coalesce(
        master,
        ["s2_plume_latitude", "l89_plume_latitude", "emit_plume_latitude", "s5p_latitude"],
    )
    master["longitude"] = _coalesce(
        master,
        ["s2_plume_longitude", "l89_plume_longitude", "emit_plume_longitude", "s5p_longitude"],
    )
    master["datetime"] = _coalesce(
        master,
        ["s2_datetime", "l89_datetime", "emit_datetime", "s5p_datetime"],
    )

    s2_triplet = [c for c in ["s2_0_std_512", "s2_-90_std_512", "s2_-360_std_512"] if c in master.columns]
    l89_triplet = [c for c in ["l89_0_std_512", "l89_-90_std_512", "l89_-360_std_512"] if c in master.columns]
    emit_triplet = [
        c
        for c in ["emit_0_simulated_512_path", "emit_-90_simulated_512_path", "emit_-180_simulated_512_path"]
        if c in master.columns
    ]
    s5p_triplet = [c for c in ["S5p_path", "s5p_minus90_path", "s5p_minus360_path"] if c in master.columns]

    master["has_s2"] = master[s2_triplet].notna().all(axis=1) if s2_triplet else False
    master["has_l89"] = master[l89_triplet].notna().all(axis=1) if l89_triplet else False
    master["has_emit"] = master[emit_triplet].notna().all(axis=1) if emit_triplet else False
    master["has_s5p"] = master[s5p_triplet].notna().all(axis=1) if s5p_triplet else False

    # Keep rows where at least one sensor has a full 3-image triplet.
    keep_mask = master[["has_s2", "has_l89", "has_emit", "has_s5p"]].any(axis=1)
    master = master[keep_mask].copy()

    # Keep only unified geo-time fields; drop duplicated per-sensor variants.
    drop_geo_time_cols = [
        "s2_datetime",
        "l89_datetime",
        "emit_datetime",
        "s5p_datetime",
        "s2_plume_latitude",
        "s2_plume_longitude",
        "l89_plume_latitude",
        "l89_plume_longitude",
        "emit_plume_latitude",
        "emit_plume_longitude",
        "s5p_latitude",
        "s5p_longitude",
    ]
    master = master.drop(columns=[c for c in drop_geo_time_cols if c in master.columns])

    final_cols = [
        "plume_id",
        "latitude",
        "longitude",
        "datetime",
        "has_s2",
        "has_l89",
        "has_emit",
        "has_s5p",
        "s2_0_std_512",
        "s2_-90_std_512",
        "s2_-360_std_512",
        "s2_plume_mask_512_path",
        "l89_0_std_512",
        "l89_-90_std_512",
        "l89_-360_std_512",
        "l89_mask_512_path",
        "emit_0_simulated_512_path",
        "emit_-90_simulated_512_path",
        "emit_-180_simulated_512_path",
        "S5p_path",
        "s5p_minus90_path",
        "s5p_minus360_path",
        "nearest_iy",
        "nearest_ix",
        "pos_centers",
    ]
    for c in final_cols:
        if c not in master.columns:
            master[c] = pd.NA
    master = master[final_cols].sort_values("plume_id").reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_csv, index=False)
    return master


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Outer-join S2/L89/EMIT/S5P by plume_id to create a master CSV.")
    parser.add_argument("--s2_csv", type=Path, default=Path("preprocess_dataset_s2/CM_S2_L2A_gee90360_std512.csv"))
    parser.add_argument("--l89_csv", type=Path, default=Path("preprocess_dataset_L89/CM_L89_L2SR_std512.csv"))
    parser.add_argument("--emit_csv", type=Path, default=Path("preprocess_dataset_EMIT/merged_with_emit_tag.csv"))
    parser.add_argument("--s5p_csv", type=Path, default=Path("preprocess_dataset_s5p/s5p_all_OFFL_with_centers.csv"))
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("preprocess_dataset_multisensor/master_multisensor_outer_join.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    master = build_master_csv(args.s2_csv, args.l89_csv, args.emit_csv, args.s5p_csv, args.output_csv)
    print(f"Saved master CSV: {args.output_csv}")
    print(f"Rows: {len(master)}")
    print(
        "Coverage:",
        {
            "has_s2": int(master["has_s2"].sum()),
            "has_l89": int(master["has_l89"].sum()),
            "has_emit": int(master["has_emit"].sum()),
            "has_s5p": int(master["has_s5p"].sum()),
        },
    )


if __name__ == "__main__":
    main()
