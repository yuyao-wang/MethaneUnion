#!/usr/bin/env python3
"""Patch an already-cropped manifest from S5P tif output to S5P npz output.

This does not recrop data. It reads existing S5P tif files referenced by the
manifest, writes one npz per available S5P sample, and updates:

    s5p_0_path   -> path/to/s5p_0.npz
    s5p_90_path  -> empty
    s5p_360_path -> empty
    s5p_plume_path -> empty

The npz layout matches the current old training data:

    npz["ch4"].shape == (3, H, W)
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tifffile


S5P_COLS = ["s5p_0_path", "s5p_90_path", "s5p_360_path", "s5p_plume_path"]


def is_valid_path(value) -> bool:
    if value is None:
        return False
    if pd.isna(value):
        return False
    s = str(value).strip()
    return bool(s) and s.lower() not in {"nan", "none", "null", "<na>"}


def as_channel_first(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"expected 2D or 3D tif array, got shape={arr.shape}")

    if arr.shape[0] == 3:
        return arr
    if arr.shape[-1] == 3:
        return np.moveaxis(arr, -1, 0)
    if arr.shape[0] == 1:
        return arr
    if arr.shape[-1] == 1:
        return np.moveaxis(arr, -1, 0)
    raise ValueError(f"cannot infer channel axis for shape={arr.shape}")


def load_s5p_stack(row: pd.Series) -> Optional[np.ndarray]:
    p0 = row.get("s5p_0_path")
    if not is_valid_path(p0):
        return None
    p0 = Path(str(p0))
    if p0.suffix.lower() == ".npz":
        return None
    if p0.suffix.lower() not in {".tif", ".tiff"}:
        return None
    if not p0.exists():
        raise FileNotFoundError(str(p0))

    arr0 = as_channel_first(tifffile.imread(str(p0)))
    if arr0.shape[0] == 3:
        return arr0.astype(np.float32)

    parts = [arr0.squeeze(0)]
    for col in ["s5p_90_path", "s5p_360_path"]:
        p = row.get(col)
        if not is_valid_path(p):
            raise ValueError(f"{p0} is single-channel but {col} is empty")
        p = Path(str(p))
        if not p.exists():
            raise FileNotFoundError(str(p))
        arr = as_channel_first(tifffile.imread(str(p)))
        if arr.shape[0] != 1:
            raise ValueError(f"expected single-channel {col}, got shape={arr.shape}")
        parts.append(arr.squeeze(0))
    return np.stack(parts, axis=0).astype(np.float32)


def out_path_for(row: pd.Series, suffix: str) -> Path:
    p0 = Path(str(row["s5p_0_path"]))
    return p0.with_suffix(suffix)


def patch_manifest(args) -> None:
    df = pd.read_csv(args.manifest, low_memory=False)
    for col in S5P_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    converted = 0
    skipped_empty = 0
    skipped_npz = 0
    failed = 0

    for idx, row in df.iterrows():
        p0 = row.get("s5p_0_path")
        if not is_valid_path(p0):
            skipped_empty += 1
            continue
        if str(p0).lower().endswith(".npz"):
            skipped_npz += 1
            continue

        try:
            stack = load_s5p_stack(row)
            if stack is None:
                skipped_empty += 1
                continue
            if stack.shape[0] != 3:
                raise ValueError(f"expected 3 S5P channels, got shape={stack.shape}")

            out_npz = out_path_for(row, ".npz")
            if (not out_npz.exists()) or args.overwrite_npz:
                meta = {
                    "source_s5p_0_path": str(row.get("s5p_0_path", "")),
                    "source_s5p_90_path": str(row.get("s5p_90_path", "")),
                    "source_s5p_360_path": str(row.get("s5p_360_path", "")),
                    "source_s5p_plume_path": str(row.get("s5p_plume_path", "")),
                    "plume_id": str(row.get("plume_id", "")),
                    "label": int(row.get("label", -1)) if not pd.isna(row.get("label", pd.NA)) else -1,
                    "format": "s5p_triplet_npz",
                    "channels": ["t0", "t_minus90", "t_minus360"],
                }
                np.savez_compressed(str(out_npz), ch4=stack, meta=np.array(meta, dtype=object))

            old_paths = [row.get(c) for c in S5P_COLS]
            df.at[idx, "s5p_0_path"] = str(out_npz)
            df.at[idx, "s5p_90_path"] = pd.NA
            df.at[idx, "s5p_360_path"] = pd.NA
            df.at[idx, "s5p_plume_path"] = pd.NA
            converted += 1

            if args.delete_tif:
                for p in old_paths:
                    if is_valid_path(p):
                        p = Path(str(p))
                        if p.suffix.lower() in {".tif", ".tiff"} and p.exists():
                            p.unlink()
        except Exception as exc:
            failed += 1
            if args.strict:
                raise
            print(f"[warn] row={idx} s5p_0_path={p0} failed: {type(exc).__name__}: {exc}", flush=True)

        if converted > 0 and converted % args.progress_every == 0:
            print(f"[progress] converted={converted} failed={failed}", flush=True)

    out_manifest = args.out_manifest
    if args.inplace:
        backup = args.manifest.with_suffix(args.manifest.suffix + ".before_s5p_npz_patch")
        if not backup.exists():
            args.manifest.replace(backup)
        out_manifest = args.manifest
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_manifest, index=False)

    print(f"saved_manifest: {out_manifest}")
    print(f"converted_tif_to_npz: {converted}")
    print(f"skipped_empty_s5p: {skipped_empty}")
    print(f"skipped_already_npz: {skipped_npz}")
    print(f"failed: {failed}")


def parse_args():
    p = argparse.ArgumentParser(description="Convert already-cropped S5P tif outputs to one npz per sample.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out_manifest", type=Path, default=None)
    p.add_argument("--inplace", action="store_true", help="overwrite manifest after writing a .before_s5p_npz_patch backup")
    p.add_argument("--overwrite_npz", action="store_true", help="rewrite npz even if it already exists")
    p.add_argument("--delete_tif", action="store_true", help="delete old S5P tif files after successful conversion")
    p.add_argument("--strict", action="store_true", help="stop at first conversion failure")
    p.add_argument("--progress_every", type=int, default=1000)
    args = p.parse_args()

    if args.out_manifest is None:
        args.out_manifest = args.manifest.with_name(args.manifest.stem + "_s5p_npz.csv")
    if args.inplace and args.out_manifest != args.manifest.with_name(args.manifest.stem + "_s5p_npz.csv"):
        raise SystemExit("--inplace and --out_manifest should not be used together")
    return args


if __name__ == "__main__":
    patch_manifest(parse_args())
