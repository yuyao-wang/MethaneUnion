#!/usr/bin/env python3
"""Standardize six-time Sentinel-2 TIFF stacks to 512x512 GDAL multiband files.

This is the six-time version of the old notebook cell that created
``s2_*_std_512.tif`` files.  It keeps the old behavior where S2 chips are
treated as ML arrays: arrays are normalized to B,H,W float32, center
cropped or zero padded to 512x512, and written with a dummy transform.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import tifffile
from rasterio.transform import from_origin
from rasterio.windows import Window
from tqdm import tqdm


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "Upgrade_data_pipeline" / "csv").exists():
            return parent
    raise RuntimeError(f"Could not find repo root from {here}")


REPO_ROOT = find_repo_root()
DEFAULT_INPUT_CSV = REPO_ROOT / "Upgrade_data_pipeline" / "csv" / "s2_6time_complete_paths.csv"
DEFAULT_OUT_CSV = REPO_ROOT / "Upgrade_data_pipeline" / "csv" / "s2_6time_complete_paths_std512.csv"
DEFAULT_CLEAN_CSV = REPO_ROOT / "Upgrade_data_pipeline" / "csv" / "s2_6time_complete_paths_std512_complete.csv"
DEFAULT_OUT_ROOT = Path("/mnt/engg-niulab/yuyao/preprocessed_512/S2")

WINDOW_SIZE = 512
EXPECTED_S2_BANDS = {12, 13}


@dataclass(frozen=True)
class Timepoint:
    name: str
    raw_col: str
    path_col: str
    legacy_col: str
    filename: str
    force_tifffile: bool = False


TIMEPOINTS = [
    Timepoint("t0", "t0_raw_path", "t0_512_path", "s2_0_std_512", "s2_0_std_512.tif", True),
    Timepoint("prev1", "prev1_raw_path", "prev1_512_path", "s2_-7_std_512", "s2_-7_std_512.tif"),
    Timepoint("prev2", "prev2_raw_path", "prev2_512_path", "s2_prev2_std_512", "s2_prev2_std_512.tif"),
    Timepoint("prev3", "prev3_raw_path", "prev3_512_path", "s2_prev3_std_512", "s2_prev3_std_512.tif"),
    Timepoint("seasonal", "seasonal_raw_path", "seasonal_512_path", "s2_-90_std_512", "s2_-90_std_512.tif"),
    Timepoint("year", "year_raw_path", "year_512_path", "s2_-360_std_512", "s2_-360_std_512.tif"),
]


def debug(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}][pid:{os.getpid()}][tid:{threading.get_ident()}] {msg}", flush=True)


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null", "<na>"}


def existing_file(value: Any) -> bool:
    if not has_value(value):
        return False
    path = Path(str(value).strip())
    return path.exists() and path.stat().st_size > 0


@contextlib.contextmanager
def suppress_stderr():
    # Do not redirect sys.stderr in worker threads. redirect_stderr is
    # process-global, so one worker can close stderr while another is using it.
    yield


def to_bhw(arr: np.ndarray) -> np.ndarray:
    """Accept H,W / B,H,W / H,W,B and return B,H,W."""
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected ndim={arr.ndim}, shape={arr.shape}")
    if arr.shape[0] in (1, 3, 4, 12, 13):
        return arr
    if arr.shape[-1] in (1, 3, 4, 12, 13):
        return np.transpose(arr, (2, 0, 1))
    return arr


def center_crop_or_pad_bhw(bhw: np.ndarray, out_size: int = WINDOW_SIZE) -> tuple[np.ndarray, str]:
    b, h, w = bhw.shape
    if h == out_size and w == out_size:
        return bhw, "noop"

    if h < out_size or w < out_size:
        out = np.zeros((b, out_size, out_size), dtype=bhw.dtype)
        y0 = max(0, (out_size - h) // 2)
        x0 = max(0, (out_size - w) // 2)
        out[:, y0 : y0 + h, x0 : x0 + w] = bhw
        return out, f"pad_center {h}x{w} -> {out_size}x{out_size}"

    y0 = (h - out_size) // 2
    x0 = (w - out_size) // 2
    return bhw[:, y0 : y0 + out_size, x0 : x0 + out_size], f"crop_center {h}x{w} -> {out_size}x{out_size}"


def center_window(ds, size: int = WINDOW_SIZE):
    if ds.width < size or ds.height < size:
        return None
    col0 = (ds.width - size) // 2
    row0 = (ds.height - size) // 2
    return Window(col0, row0, size, size)


def write_gdal_multiband_tif(out_path: Path, bhw: np.ndarray) -> None:
    bhw = np.asarray(bhw)
    if bhw.ndim != 3:
        raise ValueError(f"write expects BHW, got shape={bhw.shape}")

    b, h, w = bhw.shape
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": b,
        "dtype": str(bhw.dtype),
        "transform": from_origin(0, 0, 1, 1),
        "compress": "deflate",
        "predictor": 2 if np.issubdtype(bhw.dtype, np.floating) else 1,
        "tiled": True,
        "blockxsize": 256 if w >= 256 else w,
        "blockysize": 256 if h >= 256 else h,
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(bhw)


def read_with_tifffile(in_path: str) -> np.ndarray:
    return to_bhw(tifffile.imread(in_path)).astype(np.float32)


def read_with_rasterio_center(in_path: str) -> np.ndarray:
    with suppress_stderr():
        with rasterio.open(in_path) as ds:
            win = center_window(ds, WINDOW_SIZE)
            arr = ds.read() if win is None else ds.read(window=win)
    return to_bhw(arr).astype(np.float32)


def std_to_512(in_path: str, out_path: Path, tag: str, force_tifffile: bool, overwrite: bool) -> tuple[bool, str]:
    if not existing_file(in_path):
        return False, f"{tag}_missing:{in_path}"
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        return True, "skipped_exists"

    errors: list[str] = []
    bhw: np.ndarray | None = None
    read_note = ""

    if force_tifffile:
        try:
            bhw = read_with_tifffile(in_path)
            read_note = "tifffile"
        except Exception as exc:
            return False, f"{tag}_tifffile_err:{type(exc).__name__}:{exc}"
    else:
        try:
            bhw = read_with_rasterio_center(in_path)
            read_note = "rasterio"
            # Current S2 TIFFs include tifffile page stacks. Rasterio can open
            # them as a single-band image without raising, so validate the band
            # count before accepting the rasterio read.
            if bhw.shape[0] == 1:
                try:
                    tf_bhw = read_with_tifffile(in_path)
                    if tf_bhw.shape[0] in EXPECTED_S2_BANDS:
                        bhw = tf_bhw
                        read_note = "tifffile_after_rasterio_single_band"
                except Exception as exc:
                    errors.append(f"{tag}_tifffile_probe_err:{type(exc).__name__}:{exc}")
        except Exception as exc:
            errors.append(f"{tag}_rasterio_err:{type(exc).__name__}:{exc}")
            try:
                bhw = read_with_tifffile(in_path)
                read_note = "tifffile_fallback"
            except Exception as exc2:
                errors.append(f"{tag}_tifffile_err:{type(exc2).__name__}:{exc2}")
                return False, "; ".join(errors)

    if bhw is None:
        return False, f"{tag}_read_empty"
    if bhw.shape[0] not in EXPECTED_S2_BANDS:
        errors.append(f"{tag}_unexpected_band_count:{bhw.shape}")

    bhw2, crop_note = center_crop_or_pad_bhw(bhw.astype(np.float32, copy=False), WINDOW_SIZE)
    write_gdal_multiband_tif(out_path, bhw2)

    notes = [read_note]
    if crop_note != "noop":
        notes.append(crop_note)
    notes.extend(errors)
    return True, "; ".join(note for note in notes if note)


def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    for tp in TIMEPOINTS:
        for col in (tp.path_col, tp.legacy_col, f"has_{tp.name}_512"):
            if col not in df.columns:
                df[col] = ""
    for col in ("std_ok", "std_reason", "bug", "has_all6_512"):
        if col not in df.columns:
            df[col] = ""
    return df


def select_input_path(row: dict[str, Any], tp: Timepoint, prefer_existing_512: bool) -> str:
    input_col = f"{tp.name}_input_path"
    if existing_file(row.get(input_col, "")):
        return str(row.get(input_col, "")).strip()

    candidates = [tp.raw_col, tp.path_col, tp.legacy_col]
    if prefer_existing_512:
        candidates = [tp.path_col, tp.legacy_col, tp.raw_col]
    for col in candidates:
        value = row.get(col, "")
        if existing_file(value):
            return str(value).strip()
    return ""


def expected_out_path(out_root: Path, plume_id: str, tp: Timepoint) -> Path:
    return out_root / plume_id / tp.filename


def row_outputs_complete(out_root: Path, plume_id: str) -> bool:
    return all(existing_file(expected_out_path(out_root, plume_id, tp)) for tp in TIMEPOINTS)


def process_one_row(
    row: dict[str, Any],
    out_root: Path,
    overwrite: bool,
    prefer_existing_512: bool,
    row_level_no_overwrite: bool,
) -> dict[str, Any]:
    plume_id = str(row.get("plume_id", "")).strip()
    result: dict[str, Any] = {"plume_id": plume_id}
    if not plume_id:
        result.update({"std_ok": 0, "std_reason": "missing_plume_id", "bug": "missing_plume_id", "has_all6_512": 0})
        return result

    row_overwrite = overwrite
    if row_level_no_overwrite and not overwrite:
        row_overwrite = not row_outputs_complete(out_root, plume_id)

    bugs: list[str] = []
    all_ok = True
    for tp in TIMEPOINTS:
        in_path = select_input_path(row, tp, prefer_existing_512)
        out_path = expected_out_path(out_root, plume_id, tp)
        if not in_path:
            ok, note = False, f"{tp.name}_missing_input"
        else:
            ok, note = std_to_512(in_path, out_path, tp.name, tp.force_tifffile, row_overwrite)

        if ok and out_path.exists() and out_path.stat().st_size > 0:
            result[tp.path_col] = str(out_path)
            result[tp.legacy_col] = str(out_path)
            result[f"has_{tp.name}_512"] = 1
        else:
            all_ok = False
            result[tp.path_col] = ""
            result[tp.legacy_col] = ""
            result[f"has_{tp.name}_512"] = 0

        if note and note not in {"skipped_exists", "tifffile", "rasterio"}:
            bugs.append(f"{tp.name}:{note}")

    reason = "; ".join(bugs)
    result["std_ok"] = 1 if all_ok else 0
    result["has_all6_512"] = 1 if all_ok else 0
    result["std_reason"] = reason
    result["bug"] = reason
    return result


def flush_updates(df: pd.DataFrame, updates: list[dict[str, Any]], out_csv: Path) -> None:
    if not updates:
        return
    by_id = {str(item["plume_id"]): item for item in updates if has_value(item.get("plume_id"))}
    for idx, row in df.iterrows():
        upd = by_id.get(str(row.get("plume_id", "")).strip())
        if not upd:
            continue
        for key, value in upd.items():
            if key not in df.columns:
                df[key] = ""
            df.at[idx, key] = value
    tmp = out_csv.with_suffix(out_csv.suffix + ".part")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_csv)


def write_complete_csv(df: pd.DataFrame, clean_csv: Path) -> int:
    path_cols = [tp.path_col for tp in TIMEPOINTS]
    keep = df.copy()
    for col in path_cols:
        keep = keep[keep[col].map(has_value)]
    clean_csv.parent.mkdir(parents=True, exist_ok=True)
    keep.to_csv(clean_csv, index=False)
    return len(keep)


def run(args: argparse.Namespace) -> int:
    in_csv = Path(args.input_csv)
    out_csv = Path(args.out_csv)
    clean_csv = Path(args.clean_csv)
    out_root = Path(args.out_root)

    df = ensure_output_columns(pd.read_csv(in_csv))
    rows = df.to_dict("records")
    if args.limit:
        rows = rows[: args.limit]

    debug(f"load S2: input={in_csv} rows={len(rows)} out_root={out_root}")
    debug(f"write tables: out_csv={out_csv} clean_csv={clean_csv}")

    updates: list[dict[str, Any]] = []
    recent = deque(maxlen=200)
    fail_reasons = Counter()
    ok_count = fail_count = 0
    last_flush = time.time()
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                process_one_row,
                row,
                out_root,
                args.overwrite,
                args.prefer_existing_512,
                args.row_level_no_overwrite,
            )
            for row in rows
        ]
        for done, fut in enumerate(tqdm(as_completed(futs), total=len(futs), desc="S2 std512", mininterval=1.0), start=1):
            try:
                upd = fut.result()
            except Exception as exc:
                upd = {"plume_id": "", "std_ok": 0, "std_reason": f"worker_exception:{type(exc).__name__}:{exc}", "bug": str(exc)}
            updates.append(upd)

            is_ok = int(upd.get("std_ok", 0)) == 1
            if is_ok:
                ok_count += 1
                recent.append(1)
            else:
                fail_count += 1
                recent.append(0)
                reason = str(upd.get("std_reason", ""))[:160]
                fail_reasons[reason] += 1

            now = time.time()
            if done % args.progress_every == 0 or done == len(futs):
                recent_ok = sum(recent) / max(1, len(recent))
                elapsed_min = (now - start) / 60
                debug(
                    f"[STAT] done={done}/{len(futs)} ok={ok_count} fail={fail_count} "
                    f"recent200_ok={recent_ok:.2%} elapsed={elapsed_min:.1f} min"
                )
                debug(f"[STAT] top_fail_reasons={fail_reasons.most_common(5)}")
            if args.flush_every_sec > 0 and now - last_flush >= args.flush_every_sec:
                flush_updates(df, updates, out_csv)
                last_flush = now
                debug(f"flushed: {out_csv}")

    flush_updates(df, updates, out_csv)
    final_df = ensure_output_columns(pd.read_csv(out_csv))
    complete_count = write_complete_csv(final_df, clean_csv)
    debug(f"DONE ok={ok_count} fail={fail_count} complete_all6={complete_count}/{len(final_df)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--clean-csv", default=str(DEFAULT_CLEAN_CSV))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--flush-every-sec", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true", help="rewrite existing output TIFFs")
    parser.add_argument(
        "--row-level-no-overwrite",
        action="store_true",
        help=(
            "Skip a row only when all six target TIFFs already exist. If a row is partial, "
            "rewrite/process all six timepoints for that row without requiring global --overwrite."
        ),
    )
    parser.add_argument(
        "--prefer-existing-512",
        action="store_true",
        help="normalize existing *_512_path before raw_path; default is raw_path first",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
