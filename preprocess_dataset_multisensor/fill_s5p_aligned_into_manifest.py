import argparse
import fcntl
import hashlib
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
from netCDF4 import Dataset

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None


OUT_SIZE = 224
S5P_PATCH = 3
S5P_HALF = S5P_PATCH // 2
S5P_MISSING_THRESH = 0.50

GSD = {
    "s2": 10.0,
    "l89": 30.0,
    "emit": 60.0,
    "s5p": 3500.0,
}

CH4_CANDIDATES = [
    "methane_mixing_ratio_bias_corrected",
    "methane_mixing_ratio",
    "xch4",
]


def is_valid(v) -> bool:
    if pd.isna(v):
        return False
    s = str(v).strip()
    return s != "" and s.lower() != "nan"


def parse_centers(s) -> List[Tuple[int, int]]:
    s = str(s) if s is not None else ""
    s = s.strip()
    if (not s) or (s.lower() == "nan"):
        return []
    out: List[Tuple[int, int]] = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        if "," not in part:
            continue
        vals = [x.strip() for x in part.split(",") if x.strip() != ""]
        if len(vals) < 2:
            continue
        try:
            cy = int(float(vals[0]))
            cx = int(float(vals[1]))
        except Exception:
            continue
        out.append((cy, cx))
    return out


def missing_ratio(patch2d: np.ndarray) -> float:
    return 1.0 - (np.isfinite(patch2d).sum() / patch2d.size)


def nan_out() -> np.ndarray:
    return np.full((OUT_SIZE, OUT_SIZE), np.nan, dtype=np.float32)


@contextmanager
def silence_fd2():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old, 2)
        os.close(devnull)
        os.close(old)


def get_2d(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a)
    if x.ndim == 3:
        return x[0]
    if x.ndim == 2:
        return x
    raise ValueError(f"Unexpected dims: {x.shape}")


def to_nan_invalid(arr, attrs=None):
    a = np.array(arr, dtype=np.float32, copy=False)
    attrs = attrs or {}
    fv = attrs.get("_FillValue", None)
    mv = attrs.get("missing_value", None)
    if fv is not None:
        a = np.where(a == np.float32(fv), np.nan, a)
    if mv is not None:
        a = np.where(a == np.float32(mv), np.nan, a)
    a = np.where(np.abs(a) > 1e20, np.nan, a)
    return get_2d(a)


def pick_ch4_var(prod) -> Optional[str]:
    for k in CH4_CANDIDATES:
        if k in prod.variables:
            return k
    return None


def read_ch4(path_nc: str, ch4_hint: Optional[str] = None) -> Tuple[np.ndarray, str]:
    p = Path(str(path_nc))
    if p.suffix.lower() != ".nc":
        raise ValueError(f"S5P path must be .nc, got: {p}")
    if not p.exists():
        raise FileNotFoundError(f"S5P file not found: {p}")
    with silence_fd2():
        ds = Dataset(str(p), "r")
    try:
        prod = ds.groups["PRODUCT"]
        ch4name = ch4_hint if (ch4_hint and ch4_hint in prod.variables) else pick_ch4_var(prod)
        if ch4name is None:
            raise KeyError(f"No CH4 variable found in {p}. candidates={CH4_CANDIDATES}")
        v = prod.variables[ch4name]
        arr = to_nan_invalid(v[:], getattr(v, "__dict__", {})).astype(np.float32)
        return arr, ch4name
    finally:
        ds.close()


def resize_nan_aware(src2d: np.ndarray) -> np.ndarray:
    src = src2d.astype(np.float32, copy=False)
    fin = np.isfinite(src)
    v = np.where(fin, src, 0.0).astype(np.float32)
    w = fin.astype(np.float32)
    if torch is not None and F is not None:
        vt = torch.from_numpy(v)[None, None, :, :].float()
        wt = torch.from_numpy(w)[None, None, :, :].float()
        vr = F.interpolate(vt, size=(OUT_SIZE, OUT_SIZE), mode="bilinear", align_corners=False).squeeze().cpu().numpy()
        wr = F.interpolate(wt, size=(OUT_SIZE, OUT_SIZE), mode="bilinear", align_corners=False).squeeze().cpu().numpy()
    else:
        ys = np.linspace(0, v.shape[0] - 1, OUT_SIZE)
        xs = np.linspace(0, v.shape[1] - 1, OUT_SIZE)
        y0 = np.floor(ys).astype(np.int32)
        x0 = np.floor(xs).astype(np.int32)
        y1 = np.clip(y0 + 1, 0, v.shape[0] - 1)
        x1 = np.clip(x0 + 1, 0, v.shape[1] - 1)
        wy = (ys - y0)[:, None]
        wx = (xs - x0)[None, :]
        va = v[y0[:, None], x0[None, :]]
        vb = v[y0[:, None], x1[None, :]]
        vc = v[y1[:, None], x0[None, :]]
        vd = v[y1[:, None], x1[None, :]]
        wa = w[y0[:, None], x0[None, :]]
        wb = w[y0[:, None], x1[None, :]]
        wc = w[y1[:, None], x0[None, :]]
        wd = w[y1[:, None], x1[None, :]]
        vr = va * (1 - wx) * (1 - wy) + vb * wx * (1 - wy) + vc * (1 - wx) * wy + vd * wx * wy
        wr = wa * (1 - wx) * (1 - wy) + wb * wx * (1 - wy) + wc * (1 - wx) * wy + wd * wx * wy
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(wr > 1e-6, vr / wr, np.nan)
    return out.astype(np.float32)


def crop_3x3(arr2d: np.ndarray, cy: int, cx: int) -> Optional[np.ndarray]:
    y0, y1 = cy - S5P_HALF, cy + S5P_HALF + 1
    x0, x1 = cx - S5P_HALF, cx + S5P_HALF + 1
    if y0 < 0 or x0 < 0 or y1 > arr2d.shape[0] or x1 > arr2d.shape[1]:
        return None
    c = arr2d[y0:y1, x0:x1]
    miss = 1.0 - np.isfinite(c).sum() / c.size
    if miss > S5P_MISSING_THRESH:
        return None
    return c


def crop_3x3_t0_with_reason(arr2d: np.ndarray, cy: int, cx: int) -> Tuple[Optional[np.ndarray], str]:
    y0, y1 = cy - S5P_HALF, cy + S5P_HALF + 1
    x0, x1 = cx - S5P_HALF, cx + S5P_HALF + 1
    if y0 < 0 or x0 < 0 or y1 > arr2d.shape[0] or x1 > arr2d.shape[1]:
        return None, "out_of_bounds"
    c = arr2d[y0:y1, x0:x1]
    miss = missing_ratio(c)
    if miss > S5P_MISSING_THRESH:
        return None, f"too_missing({miss:.2f})"
    return c, "ok"


def crop_3x3_any_with_reason(arr2d: Optional[np.ndarray], cy: int, cx: int) -> Tuple[Optional[np.ndarray], str]:
    if arr2d is None:
        return None, "missing_band"
    y0, y1 = cy - S5P_HALF, cy + S5P_HALF + 1
    x0, x1 = cx - S5P_HALF, cx + S5P_HALF + 1
    if y0 < 0 or x0 < 0 or y1 > arr2d.shape[0] or x1 > arr2d.shape[1]:
        return None, "out_of_bounds"
    return arr2d[y0:y1, x0:x1], "ok"


def get_s5p_center(anchor_sensor: str, dx_anchor: int, dy_anchor: int, nearest_ix: int, nearest_iy: int) -> Tuple[int, int]:
    gsd_anchor = GSD.get(str(anchor_sensor), None)
    if gsd_anchor is None:
        gsd_anchor = GSD["s2"]
    dx_m = dx_anchor * gsd_anchor
    dy_m = dy_anchor * gsd_anchor
    dx_s5p = int(round(dx_m / GSD["s5p"]))
    dy_s5p = int(round(dy_m / GSD["s5p"]))
    return nearest_ix + dx_s5p, nearest_iy + dy_s5p


def _cache_file_name(src_path: str) -> str:
    h = hashlib.sha1(src_path.encode("utf-8")).hexdigest()[:16]
    return f"{h}_{Path(src_path).name}"


@contextmanager
def file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def cache_usage_bytes(cache_dir: Path) -> int:
    total = 0
    for p in cache_dir.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def wait_and_copy_to_cache(
    src_path: str,
    cache_dir: Path,
    cache_max_bytes: int,
    poll_seconds: int,
    lock_path: Path,
) -> Path:
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"S5P source file not found: {src}")
    need = src.stat().st_size
    dst = cache_dir / _cache_file_name(str(src))
    while True:
        with file_lock(lock_path):
            if dst.exists() and dst.stat().st_size == need:
                return dst
            used = cache_usage_bytes(cache_dir)
            can_copy = (used + need) <= cache_max_bytes
            if can_copy:
                tmp = dst.with_suffix(dst.suffix + f".{os.getpid()}.part")
                shutil.copy2(src, tmp)
                tmp.replace(dst)
                return dst
        time.sleep(max(poll_seconds, 1))


def process_plume_task(task: Dict) -> Tuple[Dict[str, int], List[Tuple[int, str, str]], List[str]]:
    pid = task["pid"]
    indices = task["indices"]
    rows = task["rows"]
    src = task["src"]
    out_root = Path(task["out_root"])
    force_overwrite = bool(task["force_overwrite_existing_s5p"])
    use_local_cache = bool(task.get("use_local_cache", False))
    cache_dir = Path(task["cache_dir"]) if task.get("cache_dir") else None
    cache_max_bytes = int(task.get("cache_max_bytes", 0))
    cache_poll_seconds = int(task.get("cache_poll_seconds", 10))
    delete_cache_after_use = bool(task.get("delete_cache_after_use", True))
    lock_path = cache_dir / ".cache.lock" if cache_dir is not None else None

    local = {
        "processed": len(indices),
        "updated": 0,
        "skipped_crop_fail": 0,
        "skipped_has_existing": 0,
        "fallback_to_nearest": 0,
        "mapped_crop_fail": 0,
        "nearest_crop_fail": 0,
    }
    local_updates: List[Tuple[int, str, str]] = []
    local_sample_crop_fail: List[str] = []

    local_cached_files: List[Path] = []
    p0 = str(src["S5p_path"])
    p90 = str(src["s5p_minus90_path"])
    p360 = str(src["s5p_minus360_path"])
    if use_local_cache:
        assert cache_dir is not None
        cache_dir.mkdir(parents=True, exist_ok=True)
        assert lock_path is not None
        p0c = wait_and_copy_to_cache(p0, cache_dir, cache_max_bytes, cache_poll_seconds, lock_path)
        p90c = wait_and_copy_to_cache(p90, cache_dir, cache_max_bytes, cache_poll_seconds, lock_path)
        p360c = wait_and_copy_to_cache(p360, cache_dir, cache_max_bytes, cache_poll_seconds, lock_path)
        local_cached_files = [p0c, p90c, p360c]
        p0, p90, p360 = str(p0c), str(p90c), str(p360c)

    try:
        t0, ch4 = read_ch4(p0, None)
        t90, _ = read_ch4(p90, ch4)
        t360, _ = read_ch4(p360, ch4)
    finally:
        if use_local_cache and delete_cache_after_use:
            for p in local_cached_files:
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass

    if not is_valid(src.get("nearest_ix")) or not is_valid(src.get("nearest_iy")):
        local["skipped_crop_fail"] += len(indices)
        local["nearest_crop_fail"] += len(indices)
        local_sample_crop_fail.append(f"{pid}:invalid_nearest")
        return local, local_updates, local_sample_crop_fail
    nearest_ix = int(float(src["nearest_ix"]))
    nearest_iy = int(float(src["nearest_iy"]))
    pcs = parse_centers(src.get("pos_centers", ""))

    for r in rows:
        i = int(r["i"])
        sample_id = int(r["id"])
        if (not force_overwrite) and is_valid(r.get("s5p_0_path")):
            local["skipped_has_existing"] += 1
            continue

        cx, cy = get_s5p_center(
            str(r.get("anchor_sensor")),
            int(r.get("dx_anchor_px", 0)),
            int(r.get("dy_anchor_px", 0)),
            nearest_ix,
            nearest_iy,
        )

        c0, r0m = crop_3x3_t0_with_reason(t0, cy, cx)
        c90, r90m = crop_3x3_any_with_reason(t90, cy, cx)
        c360, r360m = crop_3x3_any_with_reason(t360, cy, cx)
        if c0 is None:
            local["mapped_crop_fail"] += 1
            c0, r0n = crop_3x3_t0_with_reason(t0, nearest_iy, nearest_ix)
            c90, r90n = crop_3x3_any_with_reason(t90, nearest_iy, nearest_ix)
            c360, r360n = crop_3x3_any_with_reason(t360, nearest_iy, nearest_ix)
            local["fallback_to_nearest"] += 1
        else:
            r0n, r90n, r360n = "na", "na", "na"

        if c0 is None and int(r.get("label", 0)) == 1:
            for pcy, pcx in pcs:
                tc0, tr0 = crop_3x3_t0_with_reason(t0, pcy, pcx)
                if tc0 is None:
                    continue
                c0 = tc0
                c90, r90n = crop_3x3_any_with_reason(t90, pcy, pcx)
                c360, r360n = crop_3x3_any_with_reason(t360, pcy, pcx)
                r0n = f"pos_centers:{tr0}"
                break

        if c0 is None:
            local["skipped_crop_fail"] += 1
            local["nearest_crop_fail"] += 1
            if len(local_sample_crop_fail) < 3:
                local_sample_crop_fail.append(
                    f"{pid}:mapped=({r0m},{r90m},{r360m}) nearest=({r0n},{r90n},{r360n})"
                )
            continue

        r0 = resize_nan_aware(c0)
        r90 = resize_nan_aware(c90) if c90 is not None else nan_out()
        r360 = resize_nan_aware(c360) if c360 is not None else nan_out()
        stack = np.stack([r0, r90, r360], axis=0).astype(np.float32)

        group = out_root / f"group_{sample_id:08d}"
        group.mkdir(parents=True, exist_ok=True)
        p0 = group / "s5p_0.tif"
        pm = group / "s5p_plume.tif"
        tifffile.imwrite(str(p0), stack)
        msk = np.zeros((OUT_SIZE, OUT_SIZE), dtype=np.float32)
        msk[OUT_SIZE // 2, OUT_SIZE // 2] = 1.0
        tifffile.imwrite(str(pm), msk)

        local_updates.append((i, str(p0), str(pm)))
        local["updated"] += 1

    return local, local_updates, local_sample_crop_fail


def main():
    ap = argparse.ArgumentParser(description="Fill aligned S5P crops into an existing multisensor manifest.")
    ap.add_argument("--manifest_csv", type=Path, required=True)
    ap.add_argument("--master_csv", type=Path, required=True)
    ap.add_argument("--out_root", type=Path, required=True, help="Same root used by manifest group_x folders")
    ap.add_argument("--out_csv", type=Path, default=None, help="Default: overwrite manifest_csv")
    ap.add_argument("--backup_csv", type=Path, default=None, help="Optional backup path before overwrite")
    ap.add_argument("--force_overwrite_existing_s5p", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="For smoke test")
    ap.add_argument("--debug", action="store_true", help="Print debug logs")
    ap.add_argument("--debug_every", type=int, default=2000, help="Progress print frequency")
    ap.add_argument("--debug_limit", type=int, default=20, help="Always print debug for first N processed rows")
    ap.add_argument("--workers", type=int, default=1, help="Thread workers (recommend 4-8 for network storage)")
    ap.add_argument("--executor", choices=["process", "thread"], default="process", help="Parallel backend")
    ap.add_argument("--cache_dir", type=Path, default=Path("/home/yuyao/s5p_nc_cache"))
    ap.add_argument("--cache_max_gb", type=float, default=350.0)
    ap.add_argument("--cache_poll_seconds", type=int, default=20)
    ap.add_argument("--disable_local_cache", action="store_true", help="Read S5P nc directly from source paths")
    ap.add_argument("--keep_cache_files", action="store_true", help="Do not delete cached nc after plume is processed")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest_csv, low_memory=False)
    required_manifest = ["id", "plume_id", "anchor_sensor", "dx_anchor_px", "dy_anchor_px", "s5p_0_path"]
    for c in required_manifest:
        if c not in manifest.columns:
            raise RuntimeError(f"Missing manifest col: {c}")

    master_cols = [
        "plume_id",
        "has_s5p",
        "S5p_path",
        "s5p_minus90_path",
        "s5p_minus360_path",
        "nearest_ix",
        "nearest_iy",
        "pos_centers",
    ]
    master = pd.read_csv(args.master_csv, usecols=master_cols, low_memory=False)
    master["has_s5p"] = master["has_s5p"].fillna(False).astype(bool)
    master = master[
        master["has_s5p"]
        & master["S5p_path"].notna()
        & master["s5p_minus90_path"].notna()
        & master["s5p_minus360_path"].notna()
    ].copy()
    if args.debug:
        suf = master["S5p_path"].astype(str).str.extract(r"(\.[^./\\]+)$", expand=False).fillna("no_ext")
        print("[debug] S5p_path suffix top:", suf.value_counts().head(5).to_dict())
    master = master.drop_duplicates(subset=["plume_id"], keep="first")
    m = {str(r["plume_id"]): r for _, r in master.iterrows()}
    if args.debug:
        print(
            f"[debug] manifest_rows={len(manifest)} "
            f"manifest_unique_plume={manifest['plume_id'].nunique()} "
            f"master_s5p_rows={len(master)} master_s5p_unique_plume={master['plume_id'].nunique()}"
        )
        if not args.disable_local_cache:
            print(
                f"[debug] local_cache={args.cache_dir} max_gb={args.cache_max_gb} "
                f"poll_seconds={args.cache_poll_seconds} keep_cache_files={args.keep_cache_files}"
            )

    out_csv = args.out_csv if args.out_csv is not None else args.manifest_csv
    if out_csv == args.manifest_csv and args.backup_csv is not None:
        manifest.to_csv(args.backup_csv, index=False)

    for c in ["s5p_90_path", "s5p_360_path", "s5p_plume_path"]:
        if c not in manifest.columns:
            manifest[c] = pd.NA
    # Ensure path columns accept strings (avoid FutureWarning on float dtype columns).
    for c in ["s5p_0_path", "s5p_90_path", "s5p_360_path", "s5p_plume_path"]:
        manifest[c] = manifest[c].astype("object")

    updated = 0
    skipped_no_match = 0
    skipped_crop_fail = 0
    skipped_has_existing = 0
    fallback_to_nearest = 0
    processed = 0
    mapped_crop_fail = 0
    nearest_crop_fail = 0

    sample_no_match = []
    sample_crop_fail = []

    row_indices = manifest.index.to_list()
    if args.limit > 0:
        row_indices = row_indices[: args.limit]

    plume_to_indices: Dict[str, List[int]] = {}
    for i in row_indices:
        pid = str(manifest.at[i, "plume_id"])
        plume_to_indices.setdefault(pid, []).append(i)

    next_report = max(args.debug_every, 1)
    executor_cls = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor

    tasks = []
    for pid, idxs in plume_to_indices.items():
        src = m.get(pid, None)
        if src is None:
            processed += len(idxs)
            skipped_no_match += len(idxs)
            if len(sample_no_match) < 10:
                sample_no_match.append(pid)
            continue
        rows_payload = []
        for i in idxs:
            rows_payload.append(
                {
                    "i": int(i),
                    "id": int(manifest.at[i, "id"]),
                    "anchor_sensor": manifest.at[i, "anchor_sensor"],
                    "dx_anchor_px": int(manifest.at[i, "dx_anchor_px"]),
                    "dy_anchor_px": int(manifest.at[i, "dy_anchor_px"]),
                    "label": int(manifest.at[i, "label"]),
                    "s5p_0_path": manifest.at[i, "s5p_0_path"],
                }
            )
        tasks.append(
            {
                "pid": pid,
                "indices": [int(x) for x in idxs],
                "rows": rows_payload,
                "src": {
                    "S5p_path": src["S5p_path"],
                    "s5p_minus90_path": src["s5p_minus90_path"],
                    "s5p_minus360_path": src["s5p_minus360_path"],
                    "nearest_ix": src["nearest_ix"],
                    "nearest_iy": src["nearest_iy"],
                    "pos_centers": src.get("pos_centers", ""),
                },
                "out_root": str(args.out_root),
                "force_overwrite_existing_s5p": bool(args.force_overwrite_existing_s5p),
                "use_local_cache": not bool(args.disable_local_cache),
                "cache_dir": str(args.cache_dir),
                "cache_max_bytes": int(args.cache_max_gb * (1024**3)),
                "cache_poll_seconds": int(args.cache_poll_seconds),
                "delete_cache_after_use": not bool(args.keep_cache_files),
            }
        )

    with executor_cls(max_workers=max(args.workers, 1)) as ex:
        futures = [ex.submit(process_plume_task, t) for t in tasks]
        for fut in as_completed(futures):
            local, local_updates, local_sample_crop_fail = fut.result()
            processed += local["processed"]
            updated += local["updated"]
            skipped_crop_fail += local["skipped_crop_fail"]
            skipped_has_existing += local["skipped_has_existing"]
            fallback_to_nearest += local["fallback_to_nearest"]
            mapped_crop_fail += local["mapped_crop_fail"]
            nearest_crop_fail += local["nearest_crop_fail"]
            for s in local_sample_crop_fail:
                if len(sample_crop_fail) < 10:
                    sample_crop_fail.append(s)
            for i, p0, pm in local_updates:
                manifest.at[i, "s5p_0_path"] = p0
                manifest.at[i, "s5p_90_path"] = pd.NA
                manifest.at[i, "s5p_360_path"] = pd.NA
                manifest.at[i, "s5p_plume_path"] = pm

            if processed >= next_report:
                print(
                    f"[progress] processed={processed}/{len(row_indices)} updated={updated} "
                    f"skip_no_match={skipped_no_match} "
                    f"skip_crop_fail={skipped_crop_fail} skip_has_existing={skipped_has_existing} "
                    f"fallback_to_nearest={fallback_to_nearest} "
                    f"mapped_crop_fail={mapped_crop_fail} nearest_crop_fail={nearest_crop_fail}"
                )
                next_report += max(args.debug_every, 1)

    manifest.to_csv(out_csv, index=False)
    print("saved", out_csv)
    print("processed", processed)
    print("updated", updated)
    print("skipped_no_match", skipped_no_match)
    print("skipped_crop_fail", skipped_crop_fail)
    print("skipped_has_existing", skipped_has_existing)
    print("fallback_to_nearest", fallback_to_nearest)
    print("mapped_crop_fail", mapped_crop_fail)
    print("nearest_crop_fail", nearest_crop_fail)
    if sample_no_match:
        print("sample_no_match_plumes", sample_no_match[:10])
    if sample_crop_fail:
        print("sample_crop_fail_plumes", sample_crop_fail[:10])


if __name__ == "__main__":
    main()
