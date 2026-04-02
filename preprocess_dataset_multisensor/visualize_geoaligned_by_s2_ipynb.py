from pathlib import Path
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from netCDF4 import Dataset
from scipy.ndimage import zoom

# ======================
# Notebook configs
# ======================
# emi20231010t052236p04020-C
# emi20240825t064423p04027-B
# emi20241202t071056p05010-D
# emi20241217t045523p03003-R

# emi20240130t112621p08007-B
# emi20240621t082746p06048-B
# emi20240930t091314p06024-C
# emi20250128t063810p05012-B
CASE_ID = "emi20241217t045523p03003-R"
CSV_PATH = "preprocess_dataset_multisensor/master_multisensor_outer_join.csv"
OUT_DIR = "preprocess_dataset_multisensor/case_study_geoaligned"
VIS_MODE = "pca"  # "true", "swir", "pca", "gray"
EMIT_USE_RAW = False
EMIT_RAW_CSV = "preprocess_dataset_EMIT/merged_with_emit_tag.csv"
EMIT_RAW_DIR = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_EMIT"
EMIT_RAW_KEY = "emit_granule_id"  # or emit_-90_granule_id / emit_-180_granule_id
P_LOW, P_HIGH = 2, 98
TARGET = 512

# meter / pixel
GSD = {
    "s2": 10.0,
    "l89": 30.0,
    "emit": 60.0,
    "s5p": 3500.0,
}

# band index in 0-based order (assume common band ordering)
BANDS = {
    "s2_true": [3, 2, 1],      # B4,B3,B2
    "s2_swir": [11, 10, 8],    # B12,B11,B8A
    "l89_true": [3, 2, 1],     # B4,B3,B2
    "l89_swir": [6, 5, 4],     # B7,B6,B5
    # EMIT/WV3 simulated: pragmatic defaults, adjust if needed
    "emit_true": [4, 2, 1],
    "emit_swir": [15, 10, 7],
}

CH4_KEYS = ["methane_mixing_ratio_bias_corrected", "methane_mixing_ratio", "xch4"]


def resolve_path(p: str) -> str:
    if not isinstance(p, str):
        return ""
    if os.path.exists(p):
        return p
    # optional remap for mask legacy path
    if p.startswith("/data2/yuyao/methane_emission/"):
        alt = p.replace(
            "/data2/yuyao/methane_emission/",
            "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/",
            1,
        )
        if os.path.exists(alt):
            return alt
    return p


def stretch01(x: np.ndarray, p_low=P_LOW, p_high=P_HIGH) -> np.ndarray:
    a = x.astype(np.float32, copy=False)
    if np.all(~np.isfinite(a)):
        return np.zeros_like(a, dtype=np.float32)
    lo, hi = np.nanpercentile(a, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(a), np.nanmax(a)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(a, dtype=np.float32)
    y = (a - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    y = np.where(np.isfinite(y), y, 0.0)
    return y.astype(np.float32)


def resize_to(img: np.ndarray, target: int, order: int = 1) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target and w == target:
        return img
    if img.ndim == 2:
        return zoom(img, (target / h, target / w), order=order)
    return zoom(img, (target / h, target / w, 1.0), order=order)


def center_crop(arr: np.ndarray, size: int):
    h, w = arr.shape[:2]
    size = max(1, min(size, h, w))
    cy, cx = h // 2, w // 2
    y0 = max(0, cy - size // 2)
    x0 = max(0, cx - size // 2)
    y1 = min(h, y0 + size)
    x1 = min(w, x0 + size)
    y0 = max(0, y1 - size)
    x0 = max(0, x1 - size)
    return arr[y0:y1, x0:x1], (x0, y0, x1, y1)



def crop_around(arr: np.ndarray, cy: int, cx: int, size: int):
    h, w = arr.shape[:2]
    size = max(1, min(size, h, w))
    cy = int(np.clip(cy, 0, h - 1))
    cx = int(np.clip(cx, 0, w - 1))
    y0 = max(0, cy - size // 2)
    x0 = max(0, cx - size // 2)
    y1 = min(h, y0 + size)
    x1 = min(w, x0 + size)
    y0 = max(0, y1 - size)
    x0 = max(0, x1 - size)
    return arr[y0:y1, x0:x1], (x0, y0, x1, y1)

def read_hwc(path: str) -> np.ndarray:
    x = tifffile.imread(path)
    if x.ndim == 3:
        if x.shape[2] <= 64:
            return x.astype(np.float32)
        return np.transpose(x, (1, 2, 0)).astype(np.float32)
    if x.ndim == 2:
        return x[:, :, None].astype(np.float32)
    raise ValueError(f"Unsupported TIFF shape: {x.shape}")


def pick_rgb(img_hwc: np.ndarray, idx3):
    c = img_hwc.shape[2]
    safe = [min(max(i, 0), c - 1) for i in idx3]
    rgb = img_hwc[:, :, safe]
    out = np.zeros_like(rgb, dtype=np.float32)
    for k in range(3):
        out[:, :, k] = stretch01(rgb[:, :, k])
    return out


def read_s5p_2d(path_nc: str):
    with Dataset(path_nc, "r") as ds:
        prod = ds.groups["PRODUCT"]
        key = None
        for k in CH4_KEYS:
            if k in prod.variables:
                key = k
                break
        if key is None:
            raise KeyError("No CH4 variable found")
        v = prod.variables[key]
        arr = np.array(v[:], dtype=np.float32)
        attrs = getattr(v, "__dict__", {})
        fv = attrs.get("_FillValue", None)
        mv = attrs.get("missing_value", None)
        if fv is not None:
            arr = np.where(arr == np.float32(fv), np.nan, arr)
        if mv is not None:
            arr = np.where(arr == np.float32(mv), np.nan, arr)
        arr = np.where(np.abs(arr) > 1e20, np.nan, arr)
        if arr.ndim == 3:
            arr = arr[0]
    return arr, key



def pca_rgb(img_hwc: np.ndarray, black_invalid: bool = True) -> np.ndarray:
    x = img_hwc.astype(np.float32, copy=False)
    h, w, c = x.shape
    flat = x.reshape(-1, c)
    valid = np.all(np.isfinite(flat), axis=1)
    flat_valid = flat[valid]
    if flat_valid.size == 0:
        return np.zeros((h, w, 3), dtype=np.float32)
    mean = flat_valid.mean(axis=0, keepdims=True)
    flat_valid = flat_valid - mean
    # SVD for PCA
    try:
        _, _, vt = np.linalg.svd(flat_valid, full_matrices=False)
        comps = vt[:3]
    except np.linalg.LinAlgError:
        comps = np.eye(c, dtype=np.float32)[:3]
    proj = (x.reshape(-1, c) - mean).dot(comps.T).reshape(h, w, 3)
    out = np.zeros_like(proj, dtype=np.float32)
    for k in range(3):
        out[:, :, k] = stretch01(proj[:, :, k])
    if black_invalid:
        mask = valid_mask_from_data(x)
        if mask is not None:
            out[~mask] = 0.0
    return out


def gray_rgb(img_hwc: np.ndarray) -> np.ndarray:
    x = img_hwc.astype(np.float32, copy=False)
    g = np.nanmean(x, axis=2)
    g = stretch01(g)
    return np.stack([g, g, g], axis=2)

def match_rgb_stats(src: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    out = src.copy()
    for k in range(3):
        s = src[:, :, k]
        r = ref[:, :, k]
        s_mask = np.isfinite(s)
        r_mask = np.isfinite(r)
        if mask is not None:
            s_mask = s_mask & mask
        if not np.any(s_mask) or not np.any(r_mask):
            continue
        s_mean = float(np.mean(s[s_mask]))
        s_std = float(np.std(s[s_mask]))
        r_mean = float(np.mean(r[r_mask]))
        r_std = float(np.std(r[r_mask]))
        if s_std < 1e-6:
            continue
        out[:, :, k] = (s - s_mean) * (r_std / s_std) + r_mean
        out[:, :, k] = np.clip(out[:, :, k], 0.0, 1.0)
    return out

def match_rgb_hist(src: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None, bins: int = 256) -> np.ndarray:
    out = src.copy()
    for k in range(3):
        s = src[:, :, k]
        r = ref[:, :, k]
        s_mask = np.isfinite(s)
        r_mask = np.isfinite(r)
        if mask is not None:
            s_mask = s_mask & mask
        # also exclude near-zero in ref to avoid black bias
        r_mask = r_mask & (r > 1e-6)
        if not np.any(s_mask) or not np.any(r_mask):
            continue
        s_vals = s[s_mask]
        r_vals = r[r_mask]
        # histogram matching on [0,1]
        s_hist, bin_edges = np.histogram(s_vals, bins=bins, range=(0.0, 1.0), density=True)
        r_hist, _ = np.histogram(r_vals, bins=bins, range=(0.0, 1.0), density=True)
        s_cdf = np.cumsum(s_hist)
        r_cdf = np.cumsum(r_hist)
        s_cdf = s_cdf / s_cdf[-1]
        r_cdf = r_cdf / r_cdf[-1]
        # map src values by CDF matching
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
        # for each src bin, find nearest ref cdf
        r_vals_interp = np.interp(s_cdf, r_cdf, bin_centers)
        # map each pixel
        s_flat = s.flatten()
        s_idx = np.clip(np.searchsorted(bin_edges, s_flat, side='right') - 1, 0, bins - 1)
        mapped = r_vals_interp[s_idx].reshape(s.shape)
        out[:, :, k] = np.clip(mapped, 0.0, 1.0)
    return out


def valid_mask_from_data(img_hwc: np.ndarray) -> np.ndarray:
    if img_hwc is None:
        return None
    x = img_hwc.astype(np.float32, copy=False)
    finite = np.all(np.isfinite(x), axis=2)
    nonzero = np.any(np.abs(x) > 1e-3, axis=2)
    return finite & nonzero



def overlay_mask(rgb: np.ndarray, mask2d: np.ndarray, color=(1, 0, 0), alpha=0.5):
    out = rgb.copy()
    m = np.isfinite(mask2d) & (mask2d > 0)
    for i in range(3):
        out[:, :, i] = np.where(m, (1 - alpha) * out[:, :, i] + alpha * color[i], out[:, :, i])
    return out



def read_mask_2d(path: str) -> np.ndarray | None:
    if not path or not os.path.exists(path):
        return None
    m = tifffile.imread(path).astype(np.float32)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m


def normalize_mask(mask2d: np.ndarray) -> np.ndarray:
    m = mask2d.astype(np.float32, copy=False)
    m = np.where(np.isfinite(m), m, 0.0)
    pos = m[m > 0]
    if pos.size == 0:
        return np.zeros_like(m, dtype=np.float32)
    lo, hi = np.nanpercentile(pos, [P_LOW, P_HIGH])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(pos), np.nanmax(pos)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return (m > 0).astype(np.float32)
    y = (m - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    return y.astype(np.float32)


def overlay_mask_on_ax(ax, mask2d: np.ndarray | None, alpha_max: float = 0.85, gamma: float = 1.25, dark_scale: float = 0.6):
    if mask2d is None:
        return
    m_norm = normalize_mask(mask2d)
    if not np.any(m_norm > 0):
        return
    m_vis = np.power(m_norm, gamma) * dark_scale
    ax.imshow(m_vis, cmap="inferno", alpha=(m_vis * alpha_max))



def load_emit_raw_reflectance(case_id: str, key: str = EMIT_RAW_KEY):
    if not Path(EMIT_RAW_CSV).exists():
        return None
    granule_id = None
    with open(EMIT_RAW_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("plume_id") == case_id:
                granule_id = r.get(key)
                break
    if not granule_id:
        return None
    nc_path = Path(EMIT_RAW_DIR) / f"{granule_id}.nc"
    if not nc_path.exists():
        return None
    sub = f"netcdf:{nc_path}:reflectance"
    return sub


def read_emit_raw(case_id: str, mode: str):
    sub = load_emit_raw_reflectance(case_id)
    if sub is None:
        return None
    with rasterio.open(sub) as ds:
        if mode in ("true", "swir"):
            bands = BANDS["emit_true"] if mode == "true" else BANDS["emit_swir"]
            bands = [b + 1 for b in bands]
        else:
            # for PCA/gray use evenly spaced subset to keep memory reasonable
            n = min(30, ds.count)
            bands = np.linspace(1, ds.count, num=n, dtype=int).tolist()
        arr = ds.read(bands).astype(np.float32)
    # to HWC
    arr = np.transpose(arr, (1, 2, 0))
    return arr


def read_raw_plume_mask(case_id: str):
    base = Path("carbon_mapper_data_masks") / case_id / "plume.tif"
    if not base.exists():
        return None, None, None
    with rasterio.open(base) as ds:
        arr = ds.read(1).astype(np.float32)
        return arr, ds.transform, ds.crs


def reproject_mask_to(mask: np.ndarray, src_transform, src_crs, dst_shape, dst_transform, dst_crs):
    dst = np.zeros(dst_shape, dtype=np.float32)
    reproject(
        source=mask,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )
    return dst


def build_centered_transform(ref_transform, ref_width, ref_height, pixel_size: float, out_size: int):
    cx, cy = rasterio.transform.xy(ref_transform, ref_height / 2, ref_width / 2, offset="center")
    return Affine.translation(cx - (out_size * pixel_size) / 2, cy + (out_size * pixel_size) / 2) * Affine.scale(pixel_size, -pixel_size)

def plot_rgb_with_mask(ax, rgb: np.ndarray, mask2d: np.ndarray | None, title: str):
    ax.imshow(rgb)
    overlay_mask_on_ax(ax, mask2d)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

out_dir = Path(OUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

# helper funcs for multi-timepoint plotting
TIMEPOINTS = ["-360", "-90", "0"]  # first two are no-plume, last is plume

def _get_row_path(row, key: str) -> str:
    if key is None or key not in row.index:
        return ""
    v = row.get(key)
    if pd.isna(v):
        return ""
    return resolve_path(str(v))


def _s5p_col(tp: str) -> str | None:
    if tp == "0":
        return "S5p_path"
    if tp == "-90":
        return "s5p_minus90_path"
    if tp == "-360":
        return "s5p_minus360_path"
    return None


def _blank_rgb() -> np.ndarray:
    return np.zeros((TARGET, TARGET, 3), dtype=np.float32)


df = pd.read_csv(CSV_PATH)
row = df[df["plume_id"] == CASE_ID].iloc[0]

# same geo extent as S2(512 @10m)
box_s2 = TARGET
box_l89 = max(1, int(round(TARGET * GSD["s2"] / GSD["l89"])))
box_emit = max(1, int(round(TARGET * GSD["s2"] / GSD["emit"])))
box_s5p = max(1, int(round(TARGET * GSD["s2"] / GSD["s5p"])))

raw_mask, raw_transform, raw_crs = read_raw_plume_mask(CASE_ID)

fig_h = 4.5 * len(TIMEPOINTS)
fig, axes = plt.subplots(2 * len(TIMEPOINTS), 4, figsize=(18, fig_h), dpi=180)
if len(TIMEPOINTS) == 1:
    axes = np.array(axes).reshape(2, 4)

for t_i, tp in enumerate(TIMEPOINTS):
    s2_path = _get_row_path(row, f"s2_{tp}_std_512")
    l89_path = _get_row_path(row, f"l89_{tp}_std_512")
    emit_tp_used = tp
    emit_path = _get_row_path(row, f"emit_{tp}_simulated_512_path")
    if not emit_path and tp == "-360":
        emit_tp_used = "-180"
        emit_path = _get_row_path(row, "emit_-180_simulated_512_path")
    s5p_path = _get_row_path(row, _s5p_col(tp))

    s2 = read_hwc(s2_path) if s2_path else None
    l89 = read_hwc(l89_path) if l89_path else None
    emit = None
    if EMIT_USE_RAW and tp == "0":
        emit = read_emit_raw(CASE_ID, VIS_MODE)
    if emit is None and emit_path:
        emit = read_hwc(emit_path)

    s2_rgb = _blank_rgb()
    l89_rgb = _blank_rgb()
    emit_rgb = _blank_rgb()

    if VIS_MODE == "pca":
        if s2 is not None:
            s2_rgb = pca_rgb(s2)
        if l89 is not None:
            l89_rgb = pca_rgb(l89)
        if emit is not None:
            emit_rgb = pca_rgb(emit)
            if l89 is not None:
                emit_valid = valid_mask_from_data(emit)
                emit_rgb = match_rgb_hist(emit_rgb, l89_rgb, mask=emit_valid)
                if emit_valid is not None:
                    emit_rgb[~emit_valid] = 0.0
    elif VIS_MODE == "gray":
        if s2 is not None:
            s2_rgb = gray_rgb(s2)
        if l89 is not None:
            l89_rgb = gray_rgb(l89)
        if emit is not None:
            emit_rgb = gray_rgb(emit)
    else:
        if s2 is not None:
            s2_rgb = pick_rgb(s2, BANDS["s2_true"] if VIS_MODE == "true" else BANDS["s2_swir"])
        if l89 is not None:
            l89_rgb = pick_rgb(l89, BANDS["l89_true"] if VIS_MODE == "true" else BANDS["l89_swir"])
        if emit is not None:
            emit_rgb = pick_rgb(emit, BANDS["emit_true"] if VIS_MODE == "true" else BANDS["emit_swir"])

    s2_crop, b2 = center_crop(s2_rgb, box_s2)
    l89_crop, b8 = center_crop(l89_rgb, box_l89)
    emit_crop, be = center_crop(emit_rgb, box_emit)

    s2_crop = resize_to(s2_crop, TARGET)
    l89_crop = resize_to(l89_crop, TARGET)
    emit_crop = resize_to(emit_crop, TARGET)

    # only show plume overlay for tp == "0"
    s2m = None
    l89m = None
    emitm = None
    s2m_top = None

    if tp == "0":
        s2_mask_path = _get_row_path(row, "s2_plume_mask_512_path")
        l89_mask_path = _get_row_path(row, "l89_mask_512_path")

        if raw_mask is None:
            s2_mask = read_mask_2d(s2_mask_path)
            l89_mask = read_mask_2d(l89_mask_path)
        else:
            s2_mask = None
            l89_mask = None

        if raw_mask is not None and raw_crs is not None:
            l89_ref = None

            if l89_path:
                with rasterio.open(l89_path) as ds_l89:
                    if ds_l89.crs is not None:
                        l89_ref = (ds_l89.transform, ds_l89.crs, ds_l89.width, ds_l89.height)
                        l89_full = reproject_mask_to(raw_mask, raw_transform, raw_crs, (ds_l89.height, ds_l89.width), ds_l89.transform, ds_l89.crs)
                        l89m, _ = center_crop(l89_full, box_l89)
                        l89m = resize_to(l89m, TARGET, order=1)

            if emit_path:
                with rasterio.open(emit_path) as ds_emit:
                    if ds_emit.crs is not None:
                        emit_full = reproject_mask_to(raw_mask, raw_transform, raw_crs, (ds_emit.height, ds_emit.width), ds_emit.transform, ds_emit.crs)
                        emitm, _ = center_crop(emit_full, box_emit)
                        emitm = resize_to(emitm, TARGET, order=1)

                        s2_transform = build_centered_transform(ds_emit.transform, ds_emit.width, ds_emit.height, GSD["s2"], TARGET)
                        s2m = reproject_mask_to(raw_mask, raw_transform, raw_crs, (TARGET, TARGET), s2_transform, ds_emit.crs)

            if s2m is None and l89_ref is not None:
                l89_transform, l89_crs, l89_w, l89_h = l89_ref
                s2_transform = build_centered_transform(l89_transform, l89_w, l89_h, GSD["s2"], TARGET)
                s2m = reproject_mask_to(raw_mask, raw_transform, raw_crs, (TARGET, TARGET), s2_transform, l89_crs)

        if s2m is None and s2_mask is not None:
            s2m, _ = center_crop(s2_mask, box_s2)
            s2m = resize_to(s2m, TARGET, order=1)
        if l89m is None and l89_mask is not None:
            l89m, _ = center_crop(l89_mask, box_l89)
            l89m = resize_to(l89m, TARGET, order=1)
        if emitm is None and s2_mask is not None:
            emitm, _ = center_crop(s2_mask, box_s2)
            emitm = resize_to(emitm, box_emit, order=1)
            emitm = resize_to(emitm, TARGET, order=1)

        s2m_top = s2m

    # S5P
    if s5p_path:
        s5p2d, s5p_key = read_s5p_2d(s5p_path)
        iy, ix = int(float(row["nearest_iy"])), int(float(row["nearest_ix"]))
        s5p_ctx, ctx_box = crop_around(s5p2d, iy, ix, 128)
        s5p_zoom, zoom_box = crop_around(s5p2d, iy, ix, 9)
        s5p_ctx_show = stretch01(s5p_ctx)
        s5p_zoom_show = resize_to(stretch01(s5p_zoom), TARGET)
        ctx_mx = ix - ctx_box[0]
        ctx_my = iy - ctx_box[1]
        zoom_size = max(1, zoom_box[2] - zoom_box[0])
        zoom_mx = (ix - zoom_box[0]) * (TARGET / zoom_size)
        zoom_my = (iy - zoom_box[1]) * (TARGET / zoom_size)
    else:
        s5p_key = "missing"
        s5p_ctx_show = np.zeros((128, 128), dtype=np.float32)
        s5p_zoom_show = np.zeros((TARGET, TARGET), dtype=np.float32)
        ctx_mx = ctx_my = zoom_mx = zoom_my = 0

    ax_top = axes[t_i * 2 + 0]
    ax_bot = axes[t_i * 2 + 1]

    panels_top = [
        (s2_rgb, b2, f"S2 source + S2-range box ({tp}d)"),
        (l89_rgb, b8, f"L89 source + S2-range box ({tp}d)"),
        (emit_rgb, be, f"EMIT source + S2-range box ({emit_tp_used}d)"),
        (s5p_ctx_show, None, f"S5P local context ({tp}d)"),
    ]
    for i, (img, box, title) in enumerate(panels_top):
        ax = ax_top[i]
        if i < 3:
            ax.imshow(img)
            x0, y0, x1, y1 = box
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="yellow", linewidth=1.5))
        else:
            ax.imshow(img, cmap="viridis")
            ax.scatter([ctx_mx], [ctx_my], s=28, c="red", marker="x", linewidths=1.2)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plume_mask_last = emitm if tp == "0" else None
    plot_rgb_with_mask(ax_bot[0], s2_crop, None, f"S2 aligned 512x512 ({tp}d)")
    plot_rgb_with_mask(ax_bot[1], l89_crop, None, f"L89 aligned to S2 extent ({tp}d)")
    plot_rgb_with_mask(ax_bot[2], emit_crop, plume_mask_last, f"EMIT aligned to S2 extent ({emit_tp_used}d)")
    ax_bot[3].imshow(s5p_zoom_show, cmap="viridis")
    ax_bot[3].scatter([zoom_mx], [zoom_my], s=40, c="red", marker="x", linewidths=1.5)
    ax_bot[3].set_title(f"S5P zoomed (pixel marked, {s5p_key}, {tp}d)", fontsize=9)
    ax_bot[3].axis("off")

fig.suptitle(f"{CASE_ID} | VIS_MODE={VIS_MODE} | S2-extent geospatial alignment", fontsize=12)
fig.tight_layout()
out = out_dir / f"{CASE_ID}_geoaligned_by_s2_{VIS_MODE}_3tp.png"
fig.savefig(out, bbox_inches="tight")
print("saved", out)
print("box pixels s2/l89/emit/s5p =", box_s2, box_l89, box_emit, box_s5p)
