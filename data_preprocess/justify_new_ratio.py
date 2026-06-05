import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
from rasterio.warp import reproject, Resampling
import tifffile

CSV_PATH = Path("../carbon_mapper_data/csvs/merged_file_with_s2_l8_filtered_with_flags_low_cloud_only.csv")
MASK_BASE = Path("/data2/yuyao/methane_emission/carbon_mapper_data_masks")
S2_CHIP_BASE = Path("/data2/yuyao/methane_emission/carbonmapper_data_s2_l2a")
L8_CHIP_BASE = Path("/data2/yuyao/methane_emission/carbonmapper_data_l89_l2sp")

TAU = 2.0
ALPHA = 1.0
BETA = 0.3
EPS = 1e-6
DEBUG_EVENTS = 5

# Translated comment
DMDO_CACHE = {}

# Translated comment
GAMMA_S2 = 4.0
C_S2 = -1.5
GAMMA_L8 = 3.0
C_L8 = -1.2


def _resize_mask_nearest(mask_arr, target_height, target_width):
    """Simple nearest-neighbor resize for non-georeferenced masks."""

    if mask_arr.shape == (target_height, target_width):
        return mask_arr

    if target_height <= 0 or target_width <= 0:
        raise ValueError("Invalid target size for mask resize.")

    src_h, src_w = mask_arr.shape
    y_idx = np.linspace(0, src_h - 1, target_height)
    x_idx = np.linspace(0, src_w - 1, target_width)
    y_idx = np.clip(np.round(y_idx).astype(int), 0, src_h - 1)
    x_idx = np.clip(np.round(x_idx).astype(int), 0, src_w - 1)
    return mask_arr[np.ix_(y_idx, x_idx)]


def _ensure_channel_first(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    if arr.ndim == 3:
        if arr.shape[0] <= 32 and arr.shape[1] >= arr.shape[0]:
            return arr
        if arr.shape[-1] <= 32:
            return np.moveaxis(arr, -1, 0)
    raise ValueError(f"Unexpected array shape for chip: {arr.shape}")


def _read_image_with_profile(image_path):
    arr = tifffile.imread(image_path)
    arr = _ensure_channel_first(arr)
    profile = {
        "height": arr.shape[1],
        "width": arr.shape[2],
        "transform": None,
        "crs": None,
    }
    return arr, profile


def _read_mask_with_profile(mask_path):
    try:
        with rasterio.open(mask_path) as src_mask:
            arr = src_mask.read(1)
            profile = {
                "height": src_mask.height,
                "width": src_mask.width,
                "transform": src_mask.transform,
                "crs": src_mask.crs,
            }
            return arr, profile
    except Exception:
        arr = tifffile.imread(mask_path)
        if arr.ndim == 3:
            arr = arr[..., 0]
        profile = {
            "height": arr.shape[0],
            "width": arr.shape[1],
            "transform": None,
            "crs": None,
        }
        return arr, profile


def _load_mask_like_image(mask_path, ref_profile):
    """Read plume mask and align it to the reference raster grid."""

    mask_arr, mask_profile = _read_mask_with_profile(mask_path)
    if (
        mask_profile["height"] == ref_profile["height"]
        and mask_profile["width"] == ref_profile["width"]
        and mask_profile["transform"] == ref_profile["transform"]
    ):
        return mask_arr

    target_shape = (ref_profile["height"], ref_profile["width"])
    ref_crs = ref_profile.get("crs")
    src_crs = mask_profile.get("crs")
    ref_transform = ref_profile.get("transform")
    src_transform = mask_profile.get("transform")

    if (
        ref_crs is None
        or src_crs is None
        or ref_transform is None
        or src_transform is None
    ):
        return _resize_mask_nearest(mask_arr, *target_shape)

    dest = np.zeros(target_shape, dtype=np.float32)
    reproject(
        source=mask_arr,
        destination=dest,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.nearest,
        dst_nodata=0,
    )
    return dest


def _compute_DM_DO(
    image_path,
    plume_mask_path,
    methane_bands,
    other_bands,
    eps=1e-6,
    debug=False,
    debug_prefix="",
):
    """Read a plume chip and compute (D_M, D_O) metrics used by MEQR and M_s."""

    img_path = Path(image_path)
    mask_path = Path(plume_mask_path)

    if not img_path.exists() or not mask_path.exists():
        return np.nan, np.nan

    key = (str(img_path), str(mask_path))
    if key in DMDO_CACHE:
        return DMDO_CACHE[key]

    img, ref_profile = _read_image_with_profile(img_path)
    mask = _load_mask_like_image(mask_path, ref_profile)

    band_count = img.shape[0]
    valid = np.isfinite(img[0])
    plume = (mask > 0) & valid
    bg = (mask == 0) & valid

    if plume.sum() < 10 or bg.sum() < 10:
        if debug:
            print(
                f"[DEBUG]{debug_prefix} insufficient plume/bg pixels "
                f"(plume={plume.sum()}, bg={bg.sum()})"
            )
        DMDO_CACHE[key] = (np.nan, np.nan)
        return np.nan, np.nan

    DM_list = []
    DO_list = []

    for b in methane_bands:
        if b < 1 or b > band_count:
            continue
        band = img[b - 1]
        plume_vals = band[plume]
        bg_vals = band[bg]
        if plume_vals.size < 10 or bg_vals.size < 10:
            if debug:
                print(
                    f"[DEBUG]{debug_prefix} skip methane band {b} "
                    f"(plume={plume_vals.size}, bg={bg_vals.size})"
                )
            continue
        mu_p = plume_vals.mean()
        mu_b = bg_vals.mean()
        std_b = bg_vals.std()
        dm_val = abs(mu_p - mu_b) / (std_b + eps)
        DM_list.append(dm_val)
        if debug:
            print(
                f"[DEBUG]{debug_prefix} methane band {b}: mu_p={mu_p:.4f} "
                f"mu_b={mu_b:.4f} std_b={std_b:.4f} -> DM={dm_val:.4f}"
            )

    for b in other_bands:
        if b < 1 or b > band_count:
            continue
        band = img[b - 1]
        bg_vals = band[bg]
        if bg_vals.size < 10:
            continue
        std_b = bg_vals.std()
        do_val = 1.0 / (1.0 + std_b)
        DO_list.append(do_val)
        if debug:
            print(
                f"[DEBUG]{debug_prefix} background band {b}: std_b={std_b:.4f} "
                f"-> DO contribution={do_val:.4f}"
            )

    D_M = float(np.mean(DM_list)) if DM_list else np.nan
    D_O = float(np.mean(DO_list)) if DO_list else np.nan
    if debug:
        print(f"[DEBUG]{debug_prefix} D_M={D_M} D_O={D_O}")

    DMDO_CACHE[key] = (D_M, D_O)
    return D_M, D_O


def compute_MEQR_for_event(
    s2_slots,
    l8_slots,
    plume_mask_path,
    tau=TAU,
    alpha=ALPHA,
    beta=BETA,
    eps=EPS,
    debug=False,
    debug_prefix="",
):
    """
    Combine Sentinel-2 / Landsat-8 evidence into:
      - Q_s (absolute usability with time/cloud/background)
      - MEQR (noisy-OR over sensors)
      - DM_s_max (max D_M over slots, used later to derive M_s for complementarity)
    """

    def cloud_to_B(cloud):
        if cloud is None or pd.isna(cloud):
            return 1.0
        v = float(cloud)
        if v > 1.0:
            v = v / 100.0
        v = min(max(v, 0.0), 1.0)
        return 1.0 - v

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def compute_Q_and_DMmax_for_slots(slots, methane_bands, other_bands, sensor_name):
        if not slots:
            if debug:
                print(f"[DEBUG]{debug_prefix} no slots for {sensor_name}")
            return 0.0, np.nan

        q_list = []
        dm_max = None

        for slot_idx, slot in enumerate(slots, start=1):
            img_path = slot.get("image_path")
            dt_days = slot.get("dt_days")
            cloud = slot.get("cloud_cover")

            if img_path is None or dt_days is None or pd.isna(dt_days):
                if debug:
                    print(
                        f"[DEBUG]{debug_prefix} skip {sensor_name} slot {slot_idx} "
                        f"(img_path={img_path}, dt={dt_days})"
                    )
                continue

            D_M, D_O = _compute_DM_DO(
                image_path=img_path,
                plume_mask_path=plume_mask_path,
                methane_bands=methane_bands,
                other_bands=other_bands,
                eps=eps,
                debug=debug,
                debug_prefix=f"{debug_prefix}[{sensor_name}#{slot_idx}] ",
            )
            if np.isnan(D_M) or np.isnan(D_O):
                if debug:
                    print(
                        f"[DEBUG]{debug_prefix} {sensor_name} slot {slot_idx} "
                        "DM/DO is NaN, skip"
                    )
                continue

            # Translated comment
            if dm_max is None or D_M > dm_max:
                dm_max = D_M

            # Translated comment
            T = np.exp(-abs(dt_days) / tau)
            z = alpha * D_M + beta * D_O
            S = sigmoid(z)
            B = cloud_to_B(cloud)
            q_list.append(T * S * B)

        Q = float(max(q_list)) if q_list else 0.0
        if dm_max is None:
            dm_max = np.nan
        return Q, dm_max

    S2_METHANE_BANDS = [11, 12]
    S2_OTHER_BANDS = [2, 4, 8]  # Translated comment

    L8_METHANE_BANDS = [6, 7]
    L8_OTHER_BANDS = [2, 4, 5]

    Q_s2, DM_s2_max = compute_Q_and_DMmax_for_slots(
        slots=s2_slots,
        methane_bands=S2_METHANE_BANDS,
        other_bands=S2_OTHER_BANDS,
        sensor_name="S2",
    )
    Q_l8, DM_l8_max = compute_Q_and_DMmax_for_slots(
        slots=l8_slots,
        methane_bands=L8_METHANE_BANDS,
        other_bands=L8_OTHER_BANDS,
        sensor_name="L8",
    )

    sensors_Q = []
    if Q_s2 > 0:
        sensors_Q.append(Q_s2)
    if Q_l8 > 0:
        sensors_Q.append(Q_l8)

    if not sensors_Q:
        meqr = 0.0
    elif len(sensors_Q) == 1:
        meqr = sensors_Q[0]
    else:
        prod = 1.0
        for q in sensors_Q:
            prod *= (1.0 - q)
        meqr = 1.0 - prod

    if debug:
        print(
            f"[DEBUG]{debug_prefix} Q_s2={Q_s2:.4f} Q_l8={Q_l8:.4f} -> MEQR={meqr:.4f}"
        )

    # Translated comment
    return meqr, Q_s2, Q_l8, DM_s2_max, DM_l8_max


def _to_timestamp(value):
    if value is None or pd.isna(value):
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def resolve_s2_chip_path(plume_id, dt_value):
    folder = S2_CHIP_BASE / str(plume_id)
    if not folder.exists():
        return None

    ts = _to_timestamp(dt_value)
    if ts is not None:
        candidate = folder / f"s2_{ts.strftime('%Y%m%dT%H%M%SZ')}.tif"
        if candidate.exists():
            return candidate

    best = None
    best_delta = None
    for tif_path in folder.glob("s2_*.tif"):
        suffix = tif_path.stem.split("_", 1)[-1]
        try:
            stamp_ts = pd.to_datetime(suffix, utc=True)
        except Exception:
            continue

        if ts is None:
            return tif_path

        delta = abs((stamp_ts - ts).total_seconds())
        if best is None or delta < best_delta:
            best = tif_path
            best_delta = delta

    return best


def resolve_l8_chip_path(raw_path, plume_id):
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None

    path = Path(raw_path.strip())
    if path.exists():
        return path

    fallback = L8_CHIP_BASE / str(plume_id) / path.name
    if fallback.exists():
        return fallback
    return None


def build_s2_slots(row):
    slots = []
    plume_id = row["plume_id"]
    for idx in (1, 2, 3):
        dt_col = f"s2_{idx}_datetime"
        dt_days_col = f"s2_{idx}_dt_days"
        cloud_col = f"s2_{idx}_cloud_cover"

        chip_path = resolve_s2_chip_path(plume_id, row.get(dt_col))
        if chip_path is None:
            continue

        slots.append(
            {
                "image_path": chip_path,
                "dt_days": row.get(dt_days_col),
                "cloud_cover": row.get(cloud_col),
            }
        )
    return slots


def build_l8_slots(row):
    slots = []
    plume_id = row["plume_id"]
    for idx in (1, 2, 3):
        path_col = f"l8_{idx}_tif"
        dt_days_col = f"l8_{idx}_dt_days"
        cloud_col = f"l8_{idx}_cloud_cover"

        chip_path = resolve_l8_chip_path(row.get(path_col), plume_id)
        if chip_path is None:
            continue

        slots.append(
            {
                "image_path": chip_path,
                "dt_days": row.get(dt_days_col),
                "cloud_cover": row.get(cloud_col),
            }
        )
    return slots


def compute_meqr_dataframe():
    df = pd.read_csv(CSV_PATH, low_memory=False)
    # df = df[df["has_same_day_s2_l8 (±24h)"] == 1].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["plume_id"] = df["plume_id"].astype(str)

    date_cols = [
        "s2_1_datetime",
        "s2_2_datetime",
        "s2_3_datetime",
        "l8_1_datetime",
        "l8_2_datetime",
        "l8_3_datetime",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
            dt_col = col.replace("datetime", "dt_days")
            df[dt_col] = (df[col] - df["datetime"]).dt.total_seconds() / 86400.0

    meqr_values = []
    qs2_values = []
    ql8_values = []
    dm_s2_max_values = []
    dm_l8_max_values = []

    for idx, (_, row) in enumerate(df.iterrows()):
        plume_id = row["plume_id"]
        mask_path = MASK_BASE / plume_id / "plume.tif"
        debug = idx < DEBUG_EVENTS

        s2_slots = build_s2_slots(row)
        l8_slots = build_l8_slots(row)

        meqr, q_s2, q_l8, dm_s2_max, dm_l8_max = compute_MEQR_for_event(
            s2_slots=s2_slots,
            l8_slots=l8_slots,
            plume_mask_path=mask_path,
            debug=debug,
            debug_prefix=f"[{plume_id}] ",
        )

        meqr_values.append(meqr)
        qs2_values.append(q_s2)
        ql8_values.append(q_l8)
        dm_s2_max_values.append(dm_s2_max)
        dm_l8_max_values.append(dm_l8_max)

        if debug:
            print(
                f"[DEBUG][{plume_id}] DONE -> "
                f"Q_s2={q_s2:.4f}, Q_l8={q_l8:.4f}, MEQR={meqr:.4f}, "
                f"DM_s2_max={dm_s2_max}, DM_l8_max={dm_l8_max}"
            )

    df["Q_s2"] = qs2_values
    df["Q_l8"] = ql8_values
    df["MEQR"] = meqr_values
    df["DM_s2_max"] = dm_s2_max_values
    df["DM_l8_max"] = dm_l8_max_values

    # Translated comment
    def _normalize_dm(col):
        dm = df[col].to_numpy(dtype=float)
        med = np.nanmedian(dm)
        p95 = np.nanpercentile(dm, 95)
        scale = max(p95 - med, 1e-6)
        m = (dm - med) / scale
        m = np.clip(m, 0.0, 1.0)
        return m

    df["M_s2"] = _normalize_dm("DM_s2_max")
    df["M_l8"] = _normalize_dm("DM_l8_max")
    # dm_s2 = df["DM_s2_max"].to_numpy(dtype=float)
    # dm_l8 = df["DM_l8_max"].to_numpy(dtype=float)
    # df["M_s2"] = 1.0 / (1.0 + np.exp(-(GAMMA_S2 * dm_s2 + C_S2)))
    # df["M_l8"] = 1.0 / (1.0 + np.exp(-(GAMMA_L8 * dm_l8 + C_L8)))

    return df


def plot_methane_separability_scatter(df, delta=0.1):
    """
 M_s2 / M_l8(, ).
 : M_l8 ∈ [(1-delta)*M_s2, (1+delta)*M_s2]
    """
    M_s2 = df["M_s2"].values
    M_l8 = df["M_l8"].values

    dominance = []
    for ms, ml in zip(M_s2, M_l8):
        if pd.isna(ms) or pd.isna(ml):
            dominance.append("undefined")
            continue

        # Translated comment
        lower = (1.0 - delta) * ms
        upper = (1.0 + delta) * ms

        if ml >= lower and ml <= upper:
            dominance.append("dual-acceptable")
        elif ml > upper:
            dominance.append("L8-dominant")
        else:  # ml < lower
            dominance.append("S2-dominant")

    df = df.copy()
    df["dominance"] = dominance

    colors = {
        "S2-dominant": "orange",
        "L8-dominant": "blue",
        "dual-acceptable": "gray",
        "undefined": "lightgray",
    }

    plt.figure(figsize=(8, 7))
    x = np.linspace(0, 1, 300)

    # Translated comment
    y_lower = (1.0 - delta) * x
    y_upper = (1.0 + delta) * x
    y_lower = np.clip(y_lower, 0.0, 1.0)
    y_upper = np.clip(y_upper, 0.0, 1.0)

    plt.fill_between(
        x,
        y_lower,
        y_upper,
        color="lightgray",
        alpha=0.3,
        label="dual-acceptable zone",
    )

    for dom in ["S2-dominant", "L8-dominant", "dual-acceptable", "undefined"]:
        mask = df["dominance"] == dom
        if mask.sum() == 0:
            continue
        plt.scatter(
            df.loc[mask, "M_s2"],
            df.loc[mask, "M_l8"],
            s=40,
            alpha=0.8,
            color=colors[dom],
            label=dom,
        )

    # Translated comment
    plt.plot(x, x, "k--", alpha=0.4)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("M_S2 (relative methane separability)")
    plt.ylabel("M_L8 (relative methane separability)")
    plt.title("S2 vs L8 methane separability (sensor-relative)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("methane_separability_scatter.png")
    plt.show()

def main():
    df = compute_meqr_dataframe()
    print(df[["plume_id", "Q_s2", "Q_l8", "MEQR", "M_s2", "M_l8"]].head())
    plot_methane_separability_scatter(df)


if __name__ == "__main__":
    main()