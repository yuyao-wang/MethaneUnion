# download_landsat_for_cm.py

import os
import sys
import time
import threading
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
import tifffile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.utils import parse_args, load_config

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from landsat_c2_downloader import download_landsat_scene, normalize_scene_id
from landsat_stac_utils import (
    get_landsat_stac_client,
    fetch_landsat_items,
    select_landsat_items,
    item_acq_datetime,
    item_scene_id,
)

# 全局设置
WINDOW_SIZE = 512
# L8_PLUME_BASE_DIR = "/data2/yuyao/methane_emission/landsat_l2sp_plume_stacks"
base_dir = '/data2/yuyao/methane_emission/carbonmapper_data_l89_l2sp'
MAX_L8_PER_PLUME = 3
PLUME_COMPLETION_MARKER = "landsat_l2sp_complete.txt"
MAX_CLOUD_COVER_PERCENT = 20.0


def parse_iso_datetime(value: str) -> Optional[datetime]:
    if not isinstance(value, str) or len(value) == 0:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def datetime_to_iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def latlon_to_pixel(lat: float, lon: float, dataset: rasterio.io.DatasetReader):
    transformer = Transformer.from_crs("EPSG:4326", dataset.crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return ~dataset.transform * (x, y)


def crop_band_to_window(tif_path: str, plume_bounds: List[float]) -> Optional[np.ndarray]:
    if not os.path.exists(tif_path):
        print(f"[warn] band not found: {tif_path}")
        return None

    with rasterio.open(tif_path) as dataset:
        lon_min, lat_min, lon_max, lat_max = plume_bounds

        top_left = latlon_to_pixel(lat_max, lon_min, dataset)
        bottom_right = latlon_to_pixel(lat_min, lon_max, dataset)

        center_x = (top_left[0] + bottom_right[0]) / 2
        center_y = (top_left[1] + bottom_right[1]) / 2

        half = WINDOW_SIZE // 2
        col_start = int(np.floor(center_x - half))
        row_start = int(np.floor(center_y - half))
        col_end = col_start + WINDOW_SIZE
        row_end = row_start + WINDOW_SIZE

        if col_start < 0:
            col_end += -col_start
            col_start = 0
        if row_start < 0:
            row_end += -row_start
            row_start = 0
        if col_end > dataset.width:
            shift = col_end - dataset.width
            col_start -= shift
            col_end = dataset.width
        if row_end > dataset.height:
            shift = row_end - dataset.height
            row_start -= shift
            row_end = dataset.height

        col_start = max(0, col_start)
        row_start = max(0, row_start)
        window_width = max(0, col_end - col_start)
        window_height = max(0, row_end - row_start)

        if window_width == 0 or window_height == 0:
            print(f"[warn] empty window for {tif_path}")
            return None

        window = Window(col_start, row_start, window_width, window_height)
        clipped = dataset.read(1, window=window)
        return clipped


def parse_landsat_mtl(mtl_path: str) -> Dict[str, Optional[float]]:
    """
    解析 MTL.txt 中我们关心的字段：
      - DATE_ACQUIRED + SCENE_CENTER_TIME -> acq_datetime_iso
      - SUN_AZIMUTH, SUN_ELEVATION
      - IMAGE_QUALITY_OLI, IMAGE_QUALITY_TIRS
    """
    result = {
        "acq_datetime_iso": None,
        "sun_azimuth": None,
        "sun_elevation": None,
        "image_quality_oli": None,
        "image_quality_tirs": None,
    }

    if not os.path.exists(mtl_path):
        print(f"[warn] MTL not found: {mtl_path}")
        return result

    meta: Dict[str, str] = {}
    with open(mtl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("GROUP") or line.startswith("END_") or line == "END":
                continue
            if "=" not in line:
                continue
            k, v = [x.strip() for x in line.split("=", 1)]
            if v.startswith('"') and v.endswith('"'):
                v = v[1:-1]
            meta[k] = v

    date_str = meta.get("DATE_ACQUIRED")
    time_str = meta.get("SCENE_CENTER_TIME")

    if date_str and time_str:
        t = time_str
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        dt_str = f"{date_str}T{t}"
        try:
            dt = datetime.fromisoformat(dt_str)
            result["acq_datetime_iso"] = datetime_to_iso_z(dt)
        except Exception as e:
            print(f"[warn] failed to parse datetime from MTL: {dt_str} ({e})")

    def _get_float(k: str):
        v = meta.get(k)
        if v is None:
            return None
        try:
            return float(v)
        except ValueError:
            return None

    def _get_int(k: str):
        v = meta.get(k)
        if v is None:
            return None
        try:
            return int(v)
        except ValueError:
            return None

    result["sun_azimuth"] = _get_float("SUN_AZIMUTH")
    result["sun_elevation"] = _get_float("SUN_ELEVATION")
    result["image_quality_oli"] = _get_int("IMAGE_QUALITY_OLI")
    result["image_quality_tirs"] = _get_int("IMAGE_QUALITY_TIRS")

    return result


def build_landsat_stack_for_plume(
    scene_dir: str,
    scene_id: str,
    plume_bounds: List[float],
    out_tif_path: str,
) -> Optional[Dict[str, int]]:
    """
    从 scene_dir 中读取 SR_B1-7 + ST_B10，裁剪 WINDOW_SIZE × WINDOW_SIZE，并 stack 成 [8,H,W]。
    """
    band_suffixes = [f"SR_B{b}" for b in range(1, 8)] + ["ST_B10"]

    bands = []
    current_shape = None

    for suffix in band_suffixes:
        tif_name = f"{scene_id}_{suffix}.TIF"
        tif_path = os.path.join(scene_dir, tif_name)
        clipped = crop_band_to_window(tif_path, plume_bounds)
        if clipped is None:
            print(f"[warn] skip band {tif_name}")
            continue

        if current_shape is None:
            current_shape = clipped.shape
        else:
            if clipped.shape != current_shape:
                print(f"[warn] shape mismatch for {tif_name}: {clipped.shape} != {current_shape}")
                continue

        bands.append(clipped)

    if not bands:
        print(f"[warn] no valid bands for {scene_id}")
        return None

    stacked = np.stack(bands, axis=0)  # [B,H,W]
    os.makedirs(os.path.dirname(out_tif_path), exist_ok=True)
    print(f"[write] L8 stack -> {out_tif_path}")
    tifffile.imwrite(out_tif_path, stacked)

    h, w = current_shape
    return {"height": int(h), "width": int(w)}


def process_single_landsat_scene(
    scene_id: str,
    landsat_raw_root: str,
    plume_dir: str,
    plume_bounds: List[float],
    download_scene: bool = True,
) -> Optional[Dict[str, object]]:
    """
    对单个场景：
      1. download_landsat_scene 到 landsat_raw_root/scene_id
      2. parse MTL 取时间 / 太阳角 / 质量
      3. 裁剪 stack 写到 plume_dir
    """
    product_id = normalize_scene_id(scene_id)
    if download_scene:
        download_landsat_scene(scene_id, landsat_raw_root)
    scene_dir = os.path.join(landsat_raw_root, product_id)

    mtl_path = os.path.join(scene_dir, f"{product_id}_MTL.txt")
    meta = parse_landsat_mtl(mtl_path)
    acq_dt_iso = meta.get("acq_datetime_iso")

    if acq_dt_iso is None:
        parts = scene_id.split("_")
        if len(parts) >= 4 and len(parts[3]) >= 8:
            date_str = parts[3][:8]
            try:
                dt = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
                acq_dt_iso = datetime_to_iso_z(dt)
            except Exception:
                acq_dt_iso = None

    out_tif_name = f"l8_{product_id}.tif"
    out_tif_path = os.path.join(plume_dir, out_tif_name)

    dims = build_landsat_stack_for_plume(scene_dir, product_id, plume_bounds, out_tif_path)
    if dims is None:
        return None

    return {
        "scene_id": scene_id,
        "product_id": product_id,
        "datetime": acq_dt_iso or "",
        "tif_path": out_tif_path,
        "height": dims["height"],
        "width": dims["width"],
        "sun_azimuth": meta.get("sun_azimuth"),
        "sun_elevation": meta.get("sun_elevation"),
        "image_quality_oli": meta.get("image_quality_oli"),
        "image_quality_tirs": meta.get("image_quality_tirs"),
        "root_dir": scene_dir,
    }


def update_global_progress(tracker):
    if tracker is None:
        return
    with tracker["lock"]:
        tracker["completed"] += 1
        completed = tracker["completed"]
        total = tracker["total"]
        elapsed = time.time() - tracker["start_time"]
        avg_time = elapsed / completed if completed > 0 else 0
        remaining = max(0, total - completed)
        eta = remaining * avg_time
        progress_bar = tracker.get("tqdm")
        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix({"ETA(min)": f"{eta/60:.1f}"}, refresh=False)
        else:
            print(f"Completed {completed}/{total} | Elapsed: {elapsed/60:.2f} min | ETA: {eta/60:.2f} min")


def download_task_l8(row_index, row_data, plume_bounds, progress_tracker, max_scenes=3):
    """
    针对单个 plume：
    - 用 LandsatLook STAC 搜索 L8/L9 C2 L2 场景
    - 选出最多 max_scenes 个离事件时间最近的
    - 调用 download_landsat_scene 下载原始 L8 包到 raw_data_dir
    - 返回简单的 meta 给主线程写 CSV
    """
    plume_id = str(row_data.get('plume_id', 'unknown'))
    plume_dir = os.path.join(base_dir, plume_id)
    plume_marker_file = os.path.join(plume_dir, PLUME_COMPLETION_MARKER)

    try:
        if os.path.exists(plume_marker_file):
            print(f"[skip] plume {plume_id}; completion marker found at {plume_marker_file}")
            return {'index': row_index, 'selected_scenes': [], 'has_same_day_l8': 0}

        event_dt = parse_iso_datetime(row_data.get('datetime'))
        if event_dt is None:
            print(f"[warn] plume {plume_id}: invalid datetime")
            return {'index': row_index, 'selected_scenes': [], 'has_same_day_l8': 0}

        event_dt = event_dt.astimezone(timezone.utc)

        # 时间窗口：前后 7 天（你可以保持和 S2 一致）
        window_start = event_dt - timedelta(days=7)
        window_end = event_dt + timedelta(days=7)

        # --- Step 1: STAC 搜索 ---
        items = fetch_landsat_items(plume_bounds, window_start, window_end)
        if not items:
            print(f"[info] plume {plume_id}: no L8/L9 scenes found in STAC")
            return {'index': row_index, 'selected_scenes': [], 'has_same_day_l8': 0}

        # 仅保留云量低于阈值的场景
        low_cloud_items = [
            item for item in items
            if item.get("cloud_cover") is not None and item["cloud_cover"] < MAX_CLOUD_COVER_PERCENT
        ]
        if not low_cloud_items:
            print(f"[info] plume {plume_id}: no low-cloud (<{MAX_CLOUD_COVER_PERCENT}%) scenes available")
            return {'index': row_index, 'selected_scenes': [], 'has_same_day_l8': 0}

        # --- Step 2: 选场景 ---
        selected_items = select_landsat_items(low_cloud_items, event_dt, max_scenes=max_scenes)
        has_same_day = 1 if any(it["acq_time"].date() == event_dt.date() for it in selected_items) else 0

        # --- Step 3: 下载原始 L8 包 ---
        out_root = config['raw_data_dir_l8']  # 建议在 config 里单独加一个 raw_data_dir_l8
        os.makedirs(out_root, exist_ok=True)
        os.makedirs(plume_dir, exist_ok=True)

        recorded_scenes = []

        for it in selected_items:
            scene_id = it["scene_id"]
            product_id = normalize_scene_id(scene_id)
            acq_time = it["acq_time"]

            # 调用你前面写的 boto3 版本 downloader
            try:
                download_landsat_scene(scene_id, out_root)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"[error] plume {plume_id}: download scene {scene_id} failed: {exc}")
                continue

            # 这里先只记录 scene_id + acq_time + 本地路径
            # 后面你可以在单独的 L8 处理脚本里：
            #   - 遍历 scene 文件夹，合成 multi-band tif
            #   - 解析 MTL.txt 拿 SUN_AZIMUTH / SUN_ELEVATION / IMAGE_QUALITY_* 等
            try:
                scene_info = process_single_landsat_scene(
                    scene_id,
                    out_root,
                    plume_dir,
                    plume_bounds,
                    download_scene=False,
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"[error] plume {plume_id}: process scene {scene_id} failed: {exc}")
                continue

            if scene_info is None:
                continue

            if not scene_info.get("datetime"):
                scene_info["datetime"] = datetime_to_iso_z(acq_time)

            recorded_scenes.append(scene_info)

        if recorded_scenes:
            # 写 plume 层面的 completion marker
            with open(plume_marker_file, 'w') as f:
                f.write(datetime_to_iso_z(datetime.now(timezone.utc)))

        return {
            'index': row_index,
            'selected_scenes': recorded_scenes,
            'has_same_day_l8': has_same_day,
        }

    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"[error] Unknown error while processing plume {row_data.get('plume_id')}: {exc}")
        return {'index': row_index, 'selected_scenes': [], 'has_same_day_l8': 0}
    finally:
        update_global_progress(progress_tracker)


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    landsat_raw_root = config.get("l8_raw_data_dir")
    # os.makedirs(landsat_raw_root, exist_ok=True)
    # os.makedirs(L8_PLUME_BASE_DIR, exist_ok=True)

    merged_csv_path = "/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file.csv"
    output_csv_path = "/data2/yuyao/methane_emission/carbon_mapper_data/csvs/merged_file_with_l8.csv"

    df = pd.read_csv(merged_csv_path)

    # 新增 L8 列
    new_cols = []
    for i in range(1, MAX_L8_PER_PLUME + 1):
        new_cols.extend([
            f"l8_{i}_scene_id",
            f"l8_{i}_datetime",
            f"l8_{i}_tif",
            f"l8_{i}_height",
            f"l8_{i}_width",
            f"l8_{i}_sun_azimuth",
            f"l8_{i}_sun_elevation",
            f"l8_{i}_image_quality_oli",
            f"l8_{i}_image_quality_tirs",
        ])
    for col in new_cols:
        if col not in df.columns:
            df[col] = ""

    # 只处理有 plume_tif 的行
    plume_tif_mask = df["plume_tif"].apply(lambda v: isinstance(v, str) and len(v) > 0)
    processable_mask = plume_tif_mask

    total_processable = int(processable_mask.sum())
    overall_start_time = time.time()
    progress_tracker = None
    progress_bar = None
    if total_processable > 0:
        if tqdm is not None:
            progress_bar = tqdm(total=total_processable, desc="L8/L9 plumes", dynamic_ncols=True)
        progress_tracker = {
            "lock": threading.Lock(),
            "completed": 0,
            "total": total_processable,
            "start_time": overall_start_time,
            "tqdm": progress_bar,
        }

    stac_client = get_landsat_stac_client()

    futures = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for index, row in df.iterrows():
            if not processable_mask.iloc[index]:
                continue
            
            lat = row['plume_latitude']
            lon = row['plume_longitude']
    
            plume_bounds = [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01]
    
            futures.append(executor.submit(
                download_task_l8,
                index,
                row.to_dict(),
                plume_bounds,
                progress_tracker
            ))

    results = [f.result() for f in futures]

    if progress_bar is not None:
        progress_bar.close()

    # 写回 CSV
    for res in results:
        if res is None:
            continue
        idx = res["index"]
        scenes = res.get("selected_scenes", [])
        for i in range(MAX_L8_PER_PLUME):
            prefix = f"l8_{i+1}_"
            if i < len(scenes):
                info = scenes[i]
                df.at[idx, prefix + "scene_id"] = info.get("scene_id", "")
                df.at[idx, prefix + "datetime"] = info.get("datetime", "")
                df.at[idx, prefix + "tif"] = info.get("tif_path", "")
                df.at[idx, prefix + "height"] = info.get("height", "")
                df.at[idx, prefix + "width"] = info.get("width", "")
                df.at[idx, prefix + "sun_azimuth"] = info.get("sun_azimuth", "")
                df.at[idx, prefix + "sun_elevation"] = info.get("sun_elevation", "")
                df.at[idx, prefix + "image_quality_oli"] = info.get("image_quality_oli", "")
                df.at[idx, prefix + "image_quality_tirs"] = info.get("image_quality_tirs", "")

    df.to_csv(output_csv_path, index=False)
    total_elapsed = time.time() - overall_start_time
    print(f"All tasks completed in {total_elapsed/60:.2f} minutes.")
    print("Output saved to:", output_csv_path)
