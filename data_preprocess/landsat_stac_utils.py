# landsat_stac_utils.py

from datetime import timedelta, timezone
from typing import List, Dict, Optional
import requests
from carbon_mapper_sentinel2_plume_download import parse_iso_datetime
from pystac_client import Client

# USGS Landsat STAC server
STAC_URL = "https://landsatlook.usgs.gov/stac-server"

def get_landsat_stac_client() -> Client:
    # 可以全局复用，同一个进程里打开一次即可
    return Client.open(STAC_URL)

LANDSAT_STAC_SEARCH_URL = "https://landsatlook.usgs.gov/stac-server/search"
def dt_to_rfc3339(dt):
    """转成 STAC 需要的 'YYYY-MM-DDTHH:MM:SSZ' 格式"""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def fetch_landsat_items(plume_bounds, window_start, window_end, max_items=50):
    """
    用 LandsatLook STAC API 在给定 bbox + 时间窗口内搜索 L8/L9 C2 L2 SR 场景。

    plume_bounds: [lon_min, lat_min, lon_max, lat_max]
    window_start, window_end: datetime（tz-aware，最好是 UTC）
    """
    lon_min, lat_min, lon_max, lat_max = plume_bounds

    body = {
        "collections": ["landsat-c2l2-sr"],   # L8/L9 C2 L2 SR
        "platform": {"in": ["landsat-8", "landsat-9"]},
        "bbox": [lon_min, lat_min, lon_max, lat_max],
        "datetime": f"{dt_to_rfc3339(window_start)}/{dt_to_rfc3339(window_end)}",
        "limit": max_items,
        # 不强依赖 sortby，后面自己按时间排序也可以
    }

    items = []
    next_link = LANDSAT_STAC_SEARCH_URL

    while next_link:
        try:
            resp = requests.post(next_link, json=body if next_link == LANDSAT_STAC_SEARCH_URL else None)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"[error] STAC search failed: {exc}")
            break

        features = data.get("features", [])
        for feat in features:
            props = feat.get("properties", {})
            dt_str = props.get("datetime")
            if not dt_str:
                continue
            try:
                acq_time = parse_iso_datetime(dt_str)
            except Exception:
                continue

            scene_id = feat.get("id")  # 通常就是 LANDSAT_PRODUCT_ID
            if not scene_id.startswith(("LC08", "LC09")):
                continue
            if not scene_id:
                # 有些实现可能放在 landsat:landsat_product_id
                scene_id = props.get("landsat:landsat_product_id")

            if not scene_id:
                continue

            cloud_cover = props.get("eo:cloud_cover")
            if cloud_cover is None:
                cloud_cover = props.get("landsat:cloud_cover_land")
            if cloud_cover is not None:
                try:
                    cloud_cover = float(cloud_cover)
                except ValueError:
                    cloud_cover = None

            items.append({
                "scene_id": scene_id,
                "acq_time": acq_time,
                "cloud_cover": cloud_cover,
            })

        # 处理分页：STAC 通常用 links 里 rel="next"
        next_href = None
        for link in data.get("links", []):
            if link.get("rel") == "next":
                next_href = link.get("href")
                break

        # 我们一般一小块+14天里不会有很多景，简单点就只拿第一页
        # 如果你以后想更严谨，把上面 limit 减小点 + 打开这个就行
        # next_link = next_href
        next_link = None

    # 最后按时间排序一下
    items = sorted(items, key=lambda x: x["acq_time"])
    return items

def item_acq_datetime(item) -> Optional[object]:
    """
    从 STAC Item 中取出 acquisition datetime。
    一般在 properties['datetime']。
    """
    props = item.properties
    dt = props.get("datetime")
    # pystac 会帮你把 datetime 解析成 Python datetime，如果是 str 就自己 parse
    return dt


def item_scene_id(item) -> str:
    """
    获取 Landsat scene_id。一般可以直接用 item.id，
    也可以从 properties['landsat:scene_id'] 中取。
    """
    props = item.properties
    scene_id = props.get("landsat:scene_id")
    if scene_id:
        return scene_id
    return item.id


def select_landsat_items(items, event_dt, max_scenes=3):
    """
    从 STAC 返回的 items 里选出距离事件时间最近的最多 max_scenes 个场景。
    优先保留同一天的，然后前后各一景。
    """
    if not items:
        return []

    same_day = []
    before = []
    after = []

    for item in items:
        t = item["acq_time"]
        if t.date() == event_dt.date():
            same_day.append(item)
        elif t < event_dt:
            before.append(item)
        else:
            after.append(item)

    selected = []

    if same_day:
        # 同一天取离中心时间最近的
        closest_same_day = min(same_day, key=lambda p: abs((p["acq_time"] - event_dt).total_seconds()))
        selected.append(closest_same_day)

        if before:
            closest_before = max(before, key=lambda p: p["acq_time"])
            selected.append(closest_before)
        if after:
            closest_after = min(after, key=lambda p: p["acq_time"])
            selected.append(closest_after)
    else:
        # 没有同一天：直接按“离事件时间绝对距离”排序取前 max_scenes 个
        sorted_items = sorted(items, key=lambda p: abs((p["acq_time"] - event_dt).total_seconds()))
        selected = sorted_items[:max_scenes]

    # 按时间排序一下方便 debug 和后处理
    return sorted(selected, key=lambda p: p["acq_time"])
