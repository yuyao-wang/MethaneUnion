# landsat_stac_utils.py

from datetime import timedelta, timezone
from typing import List, Dict, Optional
import requests
from carbon_mapper_sentinel2_plume_download import parse_iso_datetime
from pystac_client import Client

# USGS Landsat STAC server
STAC_URL = "https://landsatlook.usgs.gov/stac-server"

def get_landsat_stac_client() -> Client:
    # Translated comment
    return Client.open(STAC_URL)

LANDSAT_STAC_SEARCH_URL = "https://landsatlook.usgs.gov/stac-server/search"
def dt_to_rfc3339(dt):
    """Translated to English."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def fetch_landsat_items(plume_bounds, window_start, window_end, max_items=50):
    """
 LandsatLook STAC API bbox + time L8/L9 C2 L2 SR .

    plume_bounds: [lon_min, lat_min, lon_max, lat_max]
 window_start, window_end: datetime(tz-aware, UTC)
    """
    lon_min, lat_min, lon_max, lat_max = plume_bounds

    body = {
        "collections": ["landsat-c2l2-sr"],   # L8/L9 C2 L2 SR
        "platform": {"in": ["landsat-8", "landsat-9"]},
        "bbox": [lon_min, lat_min, lon_max, lat_max],
        "datetime": f"{dt_to_rfc3339(window_start)}/{dt_to_rfc3339(window_end)}",
        "limit": max_items,
        # Translated comment
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

            scene_id = feat.get("id")  # Translated comment
            if not scene_id.startswith(("LC08", "LC09")):
                continue
            if not scene_id:
                # Translated comment
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

        # Translated comment
        next_href = None
        for link in data.get("links", []):
            if link.get("rel") == "next":
                next_href = link.get("href")
                break

        # Translated comment
        # Translated comment
        # next_link = next_href
        next_link = None

    # Translated comment
    items = sorted(items, key=lambda x: x["acq_time"])
    return items

def item_acq_datetime(item) -> Optional[object]:
    """
 STAC Item acquisition datetime.
 properties['datetime'].
    """
    props = item.properties
    dt = props.get("datetime")
    # Translated comment
    return dt


def item_scene_id(item) -> str:
    """
 Landsat scene_id. item.id,  properties['landsat:scene_id'] .
    """
    props = item.properties
    scene_id = props.get("landsat:scene_id")
    if scene_id:
        return scene_id
    return item.id


def select_landsat_items(items, event_dt, max_scenes=3):
    """
 STAC items distancetime max_scenes .
 , .
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
        # Translated comment
        closest_same_day = min(same_day, key=lambda p: abs((p["acq_time"] - event_dt).total_seconds()))
        selected.append(closest_same_day)

        if before:
            closest_before = max(before, key=lambda p: p["acq_time"])
            selected.append(closest_before)
        if after:
            closest_after = min(after, key=lambda p: p["acq_time"])
            selected.append(closest_after)
    else:
        # Translated comment
        sorted_items = sorted(items, key=lambda p: abs((p["acq_time"] - event_dt).total_seconds()))
        selected = sorted_items[:max_scenes]

    # Translated comment
    return sorted(selected, key=lambda p: p["acq_time"])
