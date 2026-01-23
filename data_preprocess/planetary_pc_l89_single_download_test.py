"""
Utility script to test the Planetary Computer Landsat 8/9 download flow
for a single plume/offset. This lets you verify the STAC search results
and download parameters in isolation.

Usage example:

    python data_preprocess/planetary_pc_l89_single_download_test.py \
        --plume-id GAO20191019t151221p0000-C \
        --lat 31.423484 --lon -101.55237 \
        --event-datetime 2019-10-19T15:12:21Z \
        --offset 90 --download
"""

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

from pystac_client import Client

from Planetary_Computer_carbon_mapper_landsat89_L2SP_plume_90360_download import (
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    L8_PLATFORMS,
    MAX_CLOUD_COVER,
    PLANETARY_COMPUTER_STAC_URL,
    SEARCH_TOLERANCE_DAYS,
    add_record_to_cache,
    build_output_record,
    ensure_scene_assets,
    find_cached_record,
    get_normalized_scene_id,
    is_l89_item,
    parse_cloud_cover,
    parse_plume_bounds,
    sanitize_row_value,
    search_landsat_items,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Landsat 8/9 PC download for one plume.")
    parser.add_argument("--plume-id", required=True, help="Identifier used for local folders.")
    parser.add_argument("--lat", type=float, required=True, help="Plume latitude.")
    parser.add_argument("--lon", type=float, required=True, help="Plume longitude.")
    parser.add_argument(
        "--event-datetime",
        required=True,
        help="Event timestamp in ISO format (e.g., 2019-10-19T15:12:21Z).",
    )
    parser.add_argument("--offset", type=int, default=90, help="Days to subtract from the event.")
    parser.add_argument("--tolerance", type=int, default=SEARCH_TOLERANCE_DAYS, help="Search window +/- days.")
    parser.add_argument("--cloud", type=float, default=MAX_CLOUD_COVER, help="Maximum acceptable cloud cover.")
    parser.add_argument(
        "--bounds-size",
        type=float,
        default=0.01,
        help="Half-size of bounding box in degrees (default 0.01 => ~1.1 km).",
    )
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Directory for raw Landsat archives.")
    parser.add_argument(
        "--processed-dir",
        default=str(DEFAULT_PROCESSED_DIR),
        help="Directory for processed plume stacks.",
    )
    parser.add_argument("--download", action="store_true", help="Download and crop the chosen scene.")
    return parser.parse_args()


def to_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized).astimezone(timezone.utc)


def summarize_items(items) -> List[str]:
    rows = []
    for item in items:
        scene_id = get_normalized_scene_id(item)
        platform = (item.properties or {}).get("platform")
        dt = item.datetime
        cloud = parse_cloud_cover(item)
        rows.append(
            f"{scene_id or item.id} | platform={platform} | "
            f"time={dt} | cloud={cloud}"
        )
    return rows


def main():
    args = parse_args()
    event_dt = to_datetime(args.event_datetime)
    target_dt = event_dt - timedelta(days=args.offset)
    tolerance = timedelta(days=args.tolerance)
    window_start = target_dt - tolerance
    window_end = target_dt + tolerance
    plume_bounds = [
        args.lon - args.bounds_size,
        args.lat - args.bounds_size,
        args.lon + args.bounds_size,
        args.lat + args.bounds_size,
    ]

    client = Client.open(PLANETARY_COMPUTER_STAC_URL)
    print(
        f"Searching Landsat 8/9 scenes for plume {args.plume_id} "
        f"target={target_dt.isoformat()} (offset={args.offset} days)"
    )
    items = search_landsat_items(client, plume_bounds, window_start, window_end)
    print(f"Total STAC items returned: {len(items)}")
    if not items:
        return

    filtered = []
    for item in items:
        if not is_l89_item(item):
            continue
        cloud = parse_cloud_cover(item)
        if cloud is None or cloud > args.cloud:
            continue
        filtered.append(item)
    if not filtered:
        print("No L8/L9 scenes within cloud threshold.")
        return

    summaries = summarize_items(filtered)
    print("Filtered candidates:")
    for row in summaries:
        print("  -", row)

    selected = min(filtered, key=lambda it: abs((it.datetime - target_dt).total_seconds()))
    print("Selected scene:", get_normalized_scene_id(selected), "at", selected.datetime)

    if not args.download:
        print("Download flag not set; exiting after listing candidates.")
        return

    plume_dir = Path(args.processed_dir) / args.plume_id
    os.makedirs(plume_dir, exist_ok=True)
    scene_dir = ensure_scene_assets(selected, args.raw_dir)
    record = build_output_record(selected, scene_dir, str(plume_dir), plume_bounds, args.offset)
    if record is None:
        print("Failed to build output record.")
        return
    add_record_to_cache(args.offset, record)
    print("Downloaded and cropped scene:")
    for key, value in record.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
