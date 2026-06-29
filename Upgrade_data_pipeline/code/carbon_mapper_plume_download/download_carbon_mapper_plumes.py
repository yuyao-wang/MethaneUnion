#!/usr/bin/env python3
from __future__ import annotations

import argparse
import calendar
import io
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

API_URL = "https://api.carbonmapper.org/api/v1/catalog/plume-csv"


def parse_utc_date(value: str, end_of_day: bool = False) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    if end_of_day and dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0:
        dt = dt.replace(hour=23, minute=59, second=59)
    return dt


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def month_windows(start: datetime, end: datetime) -> Iterable[tuple[datetime, datetime]]:
    cur = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while cur <= end:
        last_day = calendar.monthrange(cur.year, cur.month)[1]
        month_end = cur.replace(day=last_day, hour=23, minute=59, second=59)
        yield max(cur, start), min(month_end, end)
        nxt = cur.replace(day=28) + timedelta(days=4)
        cur = nxt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def fetch_window(start: datetime, end: datetime, limit: int, sleep_s: float, max_retries: int) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    offset = 0
    datetime_range = f"{iso_z(start)}/{iso_z(end)}"
    while True:
        params = {
            "plume_gas": "CH4",
            "sort": "desc",
            "limit": str(limit),
            "offset": str(offset),
            "datetime": datetime_range,
        }
        for attempt in range(max_retries):
            resp = requests.get(API_URL, params=params, timeout=120)
            if resp.status_code == 200:
                break
            wait = min(120, 5 * (attempt + 1))
            print(f"[warn] HTTP {resp.status_code} for {datetime_range} offset={offset}; retry in {wait}s", flush=True)
            time.sleep(wait)
        else:
            resp.raise_for_status()

        text = resp.text.strip()
        if not text:
            break
        df = pd.read_csv(io.StringIO(text))
        if df.empty:
            break
        frames.append(df)
        print(f"[fetch] {datetime_range} offset={offset} rows={len(df)}", flush=True)
        if len(df) < limit:
            break
        offset += limit
        if sleep_s > 0:
            time.sleep(sleep_s)
    return frames


def main() -> int:
    ap = argparse.ArgumentParser(description="Download Carbon Mapper CH4 plume CSV catalog and merge by plume_id.")
    ap.add_argument("--start", default="2016-01-01", help="UTC start date/time, inclusive")
    ap.add_argument("--end", default="2026-05-30", help="UTC end date/time, inclusive; date-only means end of that day")
    ap.add_argument("--out", default="data_preprocess/carbon_mapper_plumes_20160101_20260530.csv")
    ap.add_argument("--raw-out", default="", help="Optional path for non-deduplicated concatenated rows")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--max-retries", type=int, default=5)
    args = ap.parse_args()

    start = parse_utc_date(args.start)
    end = parse_utc_date(args.end, end_of_day=True)
    if end < start:
        raise ValueError("end must be after start")

    all_frames: list[pd.DataFrame] = []
    for win_start, win_end in month_windows(start, end):
        all_frames.extend(fetch_window(win_start, win_end, args.limit, args.sleep, args.max_retries))

    if not all_frames:
        raise RuntimeError("No rows returned from Carbon Mapper API")

    raw = pd.concat(all_frames, ignore_index=True)
    if "plume_id" not in raw.columns:
        raise RuntimeError("Carbon Mapper response did not contain plume_id column")

    merged = raw.drop_duplicates(subset="plume_id").copy()
    if "datetime" in merged.columns:
        merged["_sort_datetime"] = pd.to_datetime(merged["datetime"], errors="coerce", utc=True)
        merged = merged.sort_values(["_sort_datetime", "plume_id"], na_position="last").drop(columns=["_sort_datetime"])
    else:
        merged = merged.sort_values("plume_id")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)

    if args.raw_out:
        raw_out = Path(args.raw_out)
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(raw_out, index=False)

    print(f"[done] raw_rows={len(raw)} unique_plumes={len(merged)} out={out}", flush=True)
    if "datetime" in merged.columns:
        t = pd.to_datetime(merged["datetime"], errors="coerce", utc=True)
        print(f"[done] datetime_min={t.min()} datetime_max={t.max()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
