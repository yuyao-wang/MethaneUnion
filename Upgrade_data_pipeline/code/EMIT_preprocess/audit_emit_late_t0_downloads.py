#!/usr/bin/env python3
"""Live-check late EMIT t0 granules without downloading image files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
DOWNLOADER_DIR = PIPELINE_ROOT / "code" / "sensor_data_download"
sys.path.insert(0, str(DOWNLOADER_DIR))
import download_emit_missing_6time as emit_download  # noqa: E402


DEFAULT_INPUT = (
    PIPELINE_ROOT
    / "csv"
    / "carbon_mapper_plumes_20160101_20260530_with_t0_flags.csv"
)
DEFAULT_OUTPUT_DIR = (
    PIPELINE_ROOT
    / "csv"
    / "emit_late_6time_audit"
    / "emit32_late_t0_download_audit"
)


def base_event_id(value: object) -> str:
    return re.sub(r"-[A-Za-z]+$", "", str(value).strip())


def query_row(
    row: dict[str, Any],
    *,
    short_name: str,
    window_hours: float,
    search_count: int,
    retries: int,
    check_six_time: bool,
    prev_search_back_days: int,
    offset_before_days: int,
    offset_after_days: int,
    year_offset_days: int,
) -> dict[str, Any]:
    expected = pd.Timestamp(row["emit_t0_time"])
    start = expected - pd.Timedelta(hours=window_hours)
    end = expected + pd.Timedelta(hours=window_hours)
    output = {
        "plume_id": row["plume_id"],
        "base_event_id": base_event_id(row["plume_id"]),
        "source_emit_has_t0": bool(row["source_emit_has_t0"]),
        "event_time": row["datetime"],
        "expected_t0_time": expected.isoformat(),
        "latitude": float(row["plume_latitude"]),
        "longitude": float(row["plume_longitude"]),
        "query_start": start.isoformat(),
        "query_end": end.isoformat(),
        "catalogue_results": 0,
        "found": False,
        "granule_id": "",
        "granule_time": "",
        "absolute_delta_minutes": None,
        "prev1_granule_id": "",
        "prev2_granule_id": "",
        "prev3_granule_id": "",
        "seasonal_granule_id": "",
        "year_granule_id": "",
        "all6_found": False,
        "distinct_overpasses": 0,
        "error": "",
    }
    try:
        results = emit_download.earthaccess_search_point(
            output["latitude"],
            output["longitude"],
            start,
            end,
            short_name,
            search_count,
            retries,
        )
        output["catalogue_results"] = len(results)
        granule = emit_download.select_from_results(results, expected)
        if granule is None:
            return output
        granule_id = emit_download.get_granule_id(granule) or ""
        granule_time = emit_download.get_granule_time(granule)
        output["found"] = bool(granule_id and granule_time is not None)
        output["granule_id"] = granule_id
        if granule_time is not None:
            output["granule_time"] = granule_time.isoformat()
            output["absolute_delta_minutes"] = abs(
                (granule_time - expected).total_seconds()
            ) / 60.0
        if not output["found"] or not check_six_time:
            return output

        prev_results = emit_download.earthaccess_search_point(
            output["latitude"],
            output["longitude"],
            granule_time - pd.Timedelta(days=prev_search_back_days),
            granule_time,
            short_name,
            search_count,
            retries,
        )
        selected_granules = [granule]
        for index, column in [
            (1, "prev1_granule_id"),
            (2, "prev2_granule_id"),
            (3, "prev3_granule_id"),
        ]:
            selected = emit_download.select_prev_from_results(
                prev_results, granule, index
            )
            if selected is not None:
                output[column] = (
                    emit_download.get_granule_id(selected) or ""
                )
                selected_granules.append(selected)

        for column, offset_days in [
            ("seasonal_granule_id", 90),
            ("year_granule_id", year_offset_days),
        ]:
            target = granule_time - pd.Timedelta(days=offset_days)
            offset_results = emit_download.earthaccess_search_point(
                output["latitude"],
                output["longitude"],
                target - pd.Timedelta(days=offset_before_days),
                target + pd.Timedelta(days=offset_after_days),
                short_name,
                search_count,
                retries,
            )
            selected = emit_download.select_from_results(
                offset_results, target
            )
            if selected is not None:
                output[column] = (
                    emit_download.get_granule_id(selected) or ""
                )
                selected_granules.append(selected)

        six_columns = [
            "granule_id",
            "prev1_granule_id",
            "prev2_granule_id",
            "prev3_granule_id",
            "seasonal_granule_id",
            "year_granule_id",
        ]
        output["all6_found"] = all(output[column] for column in six_columns)
        output["distinct_overpasses"] = len(
            {
                emit_download.emit_overpass_key(
                    emit_download.get_granule_id(item) or ""
                )
                for item in selected_granules
                if emit_download.get_granule_id(item)
            }
        )
    except Exception as exc:  # Keep the full audit running after one query fails.
        output["error"] = f"{type(exc).__name__}: {exc}"
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--start-exclusive",
        default="2025-11-12T23:59:59Z",
    )
    parser.add_argument("--end-inclusive", default="2026-05-22T23:59:59Z")
    parser.add_argument("--window-hours", type=float, default=24.0)
    parser.add_argument("--short-name", default="EMITL2ARFL")
    parser.add_argument("--search-count", type=int, default=200)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--sample-per-month", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--include-without-t0", action="store_true")
    parser.add_argument("--check-six-time", action="store_true")
    parser.add_argument("--prev-search-back-days", type=int, default=365)
    parser.add_argument("--offset-before-days", type=int, default=180)
    parser.add_argument("--offset-after-days", type=int, default=80)
    parser.add_argument("--year-offset-days", type=int, default=180)
    parser.add_argument(
        "--login",
        action="store_true",
        help="Authenticate with Earthdata first; metadata queries normally do not require it.",
    )
    args = parser.parse_args()

    columns = [
        "plume_id",
        "datetime",
        "emit_has_t0",
        "emit_t0_time",
        "plume_latitude",
        "plume_longitude",
    ]
    frame = pd.read_csv(args.input_csv, usecols=columns, low_memory=False)
    event_time = pd.to_datetime(frame["datetime"], utc=True, errors="raise")
    expected_t0 = pd.to_datetime(
        frame["emit_t0_time"], utc=True, errors="coerce"
    )
    has_t0 = frame["emit_has_t0"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
    in_window = event_time.gt(
        pd.Timestamp(args.start_exclusive)
    ) & event_time.le(pd.Timestamp(args.end_inclusive))
    if args.include_without_t0:
        selected = frame.loc[in_window & ~has_t0].copy()
        selected["emit_t0_time"] = event_time.loc[selected.index]
    else:
        selected = frame.loc[
            in_window & has_t0 & expected_t0.notna()
        ].copy()
        selected["emit_t0_time"] = expected_t0.loc[selected.index]
    selected["source_emit_has_t0"] = has_t0.loc[selected.index]
    selected = selected.sort_values(
        ["datetime", "plume_id"], kind="stable"
    )
    if args.sample_per_month:
        selected["_month"] = pd.to_datetime(
            selected["datetime"], utc=True
        ).dt.strftime("%Y-%m")
        selected = (
            selected.groupby("_month", group_keys=False)
            .apply(
                lambda group: group.sample(
                    n=min(args.sample_per_month, len(group)),
                    random_state=args.seed,
                )
            )
            .drop(columns="_month")
            .sort_values(["datetime", "plume_id"], kind="stable")
        )
    if args.sample and len(selected) > args.sample:
        selected = selected.sample(
            n=args.sample, random_state=args.seed
        ).sort_values(["datetime", "plume_id"], kind="stable")
    if args.limit:
        selected = selected.head(args.limit)

    if args.login:
        auth = emit_download.earthaccess.login()
        if auth is None:
            raise RuntimeError("earthaccess.login() failed")

    rows = selected.to_dict("records")
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                query_row,
                row,
                short_name=args.short_name,
                window_hours=args.window_hours,
                search_count=args.search_count,
                retries=args.retries,
                check_six_time=args.check_six_time,
                prev_search_back_days=args.prev_search_back_days,
                offset_before_days=args.offset_before_days,
                offset_after_days=args.offset_after_days,
                year_offset_days=args.year_offset_days,
            )
            for row in rows
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if (
                result["found"]
                or result["error"]
                or index % args.progress_every == 0
                or index == len(rows)
            ):
                print(
                    f"[{index}/{len(rows)}] found={result['found']} "
                    f"results={result['catalogue_results']} "
                    f"{result['plume_id']} {result['granule_id']} "
                    f"{result['error']}",
                    flush=True,
                )

    result_frame = pd.DataFrame(results).sort_values(
        ["event_time", "plume_id"], kind="stable"
    )
    found = result_frame["found"].astype(bool)
    all6_found = result_frame["all6_found"].astype(bool)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "late_emit_t0_live_query.csv"
    result_frame.to_csv(output_csv, index=False)
    summary = {
        "input_csv": str(Path(args.input_csv).resolve()),
        "selection": {
            "start_exclusive": args.start_exclusive,
            "end_inclusive": args.end_inclusive,
            "requires_emit_has_t0": not args.include_without_t0,
            "sample": args.sample,
            "sample_per_month": args.sample_per_month,
            "seed": args.seed,
        },
        "query": {
            "short_name": args.short_name,
            "window_hours_each_side": args.window_hours,
            "search_count": args.search_count,
            "ignore_old_search_cache": True,
            "earthdata_login_requested": bool(args.login),
            "check_six_time": bool(args.check_six_time),
            "downloads_performed": 0,
        },
        "rows_queried": int(len(result_frame)),
        "base_events_queried": int(
            result_frame["base_event_id"].nunique()
        ),
        "rows_with_granule": int(found.sum()),
        "rows_without_granule": int((~found).sum()),
        "rows_with_all6_granules": int(all6_found.sum()),
        "rows_with_six_distinct_overpasses": int(
            result_frame["distinct_overpasses"].eq(6).sum()
        ),
        "rows_with_query_error": int(result_frame["error"].ne("").sum()),
        "unique_granules_found": int(
            result_frame.loc[found, "granule_id"].nunique()
        ),
        "output_csv": str(output_csv.resolve()),
    }
    output_json = output_dir / "summary.json"
    output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["rows_with_query_error"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
