#!/usr/bin/env python3
"""Download late EMIT six-time data resolved by the live catalogue audit."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any

import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
SENSOR_DOWNLOAD_DIR = PIPELINE_ROOT / "code" / "sensor_data_download"
sys.path.insert(0, str(SENSOR_DOWNLOAD_DIR))
import download_emit_missing_6time as emit_download  # noqa: E402


AUDIT_ROOT = PIPELINE_ROOT / "csv" / "emit_late_6time_audit"
DEFAULT_AUDITS = [
    AUDIT_ROOT
    / "emit32_late_t0_download_audit"
    / "late_emit_t0_live_query.csv",
    AUDIT_ROOT
    / "emit32_late_t0_false_negative_six_time_audit"
    / "late_emit_t0_live_query.csv",
]
DEFAULT_PLUME_CSV = (
    PIPELINE_ROOT
    / "csv"
    / "carbon_mapper_plumes_20160101_20260530_with_t0_flags.csv"
)
DEFAULT_OUTPUT_DIR = PIPELINE_ROOT / "csv" / "emit_late_6time_download"
DEFAULT_RAW_ROOT = Path("/mnt/engg-niulab/yuyao/sensors_raw_data")
DEFAULT_SCRATCH_ROOT = Path("/diniuvol/yuyao/emit_download_scratch")

TIMEPOINT_COLUMNS = {
    "t0": "granule_id",
    "prev1": "prev1_granule_id",
    "prev2": "prev2_granule_id",
    "prev3": "prev3_granule_id",
    "seasonal": "seasonal_granule_id",
    "year": "year_granule_id",
}
MANIFEST_FIELDS = [
    "plume_id",
    "sensor",
    "timepoint",
    "action",
    "event_time",
    "plume_latitude",
    "plume_longitude",
    "sensor_has_t0",
    "t0_available_time",
    "year_offset_days",
    "target_raw_dir",
    "resolved_granule_id",
]
PLUME_METADATA_COLUMNS = [
    "plume_id",
    "event_group_id",
    "datetime",
    "plume_latitude",
    "plume_longitude",
    "plume_bounds",
    "country",
    "region",
    "place",
    "ipcc_sector",
    "gas",
    "instrument",
    "platform",
    "provider",
    "plume_tif",
    "plume_png",
    "con_tif",
    "rgb_tif",
    "rgb_png",
]


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )


def read_resolved_rows(
    audit_paths: list[Path],
    *,
    require_six_distinct: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    required = {
        "plume_id",
        "event_time",
        "latitude",
        "longitude",
        "granule_time",
        "all6_found",
        "distinct_overpasses",
        *TIMEPOINT_COLUMNS.values(),
    }
    for path in audit_paths:
        if not path.exists():
            raise FileNotFoundError(f"audit CSV not found: {path}")
        frame = pd.read_csv(path, low_memory=False)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        frame["audit_source"] = str(path.resolve())
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    selected = combined.loc[truthy(combined["all6_found"])].copy()
    if require_six_distinct:
        selected = selected.loc[
            pd.to_numeric(
                selected["distinct_overpasses"], errors="coerce"
            ).eq(6)
        ].copy()
    selected = (
        selected.drop_duplicates("plume_id", keep="last")
        .sort_values(["event_time", "plume_id"], kind="stable")
        .reset_index(drop=True)
    )
    if selected.empty:
        raise RuntimeError("no complete six-time rows found in audit CSVs")

    long_rows: list[dict[str, Any]] = []
    for row in selected.to_dict("records"):
        for timepoint, column in TIMEPOINT_COLUMNS.items():
            granule_id = str(row.get(column, "")).strip()
            if not granule_id or granule_id.lower() == "nan":
                raise RuntimeError(
                    f"missing {timepoint} granule for {row['plume_id']}"
                )
            long_rows.append(
                {
                    "plume_id": str(row["plume_id"]).strip(),
                    "timepoint": timepoint,
                    "granule_id": granule_id,
                    "event_time": row["event_time"],
                    "t0_available_time": row["granule_time"],
                    "plume_latitude": float(row["latitude"]),
                    "plume_longitude": float(row["longitude"]),
                    "audit_source": row["audit_source"],
                }
            )
    resolved = pd.DataFrame(long_rows)
    conflicts = (
        resolved.groupby(["plume_id", "timepoint"])["granule_id"]
        .nunique()
        .gt(1)
    )
    if conflicts.any():
        raise RuntimeError(
            f"conflicting resolved IDs: {conflicts[conflicts].index[:5]}"
        )
    return selected, resolved


def expected_manifest(
    resolved: pd.DataFrame,
    *,
    raw_root: Path,
) -> pd.DataFrame:
    rows = []
    for row in resolved.to_dict("records"):
        plume_id = str(row["plume_id"])
        timepoint = str(row["timepoint"])
        rows.append(
            {
                "plume_id": plume_id,
                "sensor": "EMIT",
                "timepoint": timepoint,
                "action": "download",
                "event_time": row["event_time"],
                "plume_latitude": row["plume_latitude"],
                "plume_longitude": row["plume_longitude"],
                "sensor_has_t0": True,
                "t0_available_time": row["t0_available_time"],
                "year_offset_days": 180,
                "target_raw_dir": str(
                    raw_root / "EMIT" / timepoint / plume_id
                ),
                "resolved_granule_id": row["granule_id"],
            }
        )
    return pd.DataFrame(rows, columns=MANIFEST_FIELDS)


def prepare_manifest(
    resolved: pd.DataFrame,
    manifest_path: Path,
    *,
    raw_root: Path,
    rebuild: bool,
) -> pd.DataFrame:
    expected = expected_manifest(resolved, raw_root=raw_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists() or rebuild:
        expected.to_csv(manifest_path, index=False)
        emit_download.ensure_manifest_columns(manifest_path)
        return pd.read_csv(manifest_path, low_memory=False)

    existing = pd.read_csv(manifest_path, low_memory=False)
    required = {"plume_id", "timepoint", "resolved_granule_id"}
    missing = sorted(required - set(existing.columns))
    if missing:
        raise ValueError(
            f"work manifest is missing {missing}; use --rebuild-manifest"
        )
    keys = ["plume_id", "timepoint", "resolved_granule_id"]
    expected_keys = {
        tuple(str(value) for value in row)
        for row in expected[keys].itertuples(index=False, name=None)
    }
    existing_keys = {
        tuple(str(value) for value in row)
        for row in existing[keys].itertuples(index=False, name=None)
    }
    if expected_keys != existing_keys:
        raise RuntimeError(
            "audit results differ from the work manifest; "
            "use --rebuild-manifest"
        )
    return existing


def validate_scratch_space(
    path: Path,
    minimum_free_gb: float,
    *,
    create: bool,
) -> float:
    if create:
        path.mkdir(parents=True, exist_ok=True)
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_gib = shutil.disk_usage(probe).free / (1024**3)
    if free_gib < minimum_free_gb:
        raise RuntimeError(
            f"scratch free space is {free_gib:.1f} GiB, below "
            f"--min-scratch-free-gb={minimum_free_gb:.1f}"
        )
    return free_gib


def cached_granules(cache_root: Path, granule_ids: set[str]) -> set[str]:
    existing: set[str] = set()
    if not cache_root.exists():
        return existing
    for child in cache_root.iterdir():
        if child.name not in granule_ids or not child.is_dir():
            continue
        path = child / "emit_ch4_32.npz"
        if path.exists() and path.stat().st_size > 0:
            existing.add(child.name)
    return existing


def run_download(
    args: argparse.Namespace,
    manifest_path: Path,
    resolved_map: dict[tuple[str, str], str],
) -> tuple[int, int]:
    download_args = argparse.Namespace(
        manifest=str(manifest_path),
        legacy_emit_csv="",
        out_csv=str(args.output_dir / "emit_download_manifest.csv"),
        search_cache_csv=str(
            args.output_dir / "direct_id_search_cache.csv"
        ),
        raw_root=str(args.raw_root),
        cache_dir=str(args.cache_dir) if args.cache_dir else "",
        scratch_root=str(args.scratch_root),
        timepoints=",".join(TIMEPOINT_COLUMNS),
        short_name=args.short_name,
        workers=args.workers,
        download_threads=args.download_threads,
        local_io_workers=args.local_io_workers,
        limit=0,
        search_count=200,
        search_retries=args.search_retries,
        download_retries=args.download_retries,
        prev_search_back_days=365,
        offset_before_days=180,
        offset_after_days=80,
        year_offset_days=180,
        overwrite=args.overwrite,
        resume=True,
        ignore_search_cache=False,
        copy_from_cache=args.copy_from_cache,
        delete_nc_after_npz=args.delete_nc_after_npz,
        cleanup_scratch=True,
        no_master_update=False,
        download_all_links=False,
    )

    emit_download.local_io_semaphore = BoundedSemaphore(
        args.local_io_workers
    )
    emit_download.ensure_manifest_columns(manifest_path)
    rows = emit_download.load_work_rows(download_args)
    completed_records = {
        key: emit_download.master_record_to_emit(row)
        for key, row in emit_download.load_master_completed_records(
            manifest_path, "EMIT"
        ).items()
    }
    completed = set(completed_records)
    search_cache = emit_download.SearchCache(
        Path(download_args.search_cache_csv)
    )
    object_cache = emit_download.GranuleObjectCache(
        args.short_name, args.search_retries
    )
    original_find = emit_download.find_granule

    def find_resolved(
        row: pd.Series,
        timepoint: str,
        inner_args: argparse.Namespace,
        legacy: dict[str, dict[str, Any]],
        cache: Any,
        objects: Any,
        t0_granule: Any = None,
    ) -> tuple[Any, str | None, str]:
        key = (str(row["plume_id"]).strip(), str(timepoint).strip())
        granule_id = resolved_map.get(key)
        if granule_id:
            return emit_download.cache_direct_id_search(
                row,
                timepoint,
                granule_id,
                inner_args,
                cache,
                objects,
                "late_live_audit_direct_id",
            )
        return original_find(
            row,
            timepoint,
            inner_args,
            legacy,
            cache,
            objects,
            t0_granule,
        )

    emit_download.find_granule = find_resolved
    groups = [group.copy() for _, group in rows.groupby("plume_id", sort=False)]
    print(
        f"[Download] pending_rows={len(rows)} "
        f"pending_plumes={len(groups)} completed_rows={len(completed)}",
        flush=True,
    )

    all_records: list[dict[str, Any]] = []
    failures = 0
    completed_rows = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                emit_download.process_plume_group,
                group,
                download_args,
                {},
                completed,
                completed_records,
                search_cache,
                object_cache,
            ): group
            for group in groups
        }
        for future in as_completed(futures):
            group = futures[future]
            try:
                records = future.result()
            except Exception as exc:
                failures += 1
                print(
                    f"[Error] plume_id={group.iloc[0]['plume_id']} "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
            all_records.extend(records)
            for record in records:
                completed_rows += 1
                emit_download.append_rows(
                    Path(download_args.out_csv),
                    [record],
                    emit_download.OUT_FIELDS,
                )
                print(
                    f"[{completed_rows}/{len(rows)}] "
                    f"{record['status']} {record['plume_id']} "
                    f"{record['timepoint']} "
                    f"{record.get('granule_id', '')}",
                    flush=True,
                )

    if all_records:
        changed = emit_download.update_master_from_records(
            manifest_path,
            "EMIT",
            all_records,
            emit_download.record_to_master_update,
            source_log=download_args.out_csv,
        )
        print(f"[Manifest] updated_rows={changed}", flush=True)
    return failures, len(all_records)


def build_preprocess_input(
    manifest_path: Path,
    plume_csv: Path,
    output_path: Path,
) -> int:
    manifest = pd.read_csv(manifest_path, low_memory=False)
    available_columns = pd.read_csv(plume_csv, nrows=0).columns
    metadata_columns = [
        column
        for column in PLUME_METADATA_COLUMNS
        if column in available_columns
    ]
    metadata = pd.read_csv(
        plume_csv,
        usecols=metadata_columns,
        low_memory=False,
    ).drop_duplicates("plume_id", keep="last")
    metadata = metadata.rename(columns={"datetime": "event_time"})
    metadata = metadata.set_index("plume_id")

    validation_cache: dict[str, bool] = {}
    rows = []
    for plume_id, group in manifest.groupby("plume_id", sort=False):
        result: dict[str, Any] = {"plume_id": plume_id, "sensor": "EMIT"}
        if plume_id in metadata.index:
            result.update(metadata.loc[plume_id].to_dict())
        first = group.iloc[0]
        result["event_time"] = result.get(
            "event_time", first.get("event_time", "")
        )
        result["plume_latitude"] = result.get(
            "plume_latitude", first.get("plume_latitude", "")
        )
        result["plume_longitude"] = result.get(
            "plume_longitude", first.get("plume_longitude", "")
        )

        all6 = True
        for timepoint in TIMEPOINT_COLUMNS:
            match = group.loc[
                group["timepoint"].astype(str).eq(timepoint)
            ]
            if match.empty:
                all6 = False
                result[f"{timepoint}_npz_path"] = ""
                continue
            item = match.iloc[-1]
            path = str(item.get("downloaded_path", "")).strip()
            if path not in validation_cache:
                validation_cache[path] = (
                    emit_download.validate_npz(Path(path))[0]
                    if path and Path(path).exists()
                    else False
                )
            if validation_cache[path]:
                result[f"{timepoint}_npz_path"] = path
                result[f"{timepoint}_product_id"] = str(
                    item.get("product_id", "")
                    or item.get("resolved_granule_id", "")
                )
                result[f"{timepoint}_image_time"] = str(
                    item.get("image_time", "")
                )
            else:
                all6 = False
                result[f"{timepoint}_npz_path"] = ""
        result["has_t0_any"] = int(bool(result.get("t0_npz_path")))
        result["has_all6_npz"] = int(all6)
        rows.append(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    return int(frame["has_all6_npz"].sum()) if len(frame) else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-csv",
        action="append",
        dest="audit_csvs",
        help="Repeat for each live-audit CSV; defaults to both final audits.",
    )
    parser.add_argument(
        "--plume-csv",
        type=Path,
        default=DEFAULT_PLUME_CSV,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=DEFAULT_SCRATCH_ROOT,
    )
    parser.add_argument("--short-name", default="EMITL2ARFL")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--download-threads", type=int, default=4)
    parser.add_argument("--local-io-workers", type=int, default=2)
    parser.add_argument("--search-retries", type=int, default=5)
    parser.add_argument("--download-retries", type=int, default=5)
    parser.add_argument("--min-scratch-free-gb", type=float, default=200.0)
    parser.add_argument("--strict-six-distinct", action="store_true")
    parser.add_argument("--delete-nc-after-npz", action="store_true")
    parser.add_argument("--copy-from-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument(
        "--no-login",
        dest="login",
        action="store_false",
        help="Skip Earthdata login. Downloads normally require authentication.",
    )
    parser.set_defaults(login=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.local_io_workers < 1:
        raise ValueError("worker counts must be >= 1")
    args.output_dir = args.output_dir.resolve()
    args.raw_root = args.raw_root.resolve()
    args.scratch_root = args.scratch_root.resolve()
    args.plume_csv = args.plume_csv.resolve()
    if args.cache_dir:
        args.cache_dir = args.cache_dir.resolve()
    audit_paths = [
        Path(path).resolve()
        for path in (args.audit_csvs or DEFAULT_AUDITS)
    ]

    samples, resolved = read_resolved_rows(
        audit_paths,
        require_six_distinct=args.strict_six_distinct,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = args.output_dir / "resolved_six_time_granules.csv"
    resolved.to_csv(resolved_path, index=False)
    manifest_path = args.output_dir / "download_work_manifest.csv"
    manifest = prepare_manifest(
        resolved,
        manifest_path,
        raw_root=args.raw_root,
        rebuild=args.rebuild_manifest,
    )

    unique_granules = set(resolved["granule_id"].astype(str))
    cache_root = (
        args.cache_dir
        if args.cache_dir
        else args.raw_root / "EMIT" / "raw_granules"
    )
    existing = cached_granules(cache_root, unique_granules)
    scratch_free_gib = validate_scratch_space(
        args.scratch_root,
        args.min_scratch_free_gb,
        create=not args.dry_run,
    )
    summary = {
        "audit_csvs": [str(path) for path in audit_paths],
        "samples": len(samples),
        "manifest_rows": len(manifest),
        "unique_granules": len(unique_granules),
        "existing_cached_granules": len(existing),
        "missing_unique_granules": len(unique_granules - existing),
        "strict_six_distinct": args.strict_six_distinct,
        "raw_root": str(args.raw_root),
        "cache_root": str(cache_root),
        "scratch_root": str(args.scratch_root),
        "scratch_free_gib": round(scratch_free_gib, 2),
        "delete_nc_after_npz": args.delete_nc_after_npz,
        "work_manifest": str(manifest_path),
        "resolved_csv": str(resolved_path),
    }
    (args.output_dir / "download_plan.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return 0

    if args.login:
        auth = emit_download.earthaccess.login()
        if auth is None:
            raise RuntimeError("earthaccess.login() failed")

    resolved_map = {
        (str(row.plume_id), str(row.timepoint)): str(row.granule_id)
        for row in resolved.itertuples(index=False)
    }
    failed_groups, record_count = run_download(
        args, manifest_path, resolved_map
    )
    preprocess_csv = args.output_dir / "late_emit_6time_npz_paths.csv"
    complete_count = build_preprocess_input(
        manifest_path,
        args.plume_csv,
        preprocess_csv,
    )
    print(
        f"[Complete] records={record_count} failed_groups={failed_groups} "
        f"complete_samples={complete_count} preprocess_csv={preprocess_csv}",
        flush=True,
    )
    return 1 if failed_groups else 0


if __name__ == "__main__":
    raise SystemExit(main())
