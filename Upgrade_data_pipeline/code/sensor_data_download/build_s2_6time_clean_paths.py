#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


TIMEPOINTS = ["t0", "prev1", "prev2", "prev3", "seasonal", "year"]
PREV_TIMEPOINTS = {"prev1", "prev2", "prev3"}
CM_COLUMNS = ["plume_tif", "plume_png", "con_tif", "rgb_tif", "rgb_png"]


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text


def has_value(value: Any) -> bool:
    return bool(clean(value))


class ExistsCache:
    def __init__(self) -> None:
        self.cache: dict[str, bool] = {}

    def existing_path(self, value: Any) -> str:
        text = clean(value)
        if not text:
            return ""
        cached = self.cache.get(text)
        if cached is not None:
            return text if cached else ""
        try:
            path = Path(text)
            ok = path.exists() and path.stat().st_size > 0
        except OSError:
            ok = False
        self.cache[text] = ok
        return text if ok else ""


def first_existing(exists: ExistsCache, row: pd.Series, columns: list[str]) -> tuple[str, str]:
    for col in columns:
        if col not in row.index:
            continue
        path = exists.existing_path(row.get(col, ""))
        if path:
            return path, col
    return "", ""


def safe_component(value: Any) -> str:
    text = clean(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:180] if text else "missing_product"


def product_target_path(raw_root: str, product_id: Any, product_name: Any, plume_id: str, timepoint: str) -> str:
    product_key = safe_component(product_id) if has_value(product_id) else safe_component(product_name)
    return str(Path(raw_root) / "S2" / "product_crops" / product_key / plume_id / f"{timepoint}.tif")


def empty_row(plume_id: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "plume_id": plume_id,
        "event_time": "",
        "plume_latitude": "",
        "plume_longitude": "",
        "plume_bounds": "",
        "sensor": "S2",
    }
    for tp in TIMEPOINTS:
        row.update(
            {
                f"{tp}_image_time": "",
                f"{tp}_product_id": "",
                f"{tp}_product_name": "",
                f"{tp}_overpass_key": "",
                f"{tp}_raw_path": "",
                f"{tp}_512_path": "",
                f"{tp}_path_source": "",
                f"{tp}_local_status": "",
                f"{tp}_download_needed": 0,
                f"{tp}_download_target_raw_path": "",
                f"{tp}_expected_status": "",
                f"{tp}_matched_old_timepoint": "",
                f"{tp}_selection_source": "",
            }
        )
    for col in CM_COLUMNS:
        row[col] = ""
    return row


def fill_base_from_row(out: dict[str, Any], src: pd.Series) -> None:
    for col in ["event_time", "plume_latitude", "plume_longitude", "plume_bounds"]:
        if not clean(out.get(col)) and col in src.index:
            out[col] = clean(src.get(col, ""))


def set_tp_metadata(out: dict[str, Any], tp: str, source: pd.Series, prefix: str = "") -> None:
    mapping = {
        "image_time": [f"{prefix}image_time", "image_time"],
        "product_id": [f"{prefix}product_id", "product_id"],
        "product_name": [f"{prefix}product_name", "product_name"],
        "overpass_key": [f"{prefix}overpass_key", "overpass_key"],
        "selection_source": [f"{prefix}selection_source", "selection_source"],
    }
    for dst, candidates in mapping.items():
        key = f"{tp}_{dst}"
        if clean(out.get(key)):
            continue
        for col in candidates:
            if col in source.index and has_value(source.get(col, "")):
                out[key] = clean(source.get(col, ""))
                break


def product_matches(row: pd.Series, product_id: Any, product_name: Any) -> bool:
    pid = clean(product_id)
    pname = clean(product_name)
    candidates = [
        clean(row.get("product_id", "")),
        clean(row.get("product_name", "")),
    ]
    return bool((pid and pid in candidates) or (pname and pname in candidates))


def build_downloaded_raw_index(log: pd.DataFrame, exists: ExistsCache) -> dict[tuple[str, str], list[dict[str, str]]]:
    out: dict[tuple[str, str], list[dict[str, str]]] = {}
    if log.empty:
        return out
    log = log[log["status"].astype(str).eq("downloaded")].copy()
    for _, row in log.iterrows():
        plume_id = clean(row.get("plume_id", ""))
        raw_path = exists.existing_path(row.get("raw_path", ""))
        if not plume_id or not raw_path:
            continue
        record = {
            "raw_path": raw_path,
            "old_timepoint": clean(row.get("timepoint", "")),
            "product_id": clean(row.get("product_id", "")),
            "product_name": clean(row.get("product_name", "")),
            "image_time": clean(row.get("acquisition_time", "")),
            "source": "s2_download_manifest:downloaded",
        }
        for key in [record["product_id"], record["product_name"]]:
            if key:
                out.setdefault((plume_id, key), []).append(record)
    return out


def find_downloaded_raw(
    raw_index: dict[tuple[str, str], list[dict[str, str]]],
    plume_id: str,
    product_id: Any,
    product_name: Any,
) -> dict[str, str] | None:
    matches: list[dict[str, str]] = []
    for key in [clean(product_id), clean(product_name)]:
        if key:
            matches.extend(raw_index.get((plume_id, key), []))
    if not matches:
        return None
    return sorted(matches, key=lambda r: (r["old_timepoint"], r["raw_path"]))[0]


def fill_actual_paths_from_master(
    out: dict[str, Any],
    tp: str,
    row: pd.Series,
    exists: ExistsCache,
) -> None:
    raw_path, raw_source = first_existing(exists, row, ["downloaded_path", "existing_raw_path"])
    if raw_path and not clean(out.get(f"{tp}_raw_path")):
        out[f"{tp}_raw_path"] = raw_path
        out[f"{tp}_path_source"] = f"master:{raw_source}"
    tif512, tif512_source = first_existing(exists, row, ["processed_path", "existing_512_path", "target_512_path"])
    if tif512 and not clean(out.get(f"{tp}_512_path")):
        out[f"{tp}_512_path"] = tif512
        current = clean(out.get(f"{tp}_path_source"))
        out[f"{tp}_path_source"] = ";".join([v for v in [current, f"master:{tif512_source}"] if v])


def fill_512_from_wide_if_matching(
    out: dict[str, Any],
    tp: str,
    row: pd.Series,
    exists: ExistsCache,
    require_product_match: bool,
) -> None:
    path = exists.existing_path(row.get(f"{tp}_512_path", ""))
    if not path:
        return
    if require_product_match:
        if not product_matches(row, out.get(f"{tp}_product_id", ""), out.get(f"{tp}_product_name", "")):
            return
    if not clean(out.get(f"{tp}_512_path")):
        out[f"{tp}_512_path"] = path
        current = clean(out.get(f"{tp}_path_source"))
        out[f"{tp}_path_source"] = ";".join([v for v in [current, "s2_6time_complete_paths:512"] if v])


def finalize_timepoint(out: dict[str, Any], tp: str, raw_root: str) -> None:
    raw = clean(out.get(f"{tp}_raw_path"))
    tif512 = clean(out.get(f"{tp}_512_path"))
    expected_status = clean(out.get(f"{tp}_expected_status"))
    product_id = clean(out.get(f"{tp}_product_id"))
    product_name = clean(out.get(f"{tp}_product_name"))
    if raw or tif512:
        out[f"{tp}_local_status"] = "available"
        out[f"{tp}_download_needed"] = 0
        out[f"{tp}_download_target_raw_path"] = ""
        return
    if expected_status == "no_catalogue_overpass":
        out[f"{tp}_local_status"] = "no_catalogue_overpass"
        out[f"{tp}_download_needed"] = 0
        out[f"{tp}_download_target_raw_path"] = ""
        return
    if product_id or product_name:
        out[f"{tp}_local_status"] = "missing_local_file"
        out[f"{tp}_download_needed"] = 1
        out[f"{tp}_download_target_raw_path"] = product_target_path(raw_root, product_id, product_name, out["plume_id"], tp)
        return
    out[f"{tp}_local_status"] = "missing_product_metadata"
    out[f"{tp}_download_needed"] = 0
    out[f"{tp}_download_target_raw_path"] = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one clean S2 six-time path/control table from expected catalogue prevs and actual local files.")
    parser.add_argument("--audit-csv", default="Upgrade_data_pipeline/csv/s2_emit_s5p_expected_prev_catalogue_audit.csv")
    parser.add_argument("--master-csv", default="Upgrade_data_pipeline/csv/multisensor_6time_download_manifest.csv")
    parser.add_argument("--s2-download-manifest", default="Upgrade_data_pipeline/csv/s2_download_manifest.csv")
    parser.add_argument("--existing-wide-csv", default="Upgrade_data_pipeline/csv/s2_6time_complete_paths.csv")
    parser.add_argument("--out-csv", default="Upgrade_data_pipeline/csv/s2_6time_clean_paths.csv")
    parser.add_argument("--raw-root", default="/mnt/engg-niulab/yuyao/sensors_raw_data")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exists = ExistsCache()
    audit = pd.read_csv(args.audit_csv, low_memory=False)
    s2_audit = audit[audit["sensor"].astype(str).eq("S2")].copy()
    s2_audit = s2_audit.drop_duplicates(["plume_id", "expected_timepoint"], keep="last")

    master = pd.read_csv(args.master_csv, low_memory=False)
    master = master[master["sensor"].astype(str).eq("S2")].copy()
    master_by_key = {
        (clean(row.get("plume_id", "")), clean(row.get("timepoint", ""))): row
        for _, row in master.iterrows()
    }

    s2_log = pd.read_csv(args.s2_download_manifest, low_memory=False) if Path(args.s2_download_manifest).exists() else pd.DataFrame()
    raw_index = build_downloaded_raw_index(s2_log, exists)

    wide_by_plume: dict[str, pd.Series] = {}
    if Path(args.existing_wide_csv).exists():
        wide = pd.read_csv(args.existing_wide_csv, low_memory=False)
        wide_by_plume = {clean(row.get("plume_id", "")): row for _, row in wide.iterrows()}

    rows: dict[str, dict[str, Any]] = {}
    for plume_id, group in s2_audit.groupby("plume_id", sort=False):
        plume_id = clean(plume_id)
        if not plume_id:
            continue
        out = rows.setdefault(plume_id, empty_row(plume_id))
        first = group.iloc[0]
        fill_base_from_row(out, first)
        out["t0_product_id"] = clean(first.get("t0_product_id", ""))
        out["t0_product_name"] = clean(first.get("t0_product_name", ""))
        out["t0_image_time"] = clean(first.get("t0_image_time", ""))
        out["t0_overpass_key"] = clean(first.get("t0_overpass_key", ""))
        out["t0_expected_status"] = "expected"
        t0_match = find_downloaded_raw(raw_index, plume_id, out["t0_product_id"], out["t0_product_name"])
        if t0_match is not None:
            out["t0_raw_path"] = t0_match["raw_path"]
            out["t0_path_source"] = t0_match["source"]
            out["t0_matched_old_timepoint"] = t0_match["old_timepoint"]

        for _, audit_row in group.iterrows():
            tp = clean(audit_row.get("expected_timepoint", ""))
            if tp not in PREV_TIMEPOINTS:
                continue
            fill_base_from_row(out, audit_row)
            out[f"{tp}_expected_status"] = clean(audit_row.get("expected_status", ""))
            out[f"{tp}_product_id"] = clean(audit_row.get("expected_product_id", ""))
            out[f"{tp}_product_name"] = clean(audit_row.get("expected_product_name", ""))
            out[f"{tp}_image_time"] = clean(audit_row.get("expected_image_time", ""))
            out[f"{tp}_overpass_key"] = clean(audit_row.get("expected_overpass_key", ""))
            match = find_downloaded_raw(raw_index, plume_id, out[f"{tp}_product_id"], out[f"{tp}_product_name"])
            if match is not None:
                out[f"{tp}_raw_path"] = match["raw_path"]
                out[f"{tp}_path_source"] = match["source"]
                out[f"{tp}_matched_old_timepoint"] = match["old_timepoint"]

    for (plume_id, tp), master_row in master_by_key.items():
        if plume_id not in rows or tp not in TIMEPOINTS:
            continue
        out = rows[plume_id]
        fill_base_from_row(out, master_row)
        if tp not in PREV_TIMEPOINTS:
            set_tp_metadata(out, tp, master_row)
            out[f"{tp}_expected_status"] = "expected" if clean(out.get(f"{tp}_product_id")) or clean(out.get(f"{tp}_product_name")) else clean(out.get(f"{tp}_expected_status"))
            fill_actual_paths_from_master(out, tp, master_row, exists)
        else:
            # For prevs, metadata must stay catalogue-ranked and raw path identity
            # must come from the trusted downloaded raw ledger only.
            continue

    for plume_id, wide_row in wide_by_plume.items():
        if plume_id not in rows:
            continue
        out = rows[plume_id]
        for col in CM_COLUMNS:
            if not clean(out.get(col)) and col in wide_row.index:
                out[col] = clean(wide_row.get(col, ""))
        for tp in TIMEPOINTS:
            if tp not in PREV_TIMEPOINTS:
                fill_512_from_wide_if_matching(out, tp, wide_row, exists, require_product_match=False)

    for out in rows.values():
        for tp in TIMEPOINTS:
            finalize_timepoint(out, tp, args.raw_root)

    fieldnames = list(empty_row("").keys())
    output = pd.DataFrame([rows[k] for k in sorted(rows)], columns=fieldnames)
    tmp = Path(args.out_csv).with_suffix(Path(args.out_csv).suffix + f".tmp.{os.getpid()}")
    output.to_csv(tmp, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(tmp, args.out_csv)

    print(f"wrote {args.out_csv}: rows={len(output)}")
    for tp in TIMEPOINTS:
        counts = output[f"{tp}_local_status"].value_counts(dropna=False).to_dict()
        need = int(output[f"{tp}_download_needed"].fillna(0).astype(int).sum())
        print(f"{tp}: statuses={counts} download_needed={need}")
    prev_need = output[[f"{tp}_download_needed" for tp in PREV_TIMEPOINTS]].fillna(0).astype(int).sum().sum()
    print(f"prev_download_needed_total={int(prev_need)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
