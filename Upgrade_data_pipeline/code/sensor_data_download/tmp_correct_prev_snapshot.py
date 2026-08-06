import re
from pathlib import Path

import pandas as pd


CSV = Path("Upgrade_data_pipeline/csv")
OUT = CSV / "multisensor_6time_download_manifest_corrected_prev_snapshot.csv"
SUMMARY = CSV / "multisensor_6time_corrected_prev_summary.csv"
TP_ORDER = ["t0", "prev1", "prev2", "prev3", "seasonal", "year"]
PREV = ["prev1", "prev2", "prev3"]


def parse_time(x):
    if x is None or pd.isna(x) or str(x).strip() == "" or str(x).strip().lower() == "nan":
        return pd.NaT
    return pd.to_datetime(str(x), utc=True, errors="coerce")


def clean(x):
    if x is None or pd.isna(x):
        return ""
    s = str(x)
    return "" if s.lower() == "nan" else s


def exists_path(p):
    # Do not stat every remote path here. The snapshot is for time/overpass
    # correction; doing hundreds of thousands of Path.exists() calls on the
    # mounted data disk makes this audit unnecessarily slow.
    return bool(clean(p))


def s2_key(product, t):
    product = clean(product)
    m = re.search(r"_R(\d{3})_", product)
    orbit = m.group(1) if m else ""
    if pd.isna(t):
        return ""
    minute = (int(t.minute) // 10) * 10
    bucket = f"{t.year:04d}{t.month:02d}{t.day:02d}T{t.hour:02d}{minute:02d}"
    return f"S2|R{orbit}|{bucket}" if orbit else f"S2|{bucket}"


def l89_key(asset, t):
    asset = clean(asset)
    m = re.search(r"/(LC0[89])_(\d{3})(\d{3})_(\d{8})$", asset)
    if m:
        spacecraft, path, _row, date = m.groups()
        return f"L89|{spacecraft}|P{path}|{date}"
    if pd.isna(t):
        return ""
    return f"L89|{t.date().isoformat()}"


def emit_key(gid, t):
    gid = clean(gid)
    m = re.search(r"_(\d{7})_\d{3}$", gid)
    if m:
        return f"EMIT|{m.group(1)}"
    if pd.isna(t):
        return ""
    return "EMIT|" + t.strftime("%Y%m%dT%H%M")


def s5p_key(name, t):
    name = clean(name)
    m = re.search(r"_(\d{5})_\d{2}_\d{6}_", name)
    if m:
        return f"S5P|{m.group(1)}"
    if pd.isna(t):
        return ""
    return f"S5P|{t.date().isoformat()}"


def add(records, sensor, plume_id, tp, status, path, image_time, product_id="", product_name="", overpass_key="", source="", message="", cloud=""):
    t = parse_time(image_time)
    records.append(
        {
            "plume_id": clean(plume_id),
            "sensor": sensor,
            "timepoint_original": clean(tp),
            "status_original": clean(status),
            "raw_path": clean(path),
            "image_time": "" if pd.isna(t) else t.isoformat(),
            "image_ts": t,
            "product_id": clean(product_id),
            "product_name": clean(product_name),
            "overpass_key": clean(overpass_key),
            "source": clean(source),
            "message": clean(message),
            "cloud": clean(cloud),
            "path_exists": exists_path(path),
        }
    )


records = []

s2 = pd.read_csv(CSV / "s2_download_manifest.csv", dtype=str)
s2_ok = {"downloaded", "resume_skip_completed", "skip_existing_512"}
for _, r in s2[s2["status"].isin(s2_ok)].iterrows():
    t = parse_time(r.get("acquisition_time"))
    if pd.isna(t):
        continue
    path = clean(r.get("raw_path")) or clean(r.get("existing_512_path")) or clean(r.get("target_512_path"))
    add(records, "S2", r.get("plume_id"), r.get("timepoint"), r.get("status"), path, t, r.get("product_id"), r.get("product_name"), s2_key(r.get("product_name"), t), r.get("selection_source"), r.get("message"))

lsub = pd.read_csv(CSV / "l89_gee_drive_submit_manifest.csv", dtype=str)
lpull = pd.read_csv(CSV / "l89_drive_pull_manifest.csv", dtype=str)
path_map = {}
for _, r in lpull.iterrows():
    if clean(r.get("raw_path")):
        path_map[(clean(r.get("plume_id")), clean(r.get("timepoint")))] = clean(r.get("raw_path"))
l89_ok = {"submitted", "skip_gee_task_pending", "resume_skip_completed", "skip_existing_512", "skip_existing_valid", "skip_existing_valid_deleted_drive", "resume_skip_completed_deleted_drive"}
for _, r in lsub[lsub["status"].isin(l89_ok)].iterrows():
    t = parse_time(r.get("image_time"))
    if pd.isna(t):
        continue
    path = path_map.get((clean(r.get("plume_id")), clean(r.get("timepoint"))), clean(r.get("target_raw_dir")))
    add(records, "L89", r.get("plume_id"), r.get("timepoint"), r.get("status"), path, t, r.get("asset_id"), r.get("asset_id"), l89_key(r.get("asset_id"), t), "l89_gee_submit", r.get("message"), r.get("cloud"))

emit = pd.read_csv(CSV / "emit_download_manifest.csv", dtype=str)
ecache = pd.read_csv(CSV / "emit_granule_search_cache.csv", dtype=str)
ecache = ecache[ecache["status"].eq("found")].copy()
ecache["_t"] = ecache["granule_time"].map(parse_time)
ecache = ecache[ecache["_t"].notna()].copy()
ecache["_row"] = range(len(ecache))
gtime = ecache.sort_values("_row").drop_duplicates("granule_id", keep="last").set_index("granule_id")["_t"].to_dict()
emit_ok = {"downloaded", "linked_existing", "skip_existing", "resume_skip_completed"}
for _, r in emit[emit["status"].isin(emit_ok)].iterrows():
    gid = clean(r.get("granule_id"))
    t = gtime.get(gid, pd.NaT)
    if pd.isna(t):
        continue
    add(records, "EMIT", r.get("plume_id"), r.get("timepoint"), r.get("status"), r.get("raw_path"), t, gid, gid, emit_key(gid, t), r.get("selection_source"), r.get("message"))

s5p = pd.read_csv(CSV / "s5p_download_manifest.csv", dtype=str)
s5p_ok = {"downloaded", "skip_existing", "resume_skip_completed", "skip_existing_raw"}
for _, r in s5p[s5p["status"].isin(s5p_ok)].iterrows():
    t = parse_time(r.get("image_time"))
    if pd.isna(t):
        continue
    add(records, "S5P", r.get("plume_id"), r.get("timepoint"), r.get("status"), r.get("raw_path"), t, r.get("product_id"), r.get("product_name"), s5p_key(r.get("product_name"), t), r.get("selection_source"), r.get("message"))

df = pd.DataFrame(records)
if df.empty:
    raise SystemExit("no records")
df["_row"] = range(len(df))
df["_has_path"] = df["raw_path"].astype(bool) & df["path_exists"].astype(bool)
df = df.sort_values(["_has_path", "_row"]).drop_duplicates(["sensor", "plume_id", "timepoint_original", "product_id", "overpass_key"], keep="last")

out_rows = []
summary = []
for (sensor, pid), g in df.groupby(["sensor", "plume_id"], sort=False):
    t0s = g[g["timepoint_original"].eq("t0")].copy()
    if t0s.empty:
        continue
    t0 = t0s.sort_values(["image_ts", "_has_path"], ascending=[True, False]).iloc[0]
    if pd.isna(t0["image_ts"]) or not clean(t0["overpass_key"]):
        continue

    cand = g[g["timepoint_original"].isin(["t0", "prev1", "prev2", "prev3"])].copy()
    cand = cand[cand["image_ts"].notna()]
    cand = cand[cand["image_ts"] < t0["image_ts"]]
    cand = cand[cand["overpass_key"].astype(str) != clean(t0["overpass_key"])]

    chosen = []
    for _ok, gg in cand.groupby("overpass_key", sort=False):
        gg = gg.copy()
        gg["_dt"] = (t0["image_ts"] - gg["image_ts"]).dt.total_seconds()
        gg = gg.sort_values(["_dt", "_has_path"], ascending=[True, False])
        chosen.append(gg.iloc[0])
    chosen = sorted(chosen, key=lambda r: r["image_ts"], reverse=True)

    corrected = {"t0": t0}
    for i, tp in enumerate(PREV):
        corrected[tp] = chosen[i] if i < len(chosen) else None
    for tp in ["seasonal", "year"]:
        gg = g[g["timepoint_original"].eq(tp)].copy()
        corrected[tp] = None if gg.empty else gg.sort_values(["_has_path", "_row"], ascending=[False, False]).iloc[0]

    invalid_prev = []
    for tp in PREV:
        og = g[g["timepoint_original"].eq(tp)]
        if not og.empty:
            rr = og.sort_values("_row").iloc[-1]
            if pd.isna(rr["image_ts"]) or rr["image_ts"] >= t0["image_ts"] or clean(rr["overpass_key"]) == clean(t0["overpass_key"]):
                invalid_prev.append(tp)

    for tp in TP_ORDER:
        r = corrected.get(tp)
        if r is None:
            out_rows.append(
                {
                    "plume_id": pid,
                    "sensor": sensor,
                    "timepoint": tp,
                    "corrected_status": "missing_after_overpass_correction",
                    "raw_path": "",
                    "image_time": "",
                    "product_id": "",
                    "product_name": "",
                    "overpass_key": "",
                    "source_original_timepoint": "",
                    "source_original_status": "",
                    "path_exists": "",
                    "needs_download": "yes" if tp in PREV else "",
                    "correction_note": "no distinct earlier overpass available in downloaded records" if tp in PREV else "no downloaded/current record found",
                }
            )
        else:
            note = "kept"
            if tp in PREV and clean(r["timepoint_original"]) != tp:
                note = f"shifted_from_{clean(r['timepoint_original'])}"
            if tp == "t0":
                note = "actual_t0_anchor"
            out_rows.append(
                {
                    "plume_id": pid,
                    "sensor": sensor,
                    "timepoint": tp,
                    "corrected_status": "available",
                    "raw_path": clean(r["raw_path"]),
                    "image_time": clean(r["image_time"]),
                    "product_id": clean(r["product_id"]),
                    "product_name": clean(r["product_name"]),
                    "overpass_key": clean(r["overpass_key"]),
                    "source_original_timepoint": clean(r["timepoint_original"]),
                    "source_original_status": clean(r["status_original"]),
                    "path_exists": "yes" if bool(r["path_exists"]) else "no",
                    "needs_download": "no",
                    "correction_note": note,
                }
            )
    summary.append(
        {
            "sensor": sensor,
            "plume_id": pid,
            "t0_time": clean(t0["image_time"]),
            "t0_overpass_key": clean(t0["overpass_key"]),
            "invalid_original_prev_count": len(invalid_prev),
            "invalid_original_prev": ";".join(invalid_prev),
            "corrected_prev_available": sum(1 for tp in PREV if corrected.get(tp) is not None),
            "corrected_prev_missing": sum(1 for tp in PREV if corrected.get(tp) is None),
        }
    )

out = pd.DataFrame(out_rows)
out.to_csv(OUT, index=False)
sumdf = pd.DataFrame(summary)
agg = (
    sumdf.groupby("sensor")
    .agg(
        plume_with_t0=("plume_id", "nunique"),
        groups_with_invalid_original_prev=("invalid_original_prev_count", lambda x: int((x > 0).sum())),
        invalid_original_prev_total=("invalid_original_prev_count", "sum"),
        corrected_prev_available_total=("corrected_prev_available", "sum"),
        corrected_prev_missing_total=("corrected_prev_missing", "sum"),
    )
    .reset_index()
)
agg.to_csv(SUMMARY, index=False)
print("wrote", OUT, len(out), "rows")
print("wrote", SUMMARY)
print(agg.to_string(index=False))
shifts = out[(out["timepoint"].isin(PREV)) & (out["correction_note"].str.startswith("shifted_from_", na=False))]
print("shifted prev rows:", len(shifts))
if len(shifts):
    print(shifts.groupby(["sensor", "timepoint", "correction_note"]).size().reset_index(name="n").to_string(index=False))
