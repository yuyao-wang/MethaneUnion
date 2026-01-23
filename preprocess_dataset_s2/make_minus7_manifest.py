import os
import ast
import time
import threading
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

# ========= Config =========
IN_CSV = "/data2/yuyao/methane_emission/preprocess_dataset_s2/CM_S2_L2A.csv"
OUT_MANIFEST = "/data2/yuyao/methane_emission/preprocess_dataset_s2/manifest_minus7_plume_to_safe.csv"

RAW_DIR = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/data_download/raw_data_dir_s2_-7"

SEARCH_BACK_DAYS = 15
CLOUD_COVER_MAX = 20.0

# 仅对已有 S2_path 且文件存在的 plume 做 -7（你之前的逻辑）
BASE_PREFIX = "/data2/yuyao/methane_emission"

MAX_WORKERS = 12  # 只做 catalogue query，可稍大
REQ_TIMEOUT = 120

# ========= Utils =========
def debug(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}][pid:{os.getpid()}][tid:{threading.get_ident()}] {msg}", flush=True)

def parse_iso_datetime(value):
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None

def dt_to_query(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

def resolve_s2_path(p: str) -> str:
    if not isinstance(p, str) or not p:
        return ""
    if os.path.isabs(p):
        return p
    return os.path.join(BASE_PREFIX, p)

def parse_bounds(bounds_str, lat, lon):
    if isinstance(bounds_str, str) and bounds_str:
        try:
            arr = ast.literal_eval(bounds_str)
            if isinstance(arr, (list, tuple)) and len(arr) == 4:
                return [float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])]
        except Exception:
            pass
    return [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01]

def build_poly_from_bounds(bounds):
    minlon, minlat, maxlon, maxlat = bounds
    return f"({minlon} {minlat},{minlon} {maxlat},{maxlon} {maxlat},{maxlon} {minlat},{minlon} {minlat})"

# ========= Catalogue query =========
def fetch_products(poly_wkt: str, start_dt: datetime, end_dt: datetime):
    start_ts = dt_to_query(start_dt)
    end_ts = dt_to_query(end_dt)

    next_link = (
        "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
        f"$filter=Collection/Name eq 'SENTINEL-2' "
        f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        f"and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') "
        f"and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
        f"and att/OData.CSC.DoubleAttribute/Value le {CLOUD_COVER_MAX:.2f}) "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;POLYGON({poly_wkt})') "
        f"and ContentDate/Start gt {start_ts} "
        f"and ContentDate/Start lt {end_ts}"
        "&$top=1000"
    )

    products = []
    while next_link:
        resp = requests.get(next_link, timeout=REQ_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
        values = payload.get("value", [])
        for prod in values:
            st = (prod.get("ContentDate") or {}).get("Start", "")
            acq = parse_iso_datetime(st)
            if acq is None:
                continue
            products.append({"Id": prod.get("Id"), "Name": prod.get("Name"), "acq_time": acq})
        next_link = payload.get("@odata.nextLink", "")
    return products

def select_previous_overpass(products, event_dt: datetime):
    prev = [p for p in products if p["acq_time"] < event_dt]
    if not prev:
        return None
    return max(prev, key=lambda p: p["acq_time"])

def raw_exists(product_name: str) -> bool:
    if not isinstance(product_name, str) or not product_name:
        return False
    return (
        os.path.isdir(os.path.join(RAW_DIR, product_name)) or
        os.path.exists(os.path.join(RAW_DIR, product_name + ".zip")) or
        os.path.exists(os.path.join(RAW_DIR, product_name + ".SAFE.zip")) or
        os.path.exists(os.path.join(RAW_DIR, product_name + ".SAFE"))
    )

# ========= Worker =========
def process_one(row: dict):
    plume_id = str(row.get("plume_id", ""))
    lat = row.get("plume_latitude", None)
    lon = row.get("plume_longitude", None)
    event_dt = parse_iso_datetime(row.get("datetime", ""))
    s2_path = resolve_s2_path(row.get("S2_path", ""))

    out = {"plume_id": plume_id, "ok": 0, "reason": ""}

    if (not plume_id) or (lat is None) or (lon is None) or (event_dt is None):
        out["reason"] = "missing required fields"
        return out

    if not s2_path or (not os.path.exists(s2_path)):
        out["reason"] = "no existing S2_path on disk"
        return out

    bounds = parse_bounds(row.get("plume_bounds", ""), float(lat), float(lon))
    poly = build_poly_from_bounds(bounds)

    start_dt = event_dt - timedelta(days=SEARCH_BACK_DAYS)
    end_dt = event_dt + timedelta(minutes=1)

    try:
        products = fetch_products(poly, start_dt, end_dt)
        selected = select_previous_overpass(products, event_dt)
        if selected is None:
            out["reason"] = f"no previous overpass within {SEARCH_BACK_DAYS} days"
            return out

        out.update({
            "ok": 1,
            "event_datetime": event_dt.isoformat(),
            "lat": float(lat),
            "lon": float(lon),
            "s2_existing_path": s2_path,
            "s2_minus7_id": selected["Id"],
            "s2_minus7_product_name": selected["Name"],
            "s2_minus7_datetime": selected["acq_time"].isoformat(),
        })
        out["raw_dir_exists"] = int(raw_exists(selected["Name"]))
        return out

    except Exception as e:
        out["reason"] = f"catalogue error: {e}"
        return out

# ========= Main =========
if __name__ == "__main__":
    debug("load csv")
    df = pd.read_csv(IN_CSV)
    df["S2_path_abs"] = df["S2_path"].apply(resolve_s2_path)
    work_df = df[df["S2_path_abs"].apply(lambda p: isinstance(p,str) and p and os.path.exists(p))].copy()

    debug(f"input={len(df)} usable_existing_S2={len(work_df)}")

    # 简单并行
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = []
    t0 = time.time()
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(process_one, row.to_dict()) for _, row in work_df.iterrows()]
        for f in as_completed(futs):
            res = f.result()
            results.append(res)
            done += 1
            if done % 50 == 0:
                elapsed = time.time() - t0
                debug(f"progress {done}/{len(futs)} | rate={done/max(elapsed,1e-6):.2f}/s")

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_MANIFEST, index=False)
    debug(f"saved manifest: {OUT_MANIFEST} rows={len(out_df)}")
    debug(f"ok={int((out_df['ok']==1).sum())} fail={int((out_df['ok']==0).sum())}")
