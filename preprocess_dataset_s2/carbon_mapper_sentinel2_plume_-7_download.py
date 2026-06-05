import os
import re
import ast
import time
import zipfile
import shutil
import threading
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque

import requests
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
import tifffile

# =========================
# Config
# =========================
IN_CSV = "/data2/yuyao/methane_emission/preprocess_dataset_s2/CM_S2_L2A.csv"
OUT_CSV = "/data2/yuyao/methane_emission/preprocess_dataset_s2/CM_S2_L2A_-7.csv"

BASE_DIR = "/data2/yuyao/methane_emission/carbon_mapper_data/CM_S2_L2A"
RAW_DIR = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/data_download/raw_data_dir_s2_-7"

SEARCH_BACK_DAYS = 15
CLOUD_COVER_MAX = 20.0
WINDOW_SIZE = 512
MAX_WORKERS = 6

# Translated comment
# Translated comment
MAX_CENTER_SHIFT_PX = WINDOW_SIZE // 2

# 20m bands
BAND_ORDER = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12"]
BAND_TO_INDEX = {b:i for i,b in enumerate(BAND_ORDER)}
JP2_BAND_RE = re.compile(r".*_(B[0-9]{1,2}|B8A)_20m\.jp2$")

# Translated comment
CDSE_USER = os.environ.get("CDSE_USER", "")
CDSE_PASS = os.environ.get("CDSE_PASS", "")
if not CDSE_USER or not CDSE_PASS:
 print("[WARN] CDSE_USER / CDSE_PASS .: ")
    print("  export CDSE_USER='xxx' ; export CDSE_PASS='xxx'")

# =========================
# Download throttling knobs
# =========================
ZIPPER_MIN_INTERVAL_SEC = 1.2
ZIPPER_MAX_CONCURRENT = 1

# =========================
# Utils
# =========================
def debug(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}][pid:{os.getpid()}][tid:{threading.get_ident()}] {msg}", flush=True)

def parse_iso_datetime(value):
    if not isinstance(value, str) or len(value) == 0:
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

def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def resolve_s2_path(p: str) -> str:
    if not isinstance(p, str) or len(p) == 0:
        return ""
    if os.path.isabs(p):
        return p
    return os.path.join("/data2/yuyao/methane_emission", p)

def parse_bounds(bounds_str, lat, lon):
    if isinstance(bounds_str, str) and len(bounds_str) > 0:
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

def human_time(seconds: float) -> str:
    if seconds is None or seconds < 0:
        return "N/A"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"

# =========================
# Global rate limiting & locks
# =========================
_zipper_sema = threading.Semaphore(ZIPPER_MAX_CONCURRENT)

class RateLimiter:
    def __init__(self, min_interval_sec: float):
        self.min_interval = float(min_interval_sec)
        self._lock = threading.Lock()
        self._next_time = 0.0

    def wait(self):
        with self._lock:
            now = time.time()
            if now < self._next_time:
                time.sleep(self._next_time - now)
            self._next_time = time.time() + self.min_interval

zipper_rl = RateLimiter(ZIPPER_MIN_INTERVAL_SEC)

# Translated comment
_product_locks = defaultdict(threading.Lock)

def request_with_retry(method, url, session: requests.Session, *,
                       max_tries=8,
                       base_sleep=2.0,
                       timeout=300,
                       stream=False):
    last_exc = None
    for attempt in range(1, max_tries + 1):
        try:
            resp = session.request(method, url, timeout=timeout, stream=stream)

            if resp.status_code == 429:
                ra = resp.headers.get("Retry-After")
                if ra is not None:
                    try:
                        sleep_s = float(ra)
                    except Exception:
                        sleep_s = base_sleep
                else:
                    sleep_s = base_sleep * (2 ** (attempt - 1))
                sleep_s += random.uniform(0, 0.5)
                debug(f"429 Too Many Requests -> sleep {sleep_s:.2f}s (attempt {attempt}/{max_tries})")
                time.sleep(sleep_s)
                continue

            if resp.status_code in (500, 502, 503, 504):
                sleep_s = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                debug(f"{resp.status_code} server error -> sleep {sleep_s:.2f}s (attempt {attempt}/{max_tries})")
                time.sleep(sleep_s)
                continue

            resp.raise_for_status()
            return resp

        except requests.RequestException as e:
            last_exc = e
            sleep_s = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            debug(f"request error: {e} -> sleep {sleep_s:.2f}s (attempt {attempt}/{max_tries})")
            time.sleep(sleep_s)

    raise last_exc if last_exc else RuntimeError("request_with_retry failed")

# =========================
# Progress tracker
# =========================
class Progress:
    def __init__(self, total: int, print_every: int = 1):
        self.total = total
        self.print_every = max(1, int(print_every))
        self.lock = threading.Lock()
        self.start = time.time()

        self.done = 0
        self.ok = 0
        self.skipped = 0
        self.fail = 0

        self.last_durations = deque(maxlen=100)

    def update(self, status: str, duration: float):
        with self.lock:
            self.done += 1
            if status == "ok":
                self.ok += 1
            elif status == "skipped":
                self.skipped += 1
            else:
                self.fail += 1

            if duration is not None and duration > 0:
                self.last_durations.append(duration)

            if (self.done % self.print_every) == 0 or self.done == self.total:
                self._print_locked()

    def _print_locked(self):
        elapsed = time.time() - self.start
        if len(self.last_durations) > 0:
            avg = sum(self.last_durations) / len(self.last_durations)
        else:
            avg = elapsed / max(1, self.done)

        remain = (self.total - self.done) * avg
        speed = self.done / elapsed if elapsed > 0 else 0.0

        msg = (
            f"[PROGRESS] {self.done}/{self.total} "
            f"(ok={self.ok}, skipped={self.skipped}, fail={self.fail}) | "
            f"elapsed={human_time(elapsed)} | avg(last{len(self.last_durations)})={avg:.2f}s | "
            f"ETA={human_time(remain)} | {speed:.2f} items/s"
        )
        print(msg, flush=True)

# =========================
# CDSE Auth
# =========================
def get_access_token(username: str, password: str) -> str:
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    r = requests.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data=data,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["access_token"]

class RefreshableAccessToken:
    def __init__(self, user, pwd) -> None:
        self.user = user
        self.pwd = pwd
        self.value = get_access_token(user, pwd)
        self.lock = threading.Lock()

    def update(self):
        with self.lock:
            self.value = get_access_token(self.user, self.pwd)

    def get(self):
        with self.lock:
            return self.value

def refresh_token_daemon(token_obj: RefreshableAccessToken):
    while True:
        try:
            debug("refresh token")
            token_obj.update()
        except Exception as e:
            debug(f"token refresh failed: {e}")
        time.sleep(300)

# =========================
# Catalogue query: find previous overpass
# =========================
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
        resp = requests.get(next_link, timeout=120)
        resp.raise_for_status()
        payload = resp.json()
        values = payload.get("value", [])
        for prod in values:
            cd = prod.get("ContentDate", {})
            st = cd.get("Start", "")
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

# =========================
# Download & extract
# =========================
def download_product_zip(token: str, product_id: str, out_zip: str):
    url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"

    # Translated comment
    if os.path.exists(out_zip) and os.path.getsize(out_zip) > 0:
        return

    tmp = out_zip + ".part"
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except Exception:
            pass

    headers = {"Authorization": f"Bearer {token}"}

    with _zipper_sema:
        zipper_rl.wait()
        with requests.Session() as s:
            s.headers.update(headers)
            with request_with_retry("GET", url, s, stream=True, timeout=300) as r:
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp, out_zip)

def extract_needed_files(zip_path: str, out_dir: str):
    safe_mkdir(out_dir)
    marker = os.path.join(out_dir, ".extract_complete")
    if os.path.exists(marker):
        return

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        pick = []
        for n in names:
            if ("GRANULE/" in n) and ("/IMG_DATA/R20m/" in n) and n.lower().endswith(".jp2"):
                pick.append(n)
            elif n.endswith("MTD_MSIL2A.xml"):
                pick.append(n)

        for n in pick:
            if n.endswith("/"):
                continue
            fn = os.path.basename(n)
            if not fn:
                continue
            target = os.path.join(out_dir, fn)
            if os.path.exists(target):
                continue
            with z.open(n) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)

    with open(marker, "w") as f:
        f.write("ok")

# =========================
# Crop window (allow shift, no padding)
# =========================
def latlon_to_pixel(lat, lon, dataset):
    if dataset.crs is None:
        raise RuntimeError("JP2 dataset has no CRS")
    transformer = Transformer.from_crs("EPSG:4326", dataset.crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    col, row = (~dataset.transform) * (x, y)
    return float(col), float(row)

def compute_window_with_limited_shift(ds, center_lat: float, center_lon: float,
                                      window_size: int, max_shift_px: int):
    """
 padding; clamp shift, shift .
 "sample", max_shift_px 256 (=512/2).
    """
    col, row = latlon_to_pixel(center_lat, center_lon, ds)
    half = window_size // 2

    col0 = int(np.floor(col)) - half
    row0 = int(np.floor(row)) - half

    # Translated comment
    col0c = min(max(col0, 0), max(0, ds.width - window_size))
    row0c = min(max(row0, 0), max(0, ds.height - window_size))

    shift_c = abs(col0c - col0)
    shift_r = abs(row0c - row0)

    # Translated comment
    if ds.width < window_size or ds.height < window_size:
        return None

    if shift_c > max_shift_px or shift_r > max_shift_px:
        return None

    return Window(col0c, row0c, window_size, window_size), int(shift_c), int(shift_r)

def build_s2_12band_tif(extracted_dir: str, center_lat, center_lon,
                        out_tif: str, max_shift_px: int):
    """
 :  - band window()
 - band window load,  - padding; 512 failure
 (out_array, shift_c, shift_r, has_shift) None
    """
    jp2_files = list(Path(extracted_dir).glob("*.jp2"))
    band_paths = {}
    for p in jp2_files:
        m = JP2_BAND_RE.match(str(p))
        if not m:
            continue
        b = m.group(1)
        if b in BAND_TO_INDEX:
            band_paths[b] = str(p)

    # Translated comment
    ref_path = None
    for b in ["B11", "B12", "B8A", "B8", "B4", "B3", "B2", "B1"]:
        if b in band_paths:
            ref_path = band_paths[b]
            break
    if ref_path is None:
        return None

    with rasterio.open(ref_path) as ref_ds:
        computed = compute_window_with_limited_shift(
            ref_ds, float(center_lat), float(center_lon),
            WINDOW_SIZE, max_shift_px
        )
        if computed is None:
            return None
        win, shift_c, shift_r = computed

    out = np.zeros((len(BAND_ORDER), WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)

    # Translated comment
    for bname, idx in BAND_TO_INDEX.items():
        p = band_paths.get(bname)
        if not p:
            continue
        try:
            with rasterio.open(p) as ds:
                arr = ds.read(1, window=win)
                if arr.shape != (WINDOW_SIZE, WINDOW_SIZE):
                    return None
                out[idx] = arr.astype(np.float32)
        except Exception:
            return None

    safe_mkdir(os.path.dirname(out_tif))
    tifffile.imwrite(out_tif, out)

    has_shift = 1 if (shift_c > 0 or shift_r > 0) else 0
    return out, shift_c, shift_r, has_shift

# =========================
# Main task per plume
# =========================
def process_one(row, token_obj: RefreshableAccessToken):
    plume_id = str(row.get("plume_id", ""))
    lat = row.get("plume_latitude", None)
    lon = row.get("plume_longitude", None)
    event_dt = parse_iso_datetime(row.get("datetime", ""))
    s2_path = resolve_s2_path(row.get("S2_path", ""))

    if (not plume_id) or (lat is None) or (lon is None) or (event_dt is None):
        return {"plume_id": plume_id, "ok": 0, "reason": "missing required fields"}

    if not s2_path or (not os.path.exists(s2_path)):
        return {"plume_id": plume_id, "ok": 0, "reason": "no existing S2_path on disk"}

    plume_dir = os.path.join(BASE_DIR, plume_id)
    out_tif = os.path.join(plume_dir, "s2_-7.tif")

    # Translated comment
    if os.path.exists(out_tif) and os.path.getsize(out_tif) > 0:
        return {
            "plume_id": plume_id, "ok": 1, "skipped": 1,
            "s2_minus7_tif": out_tif,
            "s2_minus7_center_shift_col_px": 0,
            "s2_minus7_center_shift_row_px": 0,
            "s2_minus7_has_shift": 0,
        }

    bounds = parse_bounds(row.get("plume_bounds", ""), float(lat), float(lon))
    poly = build_poly_from_bounds(bounds)

    start_dt = event_dt - timedelta(days=SEARCH_BACK_DAYS)
    end_dt = event_dt + timedelta(minutes=1)
    products = fetch_products(poly, start_dt, end_dt)
    selected = select_previous_overpass(products, event_dt)
    if selected is None:
        return {"plume_id": plume_id, "ok": 0, "reason": f"no previous overpass within {SEARCH_BACK_DAYS} days"}

    prod_id = selected["Id"]
    prod_name = selected["Name"]
    acq_dt = selected["acq_time"]

    prod_dir = os.path.join(RAW_DIR, prod_name)
    zip_path = os.path.join(RAW_DIR, prod_name + ".zip")
    extract_marker = os.path.join(prod_dir, ".extract_complete")
    safe_mkdir(RAW_DIR)

    # Translated comment
    with _product_locks[prod_name]:
        if not os.path.exists(extract_marker):
            if (not os.path.exists(zip_path)) or (os.path.getsize(zip_path) == 0):
                debug(f"[{plume_id}] downloading previous product: {prod_name} {acq_dt.isoformat()}")
                download_product_zip(token_obj.get(), prod_id, zip_path)

            debug(f"[{plume_id}] extracting: {prod_name}")
            extract_needed_files(zip_path, prod_dir)

    safe_mkdir(plume_dir)
    debug(f"[{plume_id}] building s2_-7.tif from {prod_name}")

    built = build_s2_12band_tif(prod_dir, float(lat), float(lon), out_tif, MAX_CENTER_SHIFT_PX)
    if built is None:
        return {"plume_id": plume_id, "ok": 0, "reason": "crop failed (image < 512 or window invalid)"}

    arr, shift_c, shift_r, has_shift = built

    # Translated comment
    zero_ratio = (arr == 0).reshape(arr.shape[0], -1).mean(axis=1)
    missing = [BAND_ORDER[i] for i, zr in enumerate(zero_ratio) if zr >= 0.999]

    return {
        "plume_id": plume_id,
        "ok": 1,
        "skipped": 0,
        "event_datetime": event_dt.isoformat(),
        "s2_existing_path": s2_path,
        "s2_minus7_datetime": acq_dt.isoformat(),
        "s2_minus7_product_name": prod_name,
        "s2_minus7_raw_dir": prod_dir,
        "s2_minus7_tif": out_tif,
        "s2_minus7_missing_bands": ",".join(missing),
        # ✅ record shift
        "s2_minus7_center_shift_col_px": int(shift_c),
        "s2_minus7_center_shift_row_px": int(shift_r),
        "s2_minus7_has_shift": int(has_shift),
    }

# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    debug("script start")
    df = pd.read_csv(IN_CSV)

    # Translated comment
    df["S2_path_abs"] = df["S2_path"].apply(resolve_s2_path)
    mask = df["S2_path_abs"].apply(lambda p: isinstance(p, str) and len(p) > 0 and os.path.exists(p))
    work_df = df[mask].copy()
    debug(f"input rows={len(df)} with existing S2 on disk={len(work_df)}")

    if len(work_df) == 0:
        debug("no rows to process; exiting")
        df.to_csv(OUT_CSV, index=False)
        debug(f"saved: {OUT_CSV}")
        raise SystemExit(0)

    token_obj = RefreshableAccessToken(CDSE_USER, CDSE_PASS)
    th = threading.Thread(target=refresh_token_daemon, args=(token_obj,), daemon=True)
    th.start()

    progress = Progress(total=len(work_df), print_every=1)

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = []
        fut_start = {}

        for _, row in work_df.iterrows():
            fut = ex.submit(process_one, row.to_dict(), token_obj)
            futs.append(fut)
            fut_start[fut] = time.time()

        for fut in as_completed(futs):
            start_t = fut_start.get(fut, None)
            duration = (time.time() - start_t) if start_t else None

            status = "fail"
            try:
                res = fut.result()
                results.append(res)

                if res.get("ok") == 1 and res.get("skipped", 0):
                    status = "skipped"
                    debug(f"skip plume={res.get('plume_id')} (s2_-7.tif exists)")
                elif res.get("ok") == 1:
                    status = "ok"
                    debug(f"done plume={res.get('plume_id')} prev={res.get('s2_minus7_datetime')} "
                          f"shift=({res.get('s2_minus7_center_shift_col_px')},{res.get('s2_minus7_center_shift_row_px')}) "
                          f"missing={res.get('s2_minus7_missing_bands')}")
                else:
                    status = "fail"
                    debug(f"fail plume={res.get('plume_id')} reason={res.get('reason')}")

            except Exception as e:
                results.append({"plume_id": "UNKNOWN", "ok": 0, "reason": f"exception: {e}"})
                status = "fail"
                debug(f"exception from worker: {e}")

            progress.update(status=status, duration=duration)

    res_df = pd.DataFrame(results)

    # Translated comment
    out_df = df.copy()
    new_cols = [
        "s2_minus7_datetime",
        "s2_minus7_product_name",
        "s2_minus7_raw_dir",
        "s2_minus7_tif",
        "s2_minus7_missing_bands",
        # ✅ shift columns
        "s2_minus7_center_shift_col_px",
        "s2_minus7_center_shift_row_px",
        "s2_minus7_has_shift",
    ]
    for c in new_cols:
        if c not in out_df.columns:
            out_df[c] = ""

    ok_df = res_df[res_df.get("ok", 0) == 1].copy()
    if len(ok_df) > 0:
        merge_cols = ["plume_id"] + new_cols
        ok_df = ok_df[merge_cols].drop_duplicates("plume_id")

        out_df = out_df.merge(ok_df, on="plume_id", how="left", suffixes=("", "_new"))
        for c in new_cols:
            cn = c + "_new"
            if cn in out_df.columns:
                out_df[c] = out_df[cn].fillna(out_df[c])
                out_df.drop(columns=[cn], inplace=True)

    out_df.to_csv(OUT_CSV, index=False)
    debug(f"saved: {OUT_CSV}")
    debug("all done")
