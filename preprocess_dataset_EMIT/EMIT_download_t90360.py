from pathlib import Path
from typing import Any, Optional

import earthaccess
import pandas as pd

# 登录 NASA Earthdata
auth = earthaccess.login()

# 配置
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = "./merged_with_emit_tag.csv"
EMIT_RAW_DIR = Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_EMIT")
EMIT_RAW_DIR.mkdir(exist_ok=True)

WINDOW_DAYS = 30
OFFSETS = [
    (90, "emit_-90_granule_id"),
    (360, "emit_-360_granule_id"),
]


def _safe_get(obj: Any, keys: list[str]) -> Any:
    cur = obj
    for k in keys:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            try:
                cur = cur[k]
            except Exception:
                cur = getattr(cur, k, None)
    return cur


def get_granule_id(granule: Any) -> Optional[str]:
    candidates = [
        _safe_get(granule, ["umm", "GranuleUR"]),
        _safe_get(granule, ["meta", "native-id"]),
        _safe_get(granule, ["meta", "concept-id"]),
    ]
    for c in candidates:
        if c is not None and str(c).strip():
            return str(c).strip()
    return None


def get_granule_time(granule: Any) -> pd.Timestamp:
    dt = _safe_get(granule, ["umm", "TemporalExtent", "RangeDateTime", "BeginningDateTime"])
    ts = pd.to_datetime(dt, utc=True, errors="coerce")
    return ts


def download_one(granule: Any, granule_id: str) -> None:
    if list(EMIT_RAW_DIR.glob(f"*{granule_id}*.nc")):
        print(f"{granule_id} 已存在，跳过下载")
        return

    print(f"开始下载 {granule_id}")
    earthaccess.download([granule], str(EMIT_RAW_DIR))
    print(f"{granule_id} 下载完成")


def find_best_granule(lat: float, lon: float, target_time: pd.Timestamp) -> Optional[Any]:
    start = (target_time - pd.Timedelta(days=WINDOW_DAYS)).isoformat()
    end = (target_time + pd.Timedelta(days=WINDOW_DAYS)).isoformat()

    results = earthaccess.search_data(
        short_name="EMITL2ARFL",
        point=(float(lon), float(lat)),
        temporal=(start, end),
        count=200,
    )
    if not results:
        return None

    best = None
    best_diff = None
    for g in results:
        gt = get_granule_time(g)
        if pd.isna(gt):
            continue
        diff = abs((gt - target_time).total_seconds())
        if best is None or diff < best_diff:
            best = g
            best_diff = diff

    return best if best is not None else results[0]


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")

    for _, col in OFFSETS:
        if col not in df.columns:
            df[col] = pd.NA

    required_cols = ["datetime", "plume_latitude", "plume_longitude", "emit_granule_id"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV 缺少必要列: {missing}")

    # 只处理已有 t0 匹配记录的行（t0 已下载，不重复处理）
    work_idx = df[df["emit_granule_id"].notna()].index
    print(f"待处理行数: {len(work_idx)}")

    for n, i in enumerate(work_idx, start=1):
        row = df.loc[i]
        base_time = row["datetime"]
        lat = row["plume_latitude"]
        lon = row["plume_longitude"]

        if pd.isna(base_time) or pd.isna(lat) or pd.isna(lon):
            print(f"[{n}/{len(work_idx)}] 行 {i} 缺少时间/经纬度，跳过")
            continue

        print(f"[{n}/{len(work_idx)}] 行 {i} 开始处理")
        for days, col in OFFSETS:
            # 已有记录时不重复检索
            if pd.notna(df.at[i, col]) and str(df.at[i, col]).strip():
                continue

            target_time = base_time - pd.Timedelta(days=days)
            granule = find_best_granule(float(lat), float(lon), target_time)
            if granule is None:
                print(f"  {col}: 未找到候选 (target={target_time})")
                df.at[i, col] = pd.NA
                continue

            granule_id = get_granule_id(granule)
            if not granule_id:
                print(f"  {col}: 找到候选但无 granule_id")
                df.at[i, col] = pd.NA
                continue

            df.at[i, col] = granule_id
            print(f"  {col}: {granule_id}")
            download_one(granule, granule_id)

    df.to_csv(CSV_PATH, index=False)
    print(f"已更新 CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
