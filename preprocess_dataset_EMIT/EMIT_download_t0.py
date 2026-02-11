import pandas as pd
import earthaccess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# 登录 NASA Earthdata
auth = earthaccess.login()

# 配置
CSV_PATH = "./merged_with_emit_tag.csv"
# CSV_PATH = "./merged_with_emit_tag.csv"
EMIT_RAW_DIR = Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_EMIT")
EMIT_RAW_DIR.mkdir(exist_ok=True)

# 1. 筛选数据
df = pd.read_csv(CSV_PATH)
df['datetime'] = pd.to_datetime(df['datetime'])

filtered_df = df.drop_duplicates(subset=['emit_granule_id'])

print(f"找到待下载的独特 EMIT 颗粒数量: {len(filtered_df)}")

def download_granule(granule_id):
    """修复后的单任务下载函数"""
    print(f"开始处理: {granule_id}")
    # 检查本地是否已存在，避免重复下载
    if list(EMIT_RAW_DIR.glob(f"*{granule_id}*.nc")):
        print(f"{granule_id} 已存在，跳过下载")
        return f"{granule_id} 已存在"

    print(f"正在搜索: {granule_id}")
    # 关键点：必须指定 short_name 或 collection_concept_id
    results = earthaccess.search_data(
        short_name='EMITL2ARFL',  # 显式限定为 EMIT L2A 反射率产品
        granule_name=granule_id,
        count=1
    )

    if results:
        print(f"找到 {granule_id}，开始下载...")
        earthaccess.download(results, str(EMIT_RAW_DIR))
        print(f"{granule_id} 下载完成")
        return f"{granule_id} 下载完成"
    
    print(f"{granule_id} 未找到")
    return f"{granule_id} 未找到"

# 使用线程池加速下载
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(download_granule, filtered_df['emit_granule_id']))