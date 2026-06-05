import pandas as pd
import earthaccess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Log in to NASA Earthdata
auth = earthaccess.login()

# config
CSV_PATH = "./merged_with_emit_tag.csv"
# CSV_PATH = "./merged_with_emit_tag.csv"
EMIT_RAW_DIR = Path("/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/raw_data_dir_EMIT")
EMIT_RAW_DIR.mkdir(exist_ok=True)

# Translated comment
df = pd.read_csv(CSV_PATH)
df['datetime'] = pd.to_datetime(df['datetime'])

filtered_df = df.drop_duplicates(subset=['emit_granule_id'])

print(f"download EMIT : {len(filtered_df)}")

def download_granule(granule_id):
    """Translated to English."""
    print(f"starting processing: {granule_id}")
    # Translated comment
    if list(EMIT_RAW_DIR.glob(f"*{granule_id}*.nc")):
        print(f"{granule_id} already exists, skipping download")
 return f"{granule_id} "

 print(f": {granule_id}")
    # Translated comment
    results = earthaccess.search_data(
        short_name='EMITL2ARFL',  # Translated comment
        granule_name=granule_id,
        count=1
    )

    if results:
 print(f" {granule_id}, Starting download...")
        earthaccess.download(results, str(EMIT_RAW_DIR))
        print(f"{granule_id} download complete")
        return f"{granule_id} download complete"
    
 print(f"{granule_id} ")
 return f"{granule_id} "

# Translated comment
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(download_granule, filtered_df['emit_granule_id']))