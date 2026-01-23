"""
Simple helper to download Landsat 8/9 Collection 2 Level-2 products
(30m SR bands 1-7, ST_B10, QA + aux files) from the usgs-landsat S3 bucket.

Prerequisites:
- `pip install boto3`
- Configure AWS credentials (e.g., `aws configure`)
- S3 bucket is requester-pays, so we pass RequestPayer='requester'.

Usage:
    from landsat_c2_downloader import download_landsat_scene

    scene_id = "LC08_L2SP_172057_20210101_20210308_02_T1"
    download_landsat_scene(scene_id, "/data/yuyao/landsat_raw")
"""

import os
from dataclasses import dataclass
from typing import List, Dict

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError


BUCKET_NAME = "usgs-landsat"
COLLECTION_PREFIX = "collection02/level-2/standard/oli-tirs"
AWS_REGION = "us-west-2"  # Landsat bucket region
# boto3's TransferManager spins up its own ThreadPoolExecutor. That collides
# with the ThreadPoolExecutor we already use at a higher level and triggered
# RuntimeError: "cannot schedule new futures after interpreter shutdown"
# once the interpreter begins tearing down. We disable boto3's internal
# threading because we already parallelize downloads across plumes.
SINGLE_THREAD_TRANSFER = TransferConfig(use_threads=False)


@dataclass
class LandsatSceneLocation:
    scene_id: str
    year: str
    path: str
    row: str

    @property
    def s3_prefix(self) -> str:
        """
        S3 prefix for this scene under the usgs-landsat bucket.

        s3://usgs-landsat/collection02/level-2/standard/oli-tirs/<year>/<path>/<row>/<scene_id>/
        """
        return f"{COLLECTION_PREFIX}/{self.year}/{self.path}/{self.row}/{self.scene_id}"


def normalize_scene_id(scene_id: str) -> str:
    """
    Landsat Collection 2 Level-2 STAC items often append suffixes like
    \"_SR\" (surface reflectance) to the product ID, while the underlying
    S3 directory and filenames use the base product ID. Strip the known
    suffixes so we can build valid S3 keys.
    """
    suffixes = ("_SR", "_ST")
    for suffix in suffixes:
        if scene_id.endswith(suffix):
            return scene_id[: -len(suffix)]
    return scene_id


def parse_scene_id(scene_id: str) -> LandsatSceneLocation:
    """
    Parse a Landsat Collection 2 Level-2 scene ID.

    Example scene_id:
        LC08_L2SP_172057_20210101_20210308_02_T1

    Format:
        LXSS_LLLL_PPPRRR_YYYYMMDD_yyyymmdd_CC_TX

    We need:
        year   = YYYY
        path   = PPP
        row    = RRR
    """
    normalized_id = normalize_scene_id(scene_id)

    parts = normalized_id.split("_")
    if len(parts) < 4:
        raise ValueError(f"Invalid scene_id format: {scene_id}")

    pprrr = parts[2]
    if len(pprrr) != 6 or not pprrr.isdigit():
        raise ValueError(f"Invalid path/row in scene_id: {scene_id}")

    path = pprrr[:3]
    row = pprrr[3:]
    year = parts[3][:4]

    return LandsatSceneLocation(scene_id=normalized_id, year=year, path=path, row=row)


def build_required_filenames(scene_id: str) -> Dict[str, List[str]]:
    """
    Given a scene_id, return the list of filenames we want to download
    from the scene directory.

    - 30 m Surface Reflectance bands: SR_B1–SR_B7
    - 30 m Surface Temperature: ST_B10
    - QA / auxiliary:
        QA_PIXEL, QA_RADSAT, SR_QA_AEROSOL, ST_QA
        ANG.txt, MTL.txt
    """
    # SR bands 1-7
    sr_bands = [f"{scene_id}_SR_B{b}.TIF" for b in range(1, 8)]

    # Surface temperature band
    st_band = [f"{scene_id}_ST_B10.TIF"]

    # QA rasters
    qa_files = [
        f"{scene_id}_QA_PIXEL.TIF",
        f"{scene_id}_QA_RADSAT.TIF",
        f"{scene_id}_SR_QA_AEROSOL.TIF",
        f"{scene_id}_ST_QA.TIF",
    ]

    # Auxiliary text files
    aux_files = [
        f"{scene_id}_MTL.txt",
    ]

    return {
        "sr": sr_bands,
        "st": st_band,
        "qa": qa_files,
        "aux": aux_files,
    }


def get_s3_client(region_name: str = AWS_REGION):
    """
    Create a boto3 S3 client. Assumes AWS credentials are configured.
    """
    return boto3.client("s3", region_name=region_name)


def download_single_object(
    s3_client,
    key: str,
    local_path: str,
    request_payer: str = "requester",
) -> bool:
    """
    Download a single S3 object (key) to local_path.

    Returns True if downloaded or already exists, False if object not found.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if os.path.exists(local_path):
        # Already downloaded
        print(f"[skip] {local_path} already exists.")
        return True

    try:
        print(f"[download] s3://{BUCKET_NAME}/{key} -> {local_path}")
        s3_client.download_file(
            BUCKET_NAME,
            key,
            local_path,
            ExtraArgs={"RequestPayer": request_payer},
            Config=SINGLE_THREAD_TRANSFER,
        )
        return True
    except ClientError as e:
        # NotFound or permission issues
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "404" or "Not Found" in str(e):
            print(f"[missing] s3://{BUCKET_NAME}/{key} (404)")
            return False
        print(f"[error] downloading {key}: {e}")
        return False


def download_landsat_scene(
    scene_id: str,
    output_root: str,
    request_payer: str = "requester",
) -> Dict[str, List[str]]:
    """
    Download required files for a single Landsat 8/9 C2 L2 scene
    from the usgs-landsat requester-pays bucket.

    Parameters
    ----------
    scene_id : str
        Landsat product ID. Can include suffixes such as "_SR" from STAC items,
        e.g. "LC08_L2SP_172057_20210101_20210308_02_T1_SR".
    output_root : str
        Local directory under which a subfolder with the scene_id
        will be created.
    request_payer : str
        "requester" is required for requester-pays buckets.

    Returns
    -------
    downloaded : Dict[str, List[str]]
        Dictionary with keys "sr", "st", "qa", "aux" and values
        being lists of local file paths that were successfully downloaded.
    """
    loc = parse_scene_id(scene_id)
    s3_prefix = loc.s3_prefix  # collection02/level-2/standard/oli-tirs/...

    required = build_required_filenames(loc.scene_id)
    s3_client = get_s3_client()

    scene_local_dir = os.path.join(output_root, loc.scene_id)
    os.makedirs(scene_local_dir, exist_ok=True)

    downloaded: Dict[str, List[str]] = {"sr": [], "st": [], "qa": [], "aux": []}

    for group_name, filenames in required.items():
        for fname in filenames:
            s3_key = f"{s3_prefix}/{fname}"
            local_path = os.path.join(scene_local_dir, fname)

            ok = download_single_object(
                s3_client=s3_client,
                key=s3_key,
                local_path=local_path,
                request_payer=request_payer,
            )
            if ok:
                downloaded[group_name].append(local_path)

    return downloaded


if __name__ == "__main__":
    # Simple manual test / demo
    test_scene_id = "LC08_L2SP_172057_20210101_20210308_02_T1"
    out_dir = "./landsat_download_test"

    result = download_landsat_scene(test_scene_id, out_dir)
    print("Downloaded files:")
    for k, v in result.items():
        print(f"  {k}:")
        for p in v:
            print(f"    - {p}")
