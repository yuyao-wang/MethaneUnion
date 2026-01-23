"""
Utilities for sampling Sentinel-2 chips from the exports in ``data_download``.

The workflow implemented here follows these steps:

1. Locate GeoTIFF exports and read their affine transform / CRS metadata.
2. Lay out chip centres on a grid that safely fits inside the raster footprint.
3. For each candidate centre, extract a 512x512 window from every available
   monthly composite that belongs to the same MGRS tile (handled per scene).
4. Validate the patch (e.g. reject when more than 20% of pixels are zero or
   nodata) and persist the stacked chip as a new GeoTIFF.
5. Record bookkeeping information in ``chips.csv`` so that subsequent runs can
   skip chips that have already been generated.

This module deliberately separates the major pieces of logic so that the script
can be reused from notebooks or other batch pipelines.

The implementation depends on ``rasterio`` and ``numpy``.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window

# ---------------------------------------------------------------------------
# Configuration defaults

DATA_DOWNLOAD_DIR = Path("/data/ruoyu/original_data/monthly_l2a_cog")
OUTPUT_DIR = Path("../data/pretrain/data_download_chips_32_1000")
METADATA_CSV = Path("../data_csv/chips_32.csv")

PATCH_SIZE = 32
ZERO_FRACTION_THRESHOLD = 0.20  # Reject chips where zero/nodata > 20%
SAMPLES_PER_GRAPH = 1000  # max chips to save per input raster (make sure around 50K chips in total)
PIXEL_STRIDE = 32  # stride (in pixels) for grid sampling
RASTER_ZERO_THRESHOLD = 0.80  # Skip rasters with >80% zero/nodata pixels (approx.)
RASTER_QUALITY_DOWNSCALE = 32  # Downscale factor when estimating raster quality
SAMPLING_SEED = 42

# Sentinel-2 monthly exports follow the pattern s2_l2a_{tile}_{yearmonth}.tif
TILE_FILENAME_PATTERN = re.compile(
    r"^s2_[a-z0-9]+_(?P<tile>[0-9A-Z]{5})_(?P<date>\d{6})\.tif$",
    re.IGNORECASE,
)

CHIP_FILENAME_PATTERN = re.compile(
    r"^(?P<tile>[0-9A-Z]{5})_(?P<date>\d{6})_",
    re.IGNORECASE,
)

CSV_FIELDS = [
    "chip_path",
    "source_paths",
    "tile",
    "time_keys",
    "center_lon",
    "center_lat",
    "zero_fraction",
    "cloud_percentages",
    "notes",
]


# ---------------------------------------------------------------------------
# Dataclasses and helpers

@dataclass(frozen=True)
class SampledWindow:
    """Representation of a sampled chip window."""

    center_row: int
    center_col: int
    window: Window
    transform: Affine
    center_lon: float
    center_lat: float


@dataclass
class PatchResult:
    """Holds the data and metadata for a successfully extracted chip."""

    array: np.ndarray  # shape (time, bands, H, W)
    window: SampledWindow
    zero_fraction: float
    cloud_percentages: list[Optional[float]]
    time_keys: list[str]
    source_paths: list[Path]


# ---------------------------------------------------------------------------
# Utility functions

def list_source_files(source_dir: Path = DATA_DOWNLOAD_DIR) -> list[Path]:
    """Return all GeoTIFF exports found under ``source_dir``."""
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    return sorted(path for path in source_dir.glob("*.tif") if path.is_file())


def group_by_tile(paths: Iterable[Path]) -> dict[str, list[Tuple[str, Path]]]:
    """Group source rasters by MGRS tile code."""
    groups: dict[str, list[Tuple[str, Path]]] = {}
    for path in paths:
        match = TILE_FILENAME_PATTERN.match(path.name)
        if not match:
            logging.warning("Skipping file with unexpected name: %s", path)
            continue
        tile = match.group("tile")
        date_key = match.group("date")
        groups.setdefault(tile, []).append((date_key, path))
    for tile, items in groups.items():
        items.sort(key=lambda item: item[0])
    return groups


def load_existing_records(csv_path: Path = METADATA_CSV) -> dict[str, dict[str, str]]:
    """Load existing metadata and index by chip path."""
    if not csv_path.exists():
        return {}
    records: dict[str, dict[str, str]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chip_path = row.get("chip_path")
            if chip_path:
                records[chip_path] = row
    return records


def initialise_csv(csv_path: Path = METADATA_CSV) -> None:
    """Ensure the metadata CSV exists with the correct header."""
    if csv_path.exists():
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()


def load_existing_chip_dates(output_dir: Path = OUTPUT_DIR) -> dict[str, set[str]]:
    """
    Scan ``output_dir`` for previously generated chips and index by tile/date.

    Returns a mapping of ``tile -> set(time_key)`` so that reruns can skip
    rasters that already have chips persisted on disk.
    """
    chip_index: dict[str, set[str]] = {}
    if not output_dir.exists():
        return chip_index

    for tile_dir in output_dir.iterdir():
        if not tile_dir.is_dir():
            continue
        for chip_file in tile_dir.glob("*.tif"):
            match = CHIP_FILENAME_PATTERN.match(chip_file.name)
            if not match:
                logging.debug("Ignoring chip with unexpected name: %s", chip_file)
                continue
            tile = match.group("tile")
            date_key = match.group("date")
            chip_index.setdefault(tile, set()).add(date_key)
    return chip_index

def build_sampled_window(
    dataset: rasterio.io.DatasetReader,
    center_row: int,
    center_col: int,
    patch_size: int = PATCH_SIZE,
) -> Optional[SampledWindow]:
    """Construct a ``SampledWindow`` if the window fits entirely within the raster."""
    half = patch_size // 2
    row_off = center_row - half
    col_off = center_col - half
    if row_off < 0 or col_off < 0:
        return None
    if row_off + patch_size > dataset.height or col_off + patch_size > dataset.width:
        return None
    window = Window(col_off, row_off, patch_size, patch_size)
    transform = dataset.window_transform(window)
    center_lon, center_lat = dataset.transform * (center_col + 0.5, center_row + 0.5)

    return SampledWindow(
        center_row=center_row,
        center_col=center_col,
        window=window,
        transform=transform,
        center_lon=center_lon,
        center_lat=center_lat,
    )


def compute_zero_fraction(
    patch: np.ndarray,
    nodata: Optional[float],
) -> float:
    """Calculate the fraction of zero / nodata pixels in ``patch``."""
    if patch.size == 0:
        return 1.0
    mask_zero = patch == 0
    if nodata is not None:
        mask_zero |= patch == nodata
    return float(mask_zero.sum() / patch.size)


def fetch_cloud_percentage(dataset: rasterio.io.DatasetReader) -> Optional[float]:
    """Try to retrieve cloudiness metadata embedded in the GeoTIFF tags."""
    for key in ("CLOUDY_PIXEL_PERCENTAGE", "MEAN_CLOUD_PROBABILITY"):
        value = dataset.tags().get(key)
        if value is None:
            continue
        try:
            return float(value)
        except ValueError:
            logging.debug(
                "Failed to parse %s metadata value '%s' in %s",
                key,
                value,
                dataset.name,
            )
    return None


def extract_time_series_patch(
    stack_paths: Sequence[Path],
    sampled_window: SampledWindow,
) -> Optional[PatchResult]:
    """
    Extract the time-stack patch across ``stack_paths`` for the provided window.

    The returned array has shape (time, bands, height, width). If any source
    raster fails the quality checks the entire chip is rejected.
    """
    arrays: list[np.ndarray] = []
    zero_fractions: list[float] = []
    cloud_percentages: list[Optional[float]] = []
    time_keys: list[str] = []

    for path in stack_paths:
        match = TILE_FILENAME_PATTERN.match(path.name)
        time_keys.append(match.group("date") if match else path.stem)
        with rasterio.open(path) as src:
            window = sampled_window.window
            patch = src.read(window=window)
            if patch.shape[1] != PATCH_SIZE or patch.shape[2] != PATCH_SIZE:
                logging.debug(
                    "Rejecting window %s in %s due to unexpected shape %s",
                    window,
                    path,
                    patch.shape,
                )
                return None
            zero_fraction = compute_zero_fraction(patch, src.nodata)
            zero_fractions.append(zero_fraction)
            cloud_percentages.append(fetch_cloud_percentage(src))
            arrays.append(patch)

    worst_zero_fraction = max(zero_fractions) if zero_fractions else 1.0
    if worst_zero_fraction > ZERO_FRACTION_THRESHOLD:
        logging.debug(
            "Rejecting chip at (%s, %s); zero_fraction %.3f exceeds threshold %.2f",
            sampled_window.center_row,
            sampled_window.center_col,
            worst_zero_fraction,
            ZERO_FRACTION_THRESHOLD,
        )
        return None

    stacked = np.stack(arrays, axis=0)
    return PatchResult(
        array=stacked,
        window=sampled_window,
        zero_fraction=worst_zero_fraction,
        cloud_percentages=cloud_percentages,
        time_keys=time_keys,
        source_paths=list(stack_paths),
    )


def save_patch(
    result: PatchResult,
    output_path: Path,
) -> None:
    """Persist the patch to disk as a GeoTIFF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten (time, bands, H, W) -> (time * bands, H, W) for GeoTIFF storage.
    time_dim, band_dim, height, width = result.array.shape
    reshaped = result.array.reshape(time_dim * band_dim, height, width)

    with rasterio.open(result.source_paths[0]) as ref_src:
        meta = ref_src.meta.copy()

    meta.update(
        {
            "width": width,
            "height": height,
            "count": reshaped.shape[0],
            "transform": result.window.transform,
        }
    )

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(reshaped)


def format_chip_filename(
    tile: str,
    time_keys: Sequence[str],
    lon: float,
    lat: float,
    cloud_percentages: Sequence[Optional[float]],
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Generate an informative filename for the chip."""
    time_token = time_keys[0] if time_keys else "unknown"
    cloud_values = [cp for cp in cloud_percentages if cp is not None]
    if cloud_values:
        cloud_token = f"cloud{np.mean(cloud_values):.1f}"
    else:
        cloud_token = None
    filename_parts = [
        tile,
        time_token,
        f"{lat:.5f}",
        f"{lon:.5f}",
    ]
    if cloud_token:
        filename_parts.append(cloud_token)
    filename = "_".join(filename_parts) + ".tif"
    return output_dir / tile / filename


def write_metadata_row(record: dict[str, str], csv_path: Path = METADATA_CSV) -> None:
    """Append a metadata record to ``chips.csv``."""
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(record)


def chip_exists(
    existing_records: dict[str, dict[str, str]],
    chip_path: Path,
) -> bool:
    """Check whether ``chip_path`` is already listed in metadata."""
    return str(chip_path) in existing_records

def _axis_positions(
    size: int,
    *,
    patch_size: int = PATCH_SIZE,
    stride: int = PIXEL_STRIDE,
) -> list[int]:
    """Generate centre indices along a single axis, covering the full extent."""
    half = patch_size // 2
    if size <= patch_size:
        raise ValueError("Raster dimensions are smaller than the patch size.")

    start = half
    stop = size - half

    positions = list(range(start, stop, stride)) if stride > 0 else [start]
    if not positions or positions[-1] != stop:
        positions.append(stop)
    return positions


def generate_grid_centers(
    dataset: rasterio.io.DatasetReader,
    *,
    patch_size: int = PATCH_SIZE,
    stride: int = PIXEL_STRIDE,
    rng: Optional[np.random.Generator] = None,
) -> list[Tuple[int, int]]:
    """Return shuffled (row, col) centres covering the raster."""
    rows = _axis_positions(dataset.height, patch_size=patch_size, stride=stride)
    cols = _axis_positions(dataset.width, patch_size=patch_size, stride=stride)
    centres = [(row, col) for row in rows for col in cols]
    rng = rng or np.random.default_rng(SAMPLING_SEED)
    rng.shuffle(centres)
    return centres


def estimate_raster_zero_fraction(
    path: Path,
    *,
    downscale: int = RASTER_QUALITY_DOWNSCALE,
) -> float:
    """Approximate zero/nodata fraction for the whole raster using a coarse grid."""
    try:
        with rasterio.open(path) as src:
            rows = max(1, src.height // downscale)
            cols = max(1, src.width // downscale)
            sample = src.read(1, out_shape=(rows, cols), resampling=Resampling.nearest)
            if sample.size == 0:
                return 1.0
            mask_zero = sample == 0
            if src.nodata is not None:
                mask_zero |= sample == src.nodata
            return float(mask_zero.sum() / sample.size)
    except rasterio.errors.RasterioError as exc:
        logging.warning("Failed to assess raster quality for %s: %s", path, exc)
        return 1.0


def process_tile_stack(
    tile: str,
    items: Sequence[Tuple[str, Path]],
    *,
    samples_per_graph: int = SAMPLES_PER_GRAPH,
    pixel_stride: int = PIXEL_STRIDE,
    raster_zero_threshold: float = RASTER_ZERO_THRESHOLD,
    quality_downscale: int = RASTER_QUALITY_DOWNSCALE,
    metadata_csv: Path = METADATA_CSV,
    existing_records: Optional[dict[str, dict[str, str]]] = None,
    existing_chip_dates: Optional[dict[str, set[str]]] = None,
) -> None:
    """
    Sample and persist chips for a single tile stack.

    Parameters
    ----------
    tile:
        MGRS tile identifier (e.g. ``13REQ``).
    items:
        Sequence of ``(time_key, Path)`` sorted by time.
    samples_per_graph:
        Maximum number of chips to save per individual raster.
    pixel_stride:
        Step size (in pixels) between neighbouring grid centres.
    raster_zero_threshold:
        Maximum acceptable zero/nodata fraction (coarse estimate) for input rasters.
    quality_downscale:
        Downscale factor when estimating raster quality.
    metadata_csv:
        Path to metadata CSV for recording generated chips.
    existing_records:
        Optional metadata dictionary to avoid duplicate chips.
    """
    if not items:
        return

    existing_records = existing_records or {}
    existing_chip_dates = existing_chip_dates or {}

    quality_items: list[Tuple[str, Path]] = []
    for date_key, path in items:
        zero_fraction = estimate_raster_zero_fraction(path, downscale=quality_downscale)
        if zero_fraction > raster_zero_threshold:
            logging.info(
                "Tile %s: skipping %s due to high zero fraction %.3f (threshold %.2f)",
                tile,
                path.name,
                zero_fraction,
                raster_zero_threshold,
            )
            continue
        quality_items.append((date_key, path))

    if not quality_items:
        logging.warning("Tile %s: no rasters passed the quality filter", tile)
        return

    total_successes = 0

    for _date_key, path in quality_items:
        tile_dates = existing_chip_dates.get(tile, set())
        if _date_key in tile_dates:
            logging.info(
                "Tile %s: skipping %s because chips already exist for date %s",
                tile,
                path.name,
                _date_key,
            )
            continue

        try:
            dataset = rasterio.open(path)
        except rasterio.errors.RasterioIOError as exc:
            logging.error("Unable to open %s: %s", path, exc)
            continue

        with dataset:
            image_successes = 0
            try:
                centres = generate_grid_centers(
                    dataset,
                    patch_size=PATCH_SIZE,
                    stride=pixel_stride,
                )
            except ValueError:
                logging.warning(
                    "Tile %s: raster %s is too small for the requested patch size.",
                    tile,
                    path.name,
                )
                continue

            for row, col in centres:
                if image_successes >= samples_per_graph:
                    break
                sampled = build_sampled_window(dataset, row, col)
                if sampled is None:
                    continue
                result = extract_time_series_patch([path], sampled)
                if result is None:
                    continue

                chip_path = format_chip_filename(
                    tile,
                    result.time_keys,
                    result.window.center_lon,
                    result.window.center_lat,
                    result.cloud_percentages,
                )
                if chip_exists(existing_records, chip_path):
                    logging.info("Chip already recorded, skipping: %s", chip_path)
                    continue

                save_patch(result, chip_path)

                record = {
                    "chip_path": str(chip_path),
                    "source_paths": json.dumps([str(p) for p in result.source_paths]),
                    "tile": tile,
                    "time_keys": json.dumps(result.time_keys),
                    "center_lon": f"{result.window.center_lon:.8f}",
                    "center_lat": f"{result.window.center_lat:.8f}",
                    "zero_fraction": f"{result.zero_fraction:.4f}",
                    "cloud_percentages": json.dumps(result.cloud_percentages),
                    "notes": "",
                }
                write_metadata_row(record, csv_path=metadata_csv)
                existing_records[str(chip_path)] = record
                image_successes += 1
                total_successes += 1
                for time_key in result.time_keys:
                    existing_chip_dates.setdefault(tile, set()).add(time_key)
                logging.info(
                    "Tile %s: %s saved chip %d/%d -> %s",
                    tile,
                    path.name,
                    image_successes,
                    samples_per_graph,
                    chip_path,
                )

    logging.info(
        "Tile %s: finished with %d chips across %d rasters",
        tile,
        total_successes,
        len(quality_items),
    )


def build_all_chips(
    *,
    source_dir: Path = DATA_DOWNLOAD_DIR,
    output_dir: Path = OUTPUT_DIR,
    metadata_csv: Path = METADATA_CSV,
    samples_per_graph: int = SAMPLES_PER_GRAPH,
    pixel_stride: int = PIXEL_STRIDE,
    raster_zero_threshold: float = RASTER_ZERO_THRESHOLD,
    quality_downscale: int = RASTER_QUALITY_DOWNSCALE,
) -> None:
    """
    Entry point for batch chip extraction across all tiles.

    This scans ``source_dir`` for GeoTIFF exports, groups them by tile code and
    iterates through each stack to generate new chips. Existing entries in
    ``metadata_csv`` are honoured to prevent duplication.

    Parameters
    ----------
    samples_per_graph:
        Maximum number of chips to save per individual raster.
    pixel_stride:
        Step size (in pixels) between neighbouring grid centres.
    raster_zero_threshold:
        Maximum acceptable zero/nodata fraction (coarse estimate) for input rasters.
    quality_downscale:
        Downscale factor when estimating raster quality.
    """
    logging.info("Scanning %s for GeoTIFF exports", source_dir)
    paths = list_source_files(source_dir)
    grouped = group_by_tile(paths)
    initialise_csv(metadata_csv)
    existing_records = load_existing_records(metadata_csv)
    existing_chip_dates = load_existing_chip_dates(output_dir)

    for tile, items in grouped.items():
        logging.info("Processing tile %s with %d time slices", tile, len(items))
        process_tile_stack(
            tile,
            items,
            samples_per_graph=samples_per_graph,
            pixel_stride=pixel_stride,
            raster_zero_threshold=raster_zero_threshold,
            quality_downscale=quality_downscale,
            metadata_csv=metadata_csv,
            existing_records=existing_records,
            existing_chip_dates=existing_chip_dates,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_all_chips()
