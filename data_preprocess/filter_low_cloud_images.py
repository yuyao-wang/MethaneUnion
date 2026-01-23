"""Filter high-cloud S2/L8 images from merged records and summarize counts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CSV_PATH = Path("../carbon_mapper_data/csvs/merged_file_with_s2_l8_filtered_with_flags.csv")
OUTPUT_PATH = CSV_PATH.with_name(
    CSV_PATH.stem + "_low_cloud_only" + CSV_PATH.suffix
)
CLOUD_THRESHOLD = 20.0

S2_COLUMNS = {
    idx: {
        "datetime": f"s2_{idx}_datetime",
        "path": f"s2_{idx}_path",
        "cloud": f"s2_{idx}_cloud_cover",
    }
    for idx in range(1, 4)
}
L8_COLUMNS = {
    idx: {
        "scene": f"l8_{idx}_scene_id",
        "datetime": f"l8_{idx}_datetime",
        "tif": f"l8_{idx}_tif",
        "sun_az": f"l8_{idx}_sun_azimuth",
        "sun_el": f"l8_{idx}_sun_elevation",
        "cloud": f"l8_{idx}_cloud_cover",
    }
    for idx in range(1, 4)
}


def _parse_clouds(df: pd.DataFrame, col_names: list[str]) -> None:
    for col in col_names:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _invalidate_rows(df: pd.DataFrame, columns: dict[int, dict[str, str]], mask) -> None:
    for idx, mapping in columns.items():
        for col in mapping.values():
            if col in df.columns:
                df.loc[mask[idx], col] = pd.NA


def _validity_masks(df: pd.DataFrame, columns: dict[int, dict[str, str]]):
    masks = {}
    counts = []
    for idx, mapping in columns.items():
        cloud_col = mapping["cloud"]
        reference_cols = [col for key, col in mapping.items() if key != "cloud" and col in df.columns]
        has_reference = pd.Series(False, index=df.index)
        for col in reference_cols:
            has_reference |= df[col].notna()
        clouds = df[cloud_col] if cloud_col in df.columns else pd.Series(pd.NA, index=df.index)
        mask = has_reference & (~clouds.notna() | (clouds <= CLOUD_THRESHOLD))
        masks[idx] = ~mask
        counts.append(mask.astype(int))
    total = sum(counts) if counts else pd.Series(0, index=df.index)
    return masks, total


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    _parse_clouds(df, [mapping["cloud"] for mapping in S2_COLUMNS.values()])
    _parse_clouds(df, [mapping["cloud"] for mapping in L8_COLUMNS.values()])

    s2_invalid_masks, s2_counts = _validity_masks(df, S2_COLUMNS)
    l8_invalid_masks, l8_counts = _validity_masks(df, L8_COLUMNS)

    _invalidate_rows(df, S2_COLUMNS, s2_invalid_masks)
    _invalidate_rows(df, L8_COLUMNS, l8_invalid_masks)

    df["remaining_s2_count"] = s2_counts
    df["remaining_l8_count"] = l8_counts
    df["remaining_image_count"] = s2_counts + l8_counts

    filtered_df = df[df["remaining_image_count"] > 0].copy()
    filtered_df.to_csv(OUTPUT_PATH, index=False)

    total_items = len(filtered_df)
    count_distribution = filtered_df["remaining_image_count"].value_counts().sort_index()

    print(f"Output file: {OUTPUT_PATH}")
    print(s2_counts.describe())
    print(l8_counts.describe())
    print(f"Remaining items: {total_items}")
    for n in range(1, 4):
        print(
            f"Items with {n} valid image(s): {count_distribution.get(n, 0)}"
        )


if __name__ == "__main__":
    main()
