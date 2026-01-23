"""Append plume/con mask availability flags to a Carbon Mapper CSV."""

from __future__ import annotations

import csv
from pathlib import Path

INPUT_CSV = Path("carbon_mapper_data/csvs/merged_file_with_s2.csv")
OUTPUT_CSV = Path("carbon_mapper_data/csvs/merged_file_with_s2_with_mask_flags.csv")
MASK_ROOT = Path("carbon_mapper_data_masks")


def main() -> None:
    fieldnames: list[str]

    with INPUT_CSV.open(newline="", encoding="utf-8") as src, OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError(f"Input file {INPUT_CSV} is missing a header row.")
        fieldnames = list(reader.fieldnames)
        fieldnames.extend(["plume_tif_exists", "con_tif_exists"])
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            plume_id = (row.get("plume_id") or "").strip()
            plume_dir = MASK_ROOT / plume_id if plume_id else None
            plume_exists = (
                Path(plume_dir, "plume.tif").is_file() if plume_dir else False
            )
            con_exists = (
                Path(plume_dir, "con.tif").is_file() if plume_dir else False
            )
            row["plume_tif_exists"] = "1" if plume_exists else "0"
            row["con_tif_exists"] = "1" if con_exists else "0"
            writer.writerow(row)


if __name__ == "__main__":
    main()
