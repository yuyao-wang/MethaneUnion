from __future__ import annotations

import csv
import io
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from torch.utils.data import Dataset


_SPLIT_DIRS = {
    "temporal": "temporal_split",
    "geo": "geo_split",
}

_SCALE_DIRS = {
    120: "120m_GSD",
    360: "360m_GSD",
    480: "480m_GSD",
    960: "960m_GSD",
    "original": "original_scale",
    "original_scale": "original_scale",
}


@dataclass(frozen=True)
class _SensorColumns:
    paths: dict[str, str]


_SENSOR_COLUMNS: dict[str, _SensorColumns] = {
    "S2": _SensorColumns(
        {
            "t0": "S2_t0_path",
            "pre": "S2_pre_path",
            "pre_pre": "S2_pre_pre_path",
            "plume_mask": "S2_plume_label_path",
        }
    ),
    "L89": _SensorColumns(
        {
            "t0": "L89_t0_path",
            "pre": "L89_pre_path",
            "pre_pre": "L89_pre_pre_path",
            "plume_mask": "L89_plume_label_path",
        }
    ),
    "EMIT": _SensorColumns(
        {
            "t0": "EMIT_t0_path",
            "pre": "EMIT_pre_path",
            "pre_pre": "EMIT_pre_pre_path",
            "plume_mask": "EMIT_plume_label_path",
        }
    ),
    "S5P": _SensorColumns({"temporal": "S5p_temporal_path"}),
}


class MethaneUnionDataset(Dataset):
    """Minimal loader for released MethaneUnion manifests and sample files."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        scale_m: int | str,
        sensors: list[str] | tuple[str, ...] | None = None,
        split_scheme: str = "temporal",
        load_arrays: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.scale_m = scale_m
        self.split_scheme = split_scheme
        self.sensors = self._normalize_sensors(sensors)
        self.load_arrays = load_arrays
        self._archive_index: dict[str, Path] | None = None

        manifest_path = self._resolve_manifest_path()
        self.rows = self._read_manifest(manifest_path)
        self._validate_columns()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        available_sensors = self._parse_available_sensors(row["available_sensor"])
        sensors_to_load = [sensor for sensor in self.sensors if sensor in available_sensors]

        observations: dict[str, Any] = {}
        for sensor in sensors_to_load:
            observations[sensor] = self._load_sensor_observation(sensor, row)

        return {
            "id": int(row["id"]),
            "label": int(row["label"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "available_sensors": available_sensors,
            "loaded_sensors": sensors_to_load,
            "observations": observations,
            "metadata": dict(row),
        }

    def _resolve_manifest_path(self) -> Path:
        try:
            split_dir = _SPLIT_DIRS[self.split_scheme]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported split_scheme={self.split_scheme!r}. "
                f"Expected one of {sorted(_SPLIT_DIRS)}."
            ) from exc

        scale_dir = self._normalize_scale(self.scale_m)
        manifest_path = self.root / "datasets" / split_dir / scale_dir / f"{self.split}.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        return manifest_path

    def _normalize_scale(self, scale_m: int | str) -> str:
        try:
            return _SCALE_DIRS[scale_m]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported scale_m={scale_m!r}. Expected one of {list(_SCALE_DIRS)}."
            ) from exc

    def _normalize_sensors(
        self, sensors: list[str] | tuple[str, ...] | None
    ) -> list[str]:
        if sensors is None:
            return list(_SENSOR_COLUMNS)

        normalized = []
        for sensor in sensors:
            if sensor not in _SENSOR_COLUMNS:
                raise ValueError(
                    f"Unsupported sensor={sensor!r}. Expected one of {sorted(_SENSOR_COLUMNS)}."
                )
            normalized.append(sensor)
        return normalized

    def _read_manifest(self, manifest_path: Path) -> list[dict[str, str]]:
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _validate_columns(self) -> None:
        if not self.rows:
            return

        required = {"id", "label", "latitude", "longitude", "available_sensor"}
        for sensor_columns in _SENSOR_COLUMNS.values():
            required.update(sensor_columns.paths.values())

        missing = required.difference(self.rows[0].keys())
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Manifest is missing required columns: {missing_list}")

    def _parse_available_sensors(self, value: str) -> list[str]:
        return [sensor.strip() for sensor in value.split(",") if sensor.strip()]

    def _load_sensor_observation(self, sensor: str, row: dict[str, str]) -> dict[str, Any]:
        paths: dict[str, str] = {}
        data: dict[str, Any] = {}

        for logical_name, column_name in _SENSOR_COLUMNS[sensor].paths.items():
            relative_path = row.get(column_name, "").strip()
            if not relative_path:
                continue
            paths[logical_name] = relative_path
            if self.load_arrays:
                data[logical_name] = self._load_data_file(relative_path)

        sensor_sample = {
            "sensor": sensor,
            "paths": paths,
        }
        if self.load_arrays:
            sensor_sample["data"] = data
        return sensor_sample

    def _load_data_file(self, relative_path: str) -> Any:
        extracted_path = self.root / relative_path
        suffix = Path(relative_path).suffix.lower()

        if extracted_path.exists():
            if suffix == ".tif":
                return self._read_tiff_from_path(extracted_path)
            if suffix == ".npz":
                return self._read_npz_from_bytes(extracted_path.read_bytes())
            raise ValueError(f"Unsupported file type for {relative_path!r}")

        archive_path = self._resolve_archive_member(relative_path)
        with tarfile.open(archive_path, "r:gz") as archive:
            member = archive.getmember(relative_path)
            handle = archive.extractfile(member)
            if handle is None:
                raise FileNotFoundError(
                    f"Archive member could not be opened: {relative_path} in {archive_path}"
                )
            payload = handle.read()

        if suffix == ".tif":
            return self._read_tiff_from_bytes(payload)
        if suffix == ".npz":
            return self._read_npz_from_bytes(payload)
        raise ValueError(f"Unsupported file type for {relative_path!r}")

    def _read_tiff_from_path(self, path: Path) -> np.ndarray:
        with rasterio.open(path) as dataset:
            return dataset.read()

    def _read_tiff_from_bytes(self, payload: bytes) -> np.ndarray:
        with MemoryFile(payload) as memfile:
            with memfile.open() as dataset:
                return dataset.read()

    def _read_npz_from_bytes(self, payload: bytes) -> dict[str, np.ndarray]:
        with np.load(io.BytesIO(payload)) as data:
            return {key: data[key] for key in data.files}

    def _resolve_archive_member(self, relative_path: str) -> Path:
        if self._archive_index is None:
            self._archive_index = self._build_archive_index()

        try:
            return self._archive_index[relative_path]
        except KeyError as exc:
            raise FileNotFoundError(
                f"Could not find {relative_path!r} under {self.root} as an extracted file "
                "or inside dataset_part_*.tar.gz."
            ) from exc

    def _build_archive_index(self) -> dict[str, Path]:
        index: dict[str, Path] = {}
        archive_paths = sorted(self.root.glob("dataset_part_*.tar.gz"))
        if not archive_paths:
            raise FileNotFoundError(
                f"No extracted files or dataset_part_*.tar.gz archives were found under {self.root}"
            )

        for archive_path in archive_paths:
            with tarfile.open(archive_path, "r:gz") as archive:
                for member in archive.getmembers():
                    if member.isfile():
                        index.setdefault(member.name, archive_path)
        return index
