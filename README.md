# MethaneUnion

<p align="center">
  <img src="Pictures/methaneunion_pipeline.png" alt="MethaneUnion dataset construction pipeline" width="100%">
</p>

<p align="center">
  <b>An event-centered partial multi-sensor satellite dataset for methane plume detection.</b>
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/yuyao42/MethaneUnion">Dataset</a> ·
  <a href="https://github.com/yuyao-wang/MethaneFuse">MethaneFuse</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#dataset-protocols">Protocols</a> ·
  <a href="#data-format">Data Format</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Task-Methane%20Plume%20Detection-green" alt="Task">
  <img src="https://img.shields.io/badge/Sensors-S2%20%7C%20L8%2F9%20%7C%20EMIT%20%7C%20S5P-blue" alt="Sensors">
  <img src="https://img.shields.io/badge/Data-Multi--sensor%20Satellite-purple" alt="Data">
  <img src="https://img.shields.io/badge/License-CC--BY--NC--4.0-lightgrey" alt="License">
</p>

## Overview

**MethaneUnion** is a partial multi-sensor satellite dataset for methane plume classification and segmentation. It is built from Carbon Mapper plume reports and matched satellite observations from Sentinel-2, Landsat 8/9, EMIT, and Sentinel-5P.

Unlike fully paired multi-sensor datasets, MethaneUnion preserves the sensor availability patterns that occur in real satellite observations. Each plume event is associated only with the subset of sensors available near the target location and time. This makes the dataset suitable for studying methane plume detection under realistic missing-sensor conditions.

MethaneUnion is the dataset and evaluation benchmark used by [MethaneFuse](https://github.com/yuyao-wang/MethaneFuse), a two-stage learning framework for methane plume detection from naturally available satellite observations.

## Highlights

* **Event-centered benchmark** built from Carbon Mapper plume reports.
* **Partial multi-sensor observations** from Sentinel-2, Landsat 8/9, EMIT, and Sentinel-5P.
* **8,981 observable multi-sensor events** after sensor matching and quality filtering.
* **Classification and segmentation protocols** at 120 m, 360 m, 480 m, and 960 m query scales.
* **Event-level split control** to avoid leakage between train and test samples derived from the same plume report.
* **Minimal Python loader** for released split CSVs, extracted sensor files, and original archive files.

## Data Access

The processed dataset and manifests are released on Hugging Face:

```text
https://huggingface.co/datasets/yuyao42/MethaneUnion
```

A typical released layout is:

```text
data/MethaneUnion/
├── datasets/
│   ├── temporal_split/
│   │   └── 480m_GSD/
│   │       ├── train.csv
│   │       └── test.csv
│   └── geo_split/
│       └── 480m_GSD/
│           ├── train.csv
│           └── test.csv
├── data/                            # Extracted sensor files, optional
└── dataset_part_001.tar.gz          # Original archives, also supported
```

The loader supports both extracted files and the original `dataset_part_*.tar.gz` archives.

## Quick Start

Clone the repository:

```bash
git clone https://github.com/yuyao-wang/MethaneUnion.git
cd MethaneUnion
```

Load a released split:

```python
from methaneunion import MethaneUnionDataset

dataset = MethaneUnionDataset(
    root="data/MethaneUnion",
    split="test",
    scale_m=480,
    sensors=["S2", "L89", "EMIT", "S5P"],
)

sample = dataset[0]

print(sample.keys())
print(sample["loaded_sensors"])
print(sample["observations"]["S2"]["data"]["t0"].shape)
```

Each returned sample contains:

```text
id
label
latitude
longitude
available_sensors
loaded_sensors
observations
metadata
```

The `observations` field stores the loaded sensor arrays and their original relative paths. The `metadata` field stores the raw CSV row as a dictionary.

By default, the loader reads the temporal split:

```python
dataset = MethaneUnionDataset(
    root="data/MethaneUnion",
    split_scheme="temporal",
    split="test",
    scale_m=480,
)
```

To use the geo-cluster split:

```python
dataset = MethaneUnionDataset(
    root="data/MethaneUnion",
    split_scheme="geo",
    split="test",
    scale_m=480,
)
```

## Dataset Statistics

MethaneUnion expands valid Sentinel-2-only coverage from **3,211** events to **8,981** observable multi-sensor events after sensor matching and quality filtering.

| Dataset view                                | Count / Description                                                        |
| ------------------------------------------- | -------------------------------------------------------------------------- |
| Sentinel-2-only valid observations          | 3,211 events                                                               |
| MethaneUnion observable multi-sensor events | 8,981 events                                                               |
| Supported sensors                           | Sentinel-2, Landsat 8/9, EMIT, Sentinel-5P                                 |
| Label types                                 | Query-level classification labels; dense plume masks for supported sensors |
| Evaluation scales                           | 120 m, 360 m, 480 m, 960 m                                                 |

## Data Sources

| Source        | Product                                    | Role                                                |
| ------------- | ------------------------------------------ | --------------------------------------------------- |
| Carbon Mapper | Plume reports and plume masks              | Event anchors and reference plume masks             |
| Sentinel-2    | Level-2A surface reflectance               | Fine-resolution multispectral and SWIR observations |
| Landsat 8/9   | Collection 2 Level-2 surface reflectance   | Additional multispectral and SWIR observations      |
| EMIT          | Level-2A hyperspectral surface reflectance | Hyperspectral methane-sensitive observations        |
| Sentinel-5P   | Level-2 methane product                    | Coarse atmospheric CH4 context                      |

## Dataset Protocols

MethaneUnion provides two derived learning protocols: `D_pre` for sensor-native pretraining and `D_scale` for scale-controlled downstream evaluation.

### `D_pre`: Sensor-native pretraining protocol

`D_pre` uses fixed-pixel crops from each available sensor. Each crop receives a binary methane label according to whether its geographic footprint overlaps the Carbon Mapper plume mask.

| Sensor      |      Crop size |
| ----------- | -------------: |
| Sentinel-2  | 32 × 32 pixels |
| Landsat 8/9 | 32 × 32 pixels |
| EMIT        | 32 × 32 pixels |
| Sentinel-5P |   3 × 3 pixels |

This protocol is used for Stage 1 sensor-native representation learning in MethaneFuse.

### `D_scale`: Multi-scale query protocol

`D_scale` defines query-level classification and segmentation samples at controlled geographic footprints.

| Query scale | Classification | Segmentation |
| ----------: | :------------: | :----------: |
|       120 m |        ✓       |       ✓      |
|       360 m |        ✓       |       ✓      |
|       480 m |        ✓       |       ✓      |
|       960 m |        ✓       |       ✓      |

For Sentinel-2, Landsat 8/9, and EMIT, dense segmentation masks are generated by reprojecting the Carbon Mapper plume mask to the sensor grid. Sentinel-5P is used as coarse methane context and is not used for dense plume-mask supervision.

## Splits

MethaneUnion provides two event-level split protocols:

| Split             | Purpose                                                    |
| ----------------- | ---------------------------------------------------------- |
| Temporal split    | Main deployment-aligned evaluation protocol                |
| Geo-cluster split | Spatial robustness evaluation under held-out macro-regions |

Both split protocols are applied at the Carbon Mapper event level before crop and query construction. All observations, crops, and queries derived from the same plume report are assigned to the same partition to avoid event leakage.

## Data Format

Each event-level record contains:

```text
event_id
event_time
latitude
longitude
available_sensors
sensor_observation_paths
plume_mask_path
split
metadata
```

Each query-level record contains:

```text
query_id
event_id
query_time
query_center
query_scale_m
available_sensors
classification_label
segmentation_mask_paths
sensor_crop_paths
split
```

The released CSV manifests include sensor-specific paths such as:

```text
S2_t0_path
S2_pre_path
S2_pre_pre_path
S2_plume_label_path

L89_t0_path
L89_pre_path
L89_pre_pre_path
L89_plume_label_path

EMIT_t0_path
EMIT_pre_path
EMIT_pre_pre_path
EMIT_plume_label_path

S5p_temporal_path
```

## Repository Structure

```text
MethaneUnion/
├── Pictures/                        # README figures and visual assets
├── configs/                         # Configuration files
├── data_downloading/                # Data acquisition utilities
├── data_preprocess/                 # Shared preprocessing code
├── methaneunion/                    # Minimal released-dataset loader
├── preprocess_dataset_EMIT/         # EMIT preprocessing pipeline
├── preprocess_dataset_L89/          # Landsat 8/9 preprocessing pipeline
├── preprocess_dataset_multisensor/  # Multi-sensor matching and fusion preparation
├── preprocess_dataset_query_multi/  # Query construction pipeline
├── preprocess_dataset_s2/           # Sentinel-2 preprocessing pipeline
├── preprocess_dataset_s5p/          # Sentinel-5P preprocessing pipeline
├── train/                           # Training code and model components
└── util/                            # Utility functions
```

## Relation to MethaneFuse

MethaneUnion provides the dataset and evaluation protocols used by MethaneFuse.

* **MethaneUnion**: dataset construction, released manifests, sensor files, and evaluation protocols.
* **MethaneFuse**: model implementation, training framework, baselines, and evaluation code.

MethaneFuse repository:

```text
https://github.com/yuyao-wang/MethaneFuse
```

## Citation

If you use MethaneUnion or MethaneFuse, please cite:

```bibtex
@misc{wang2026methaneunion,
  title  = {MethaneUnion: An Event-Centered Partial Multi-Sensor Satellite Dataset for Methane Plume Detection},
  author = {Wang, Yuyao},
  year   = {2026},
  note   = {Dataset and code release}
}
```

## License

The released dataset is provided under the CC BY-NC 4.0 license.

## Contact

For questions about MethaneUnion or MethaneFuse, please open an issue or contact the repository maintainer.
