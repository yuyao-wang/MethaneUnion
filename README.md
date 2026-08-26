# MethaneUnion

**A reproducible multi-source geospatial data pipeline and dataset for methane plume detection under incomplete satellite coverage.**

- **4 satellite sources:** Sentinel-2 · Landsat 8/9 · EMIT · Sentinel-5P
- **8,981 observable multi-sensor events** after matching and quality filtering
- **Temporal and geographic alignment** across heterogeneous sensor observations
- **Leakage-safe event-level splits** before crop and query generation

<p align="center">
  <a href="https://huggingface.co/datasets/yuyao42/MethaneUnion">Dataset</a> ·
  <a href="https://github.com/yuyao-wang/MethaneFuse">MethaneFuse</a> ·
  <a href="#data-pipeline">Data Pipeline</a> ·
  <a href="#data-quality--validation">Validation</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="Pictures/methaneunion_pipeline.png" alt="MethaneUnion dataset construction pipeline" width="100%">
</p>

## Overview

**MethaneUnion** is a partial multi-sensor satellite dataset for methane plume classification and segmentation. It is built from Carbon Mapper plume reports and matched satellite observations from Sentinel-2, Landsat 8/9, EMIT, and Sentinel-5P.

Unlike fully paired multi-sensor datasets, MethaneUnion preserves the sensor availability patterns that occur in real satellite observations. Each plume event is associated only with the subset of sensors available near the target location and time. This makes the dataset suitable for studying methane plume detection under realistic missing-sensor conditions.

MethaneUnion is the dataset and evaluation benchmark used by [MethaneFuse](https://github.com/yuyao-wang/MethaneFuse), a two-stage learning framework for methane plume detection from naturally available satellite observations.

## Data Pipeline

```text
Carbon Mapper plume events
        ↓
Source discovery and acquisition [S2 · L8/9 · EMIT · S5P]
        ↓
Temporal matching → geospatial alignment → sensor-specific quality filtering
        ↓
Event-level sensor availability → query and crop generation
        ↓
Leakage-safe temporal/geographic splits → released manifests + Python loader
```

Carbon Mapper reports provide the event location, time, and reference plume mask. Sensor observations are discovered independently, aligned to the event in space and time, filtered with sensor-specific checks, and retained without requiring every sensor to be present.

Quality control is applied before query generation and again at release time. The published manifests keep all samples derived from the same plume event in one partition, preventing event-level train/test leakage.

## Engineering Challenges

1. **Heterogeneous sensors.** Spatial resolution, revisit schedule, spectral bands, file formats, and geographic coverage differ across the four sources. The pipeline normalizes these inputs into an event-centered representation while preserving sensor-specific data.
2. **Missing observations.** MethaneUnion does not force a fully paired dataset. It records the sensors actually available for each event so models and evaluations reflect real satellite coverage.
3. **Leakage control.** One plume report can produce multiple observations, crops, and queries. Splits are assigned at the event level before derived samples are generated.

## Data Quality & Validation

The repository includes a manifest-level quality report generator and test fixtures. It audits event counts, sensor availability, required columns, coordinate/time validity, path availability, and train/test event overlap without requiring the full dataset to be downloaded.

```bash
python scripts/build_data_quality_report.py \
  --source-manifest path/to/source_events.csv \
  --release-manifest path/to/released_events.csv \
  --train-manifest path/to/train.csv \
  --test-manifest path/to/test.csv \
  --output-json artifacts/data_quality/summary.json \
  --output-markdown artifacts/data_quality/report.md \
  --fail-on-issues
```

Use `--verify-files --data-root <dataset-root>` to check non-empty manifest paths against local files. See [Data Quality Report](docs/data_quality.md) for the validation contract and [Generated Artifacts](docs/generated_artifacts.md) for repository output policy.

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

Load a released split:

```python
from methaneunion import MethaneUnionDataset

dataset = MethaneUnionDataset(
    root="data/MethaneUnion",
    split_scheme="temporal",  # use "geo" for the geo-cluster split
    split="test",
    scale_m=480,
    sensors=["S2", "L89", "EMIT", "S5P"],
)

sample = dataset[0]
print(sample["loaded_sensors"])
print(sample["observations"]["S2"]["data"]["t0"].shape)
```

Each sample provides its ID, label, coordinates, available/loaded sensors, observations, and raw manifest metadata. The loader supports extracted files and the original `dataset_part_*.tar.gz` archives.

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

MethaneUnion provides two derived views:

| Protocol | Purpose | Scales / crop sizes |
| --- | --- | --- |
| `D_pre` | Sensor-native pretraining | 32 × 32 for S2/L8/9/EMIT; 3 × 3 for S5P |
| `D_scale` | Query-level classification and segmentation | 120 m, 360 m, 480 m, 960 m |

Dense masks are generated for Sentinel-2, Landsat 8/9, and EMIT by reprojecting the Carbon Mapper plume mask. Sentinel-5P provides coarse CH₄ context and is not used for dense-mask supervision.

## Splits

MethaneUnion provides two event-level split protocols:

| Split             | Purpose                                                    |
| ----------------- | ---------------------------------------------------------- |
| Temporal split    | Main deployment-aligned evaluation protocol                |
| Geo-cluster split | Spatial robustness evaluation under held-out macro-regions |

Both split protocols are applied at the Carbon Mapper event level before crop and query construction. All observations, crops, and queries derived from the same plume report are assigned to the same partition to avoid event leakage.

## Data Format

Event-level records contain the event ID/time, coordinates, available sensors, observation paths, plume-mask path, split, and source metadata. Query-level records add the query ID/time/center/scale, classification label, segmentation-mask paths, and sensor-crop paths.

Released manifests use sensor-specific path columns:

```text
S2_{t0,pre,pre_pre,plume_label}_path
L89_{t0,pre,pre_pre,plume_label}_path
EMIT_{t0,pre,pre_pre,plume_label}_path
S5p_temporal_path
```

## Repository Structure

```text
MethaneUnion/
├── Pictures/                        # README figures and visual assets
├── configs/                         # Configuration files
├── data_downloading/                # Data acquisition utilities
├── data_preprocess/                 # Shared preprocessing code
├── docs/                             # Validation and operations documentation
├── methaneunion/                    # Minimal released-dataset loader
├── preprocess_dataset_EMIT/         # EMIT preprocessing pipeline
├── preprocess_dataset_L89/          # Landsat 8/9 preprocessing pipeline
├── preprocess_dataset_multisensor/  # Multi-sensor matching and fusion preparation
├── preprocess_dataset_query_multi/  # Query construction pipeline
├── preprocess_dataset_s2/           # Sentinel-2 preprocessing pipeline
├── preprocess_dataset_s5p/          # Sentinel-5P preprocessing pipeline
├── scripts/                          # Repository-level validation utilities
├── tests/                            # Small deterministic fixtures and smoke tests
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

The MethaneFuse paper, which introduces MethaneUnion and MethaneFuse, has been accepted for publication at the 2026 IEEE International Conference on Data Mining (ICDM 2026). If you use either resource, please cite:

```bibtex
@inproceedings{wang2026methanefuse,
  title     = {MethaneFuse: Learning from Multi-Sensor Satellite Observations for Methane Plume Detection},
  author    = {Wang, Yuyao and Leung, Juliana Y. and Niu, Di},
  booktitle = {2026 IEEE International Conference on Data Mining (ICDM)},
  year      = {2026},
  note      = {Accepted for publication}
}
```

## License

The released dataset is provided under the CC BY-NC 4.0 license.

## Contact

For questions about MethaneUnion or MethaneFuse, please open an issue or contact the repository maintainer.
