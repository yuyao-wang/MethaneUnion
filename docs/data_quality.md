# Data Quality Report

MethaneUnion's quality report is generated from source, release, and split manifests. It is intended to make dataset construction and leakage controls auditable without requiring the full raster archive.

## Validation contract

The generator reports:

- source and released row/event counts;
- manifest SHA-256 values and Git commit provenance;
- sensor availability counts and missing-sensor combinations;
- missing or invalid event IDs, coordinates, and datetimes;
- required-column failures;
- train/test event overlap;
- optional existence checks for non-empty `*_path` fields.

An event identifier must be explicit. The generator auto-detects `event_id` or `plume_id`; use `--event-column` for another schema. It deliberately does not assume that a sample-level `id` is an event identifier.

## Run a metadata-only audit

```bash
python scripts/build_data_quality_report.py \
  --source-manifest path/to/source_events.csv \
  --release-manifest path/to/released_events.csv \
  --train-manifest path/to/train.csv \
  --test-manifest path/to/test.csv \
  --event-column plume_id \
  --required-column latitude \
  --required-column longitude \
  --expected-release-events 8981 \
  --output-json artifacts/data_quality/summary.json \
  --output-markdown artifacts/data_quality/report.md \
  --fail-on-issues
```

`--fail-on-issues` returns a non-zero exit code for missing required fields, invalid event metadata, unexpected event counts, missing files during file verification, or train/test event overlap.

## Verify extracted files

```bash
python scripts/build_data_quality_report.py \
  --release-manifest path/to/released_events.csv \
  --event-column plume_id \
  --verify-files \
  --data-root path/to/MethaneUnion \
  --output-json artifacts/data_quality/files.json \
  --output-markdown artifacts/data_quality/files.md \
  --fail-on-issues
```

File verification checks extracted paths. Archive-backed releases should either be extracted first or audited with an archive-aware follow-up validator.

## Release criteria

A release report is acceptable only when:

- the declared event count matches the manifest;
- required columns are present;
- event identifiers, coordinates, and datetimes are valid;
- train/test event overlap is zero;
- any missing files are explained or resolved;
- the report records the input hashes and Git commit.

The published facts of 3,211 valid Sentinel-2-only observations and 8,981 observable multi-sensor events should be re-asserted against the exact release manifests rather than hard-coded into the generator.
