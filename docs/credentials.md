# Credential handling

MethaneUnion uses Google Earth Engine (GEE) and the Copernicus Data Space Ecosystem (CDSE) as data-access services. Credentials are local runtime inputs and must not be committed to this repository.

## CDSE

Provide CDSE account credentials through environment variables:

```bash
export CDSE_USERNAME0="..."
export CDSE_PASSWORD0="..."
```

Downloaders that support a credential pool accept additional matching pairs such as `CDSE_USERNAME1` and `CDSE_PASSWORD1`. Use `data_preprocess/configs/carbon_mapper_sentinel2_plume_download.example.yaml` for non-secret path and proxy settings. Its local counterpart without `.example` is ignored by Git.

## GEE

Use the existing local Earth Engine authentication available in the runtime environment. Keep generated tokens and machine-specific authentication files outside this repository. MethaneUnion does not require Google Cloud Storage, BigQuery, Compute Engine, AWS, or another paid cloud service for the supported GEE/CDSE workflow.

The legacy paths `data_downloading/credentials.json` and `data_downloading/token.json` remain ignored so an existing local setup is not accidentally committed.

## Existing Git history

Earlier commits contained non-empty credential and token files. Removing values from the current tree does not remove them from existing Git history. Account-side credential changes and any history rewrite are separate operations and must be reviewed before they are performed.

Never paste credential values into issues, logs, test fixtures, or generated quality reports.
