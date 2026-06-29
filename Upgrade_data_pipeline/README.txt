Upgrade_data_pipeline active tables:

1. csv/carbon_mapper_plumes_20160101_20260530_with_plume_tif.csv
   Carbon Mapper plume catalogue through 2026-05-30, filtered to plume_tif non-empty.

2. csv/carbon_mapper_plumes_20160101_20260530_with_t0_flags.csv
   Same plume rows with t0 metadata flags/times for S2, L89, EMIT, and S5P.
   Use has_any_t0 == True to get the downloadable/usable t0 candidate set.

Other intermediate/partial CSVs were moved to _deleted_csv_archive_20260624 for review/recovery and should not be used as active pipeline inputs.
