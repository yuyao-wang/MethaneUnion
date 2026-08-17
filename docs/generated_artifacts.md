# Generated artifact policy

Pipeline outputs, resumable manifests, logs, and visual QA files are runtime artifacts. They are not source code and should not be committed by default.

## A100 inventory snapshot

Snapshot date: 2026-08-17  
Checkout: `/home/yuyao/methane_train`  
Branch at inspection: `codex/publish-pipeline-updates-20260806` (`e99900c`)

The production checkout contained 153 untracked files totaling approximately 2.0 GB:

| Location | Files | Typical contents |
| --- | ---: | --- |
| `Upgrade_data_pipeline/csv/` | 119 | manifests, audit tables, state snapshots, split summaries |
| `Upgrade_data_pipeline/temp/` | 29 | visual QA images and temporary samples |
| `Upgrade_data_pipeline/logs/` | 2 | pipeline logs |
| `Upgrade_data_pipeline/code/` | 1 | local execution log |
| repository root | 2 | EMIT preprocessing logs |

By file type, the snapshot contained 105 CSV, 28 PNG, 8 JSON, 6 log, 1 PID, and several temporary or reconciliation backup files.

## Retention rules

- **Keep outside Git:** downloaded imagery, generated crops, temporary files, logs, PID files, resumable state, OAuth tokens, and machine-specific absolute paths.
- **Publish as release artifacts:** complete dataset archives and large released manifests. MethaneUnion currently uses Hugging Face for dataset distribution.
- **Commit selectively:** small schema examples, deterministic test fixtures, aggregate validation reports, and manifests required to reproduce a documented release.
- **Do not delete automatically:** files under the A100 production checkout may contain recovery state. Classify and back them up before cleanup.

The repository `.gitignore` excludes the current A100 output directories without deleting their contents. A separate clean worktree should be used for documentation, validation, and refactoring changes.
