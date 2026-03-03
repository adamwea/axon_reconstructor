# Reconstruction Step — Part 5: Summary JSON and Checkpoint Completion

Scope: this document covers how reconstruction writes its summary JSON and finalizes stage checkpointing.

Primary code path:
- `axon_reconstructor.pipeline.reconstruction.runner.reconstruct_from_templates`

---

## 1. Summary JSON

Reconstruction writes:

- `<well>/stg5_reconstruction_outputs/reconstruction_summary.json`

The summary includes:

- dataset pointers (`h5_path`, `stream_id`, `well_out_dir`)
- templates provenance:
  - `templates_merged_units_dir`
  - `templates_full_channels_templates_dir` (if present)
  - `template_source` flags (`use_full_channels_templates`, `require_full_channels_templates`)
- reconstruction output pointers
- axon_velocity params (filtered to accepted kwargs)
- per-unit entries with:
  - input file paths
  - output file paths
  - status/error info

The summary is intended for:

- debugging and provenance
- lightweight pipeline monitoring
- downstream reporting / bookkeeping

---

## 2. Checkpoint completion

Finally reconstruction writes a checkpoint update:

- stage: `ProcessingStage.ANALYZER_COMPLETE`
- extra fields:
  - `reconstruction_out_dir`
  - `reconstruction_summary_json`
  - `all_units_overview_pdf` (if enabled)

---

## End of reconstruction stage documentation series

At this point reconstruction has produced:

- per-unit JSON and (optionally) PDFs
- an all-units overview PDF (optional)
- a summary JSON
- a completed checkpoint
