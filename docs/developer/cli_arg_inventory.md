# CLI Argument Inventory (3.2f)

Status: archived 3.2f design artifact (captures pre-retirement state before 3.2g cleanup).

Purpose:
- Provide a source-of-truth map for current argument ownership, defaults, and behavior.
- Drive stage-first CLI consolidation and removal of debug-named canonical command paths.

Important:
- This table is intentionally historical and reflects the transition-era surface.
- The current active CLI surface is canonical `stage`, `analysis-deck`, and `scope-run` commands; `debug-*` aliases/wrappers have been retired.

Sources used:
- `src/axon_reconstructor/cli.py`
- `src/axon_reconstructor/pipeline/raw_preprocessing/debug_stage.py`
- `src/axon_reconstructor/pipeline/spikesorting/debug_stage.py`
- `src/axon_reconstructor/pipeline/waveforms/debug_stage.py`
- `src/axon_reconstructor/pipeline/templates/debug_stage.py`
- `src/axon_reconstructor/pipeline/reconstruction/debug_stage.py`
- `src/axon_reconstructor/cli.py` (canonical `stage analysis` args)
- `src/axon_reconstructor/pipeline/analysis/runner.py`
- `src/axon_reconstructor/pipeline/analysis/analysis_deck.py`
- `docs/examples/debug.env.example`

Notes:
- `Current CLI owner` references where parsing is currently defined.
- `Current builder/normalizer` references where values are resolved/merged with env/defaults.
- `Target command path` is provisional and subject to review.
- Gate B row-level decision default: rows are `provisional-approved` unless `Notes` includes an explicit Gate B override tag.

## Inventory Table (v0)

| Arg | Stage | Substage/Scope | Current CLI owner | Current builder/normalizer | Wrapper usage | Env key(s) | Default source | Behavior / side effects | Classification | Target owner | Target command path | Compatibility alias needed? | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `--env-file` | all debug stage cmds + `debug-steps` | env loading | `cli.py` (`register_debug_subcommands`) | `_load_env_for_debug`, `debug_env` loaders | yes | n/a (path to env file) | wrapper/CLI | chooses env source file(s) | diagnostic | stage-common arg registry | `axon-reconstructor stage <stage>` | yes | keep for debugger workflows |
| `--debug/--no-debug` | all debug stage cmds + deck/analysis + steps | logging toggle | `cli.py` | `_configure_logging_for_debug`; wrappers use env fallback | yes | `AXON_RECON_DEBUG` | env/CLI | sets log level; affects SI progress hints in waveforms | diagnostic | stage-common arg registry | `axon-reconstructor stage <stage>` | yes | not stage-specific |
| `--h5-path` | preprocess/spikesort/waveforms/templates/reconstruct/analysis/deck/steps | dataset selector | `cli.py` | stage `build_*_debug_config` and analysis/deck wrappers | yes | `AXON_RECON_H5_PATH` | env/CLI | input recording path | runtime | stage-common arg registry | `axon-reconstructor stage <stage>` | yes | normalize to `Path` once |
| `--stream-id` | preprocess/spikesort/waveforms/templates/reconstruct/analysis/deck/steps | well selector | `cli.py` | stage builders + analysis/deck wrappers | yes | `AXON_RECON_STREAM_ID` | env/CLI | target well/stream | runtime | stage-common arg registry | `axon-reconstructor stage <stage>` | yes | required in canonical stage run |
| `--mea-output-root` | preprocess/spikesort/waveforms/templates/reconstruct/analysis/deck/steps | output root | `cli.py` | stage builders + analysis/deck wrappers | yes | `AXON_RECON_MEA_OUTPUT_ROOT` | env/CLI | per-well output location | runtime | stage-common arg registry | `axon-reconstructor stage <stage>` | yes | required in canonical stage run |
| `--force-restart/--no-force-restart` | preprocess/spikesort/waveforms/templates/reconstruct/analysis/deck/steps | overwrite/resume control | `cli.py` | stage builders + analysis/deck wrappers | yes | `AXON_RECON_FORCE_RESTART` | env/CLI | bypass/rebuild cached artifacts | diagnostic | stage-common arg registry | `axon-reconstructor stage <stage>` | yes | keep alias `--force` temporarily |
| `--force` | spikesort/waveforms/templates/reconstruct/analysis | convenience alias | `cli.py` | stage builders map to `force_restart` | yes | (indirect) | CLI | alias for force restart semantics | wrapper-only | wrappers (or deprecate) | wrapper scripts only | no (if retained in wrappers) | remove from canonical stage surface; Gate B: provisional-deprecate-from-canonical |
| `--unit-limit` | templates/reconstruct/analysis/deck | unit subset control | `cli.py` | templates/recon builders + analysis/deck wrappers | yes | `AXON_RECON_UNIT_LIMIT`, `AXON_RECON_TEMPLATES_UNIT_LIMIT`, `AXON_RECON_RECON_UNIT_LIMIT` | env/CLI | restricts units processed | tuning | stage arg registry (templates/reconstruct/analysis) | `stage templates|reconstruct|analysis` | yes | `none/null/all` parsing shared util |
| `--unit-ids` | templates/reconstruct/analysis/deck | unit allowlist | `cli.py` | templates/recon builders + analysis/deck wrappers | yes | `AXON_RECON_UNIT_IDS` | env/CLI | process selected units | runtime | stage arg registry (templates/reconstruct/analysis) | `stage templates|reconstruct|analysis` | yes | normalize list/int parsing once |
| `--n-jobs` | preprocess/spikesort/waveforms/templates | parallelism | `cli.py` | stage builders | yes | `AXON_RECON_N_JOBS` | env/CLI | thread/process parallelism | tuning | stage-common perf registry | `stage <stage>` | yes | unify semantics across stages |
| `--chunk-duration` | spikesort/waveforms | chunking | `cli.py` | spikesort/waveforms builders | yes | `AXON_RECON_CHUNK_DURATION` | env/CLI | chunk size for compute-heavy operations | tuning | stage-common perf registry | `stage spikesort|waveforms` | yes | string parsing shared |
| `--break-before-run/--no-break-before-run` | preprocess/spikesort | debugger stop | `cli.py` | preprocess/spikesort builders | yes | `AXON_RECON_BREAK_BEFORE_RUN` | env/CLI | optional breakpoint before stage execution | diagnostic | preprocess/spikesort arg registries | `axon-reconstructor stage preprocess|spikesort` | yes (for one release) | canonical stage flag in target model; Gate B: provisional-canonical-only |
| `--temporal-resample-factor` | preprocess | resampling | `cli.py` | preprocessing builder | yes | `AXON_RECON_TEMPORAL_RESAMPLE_FACTOR` | env/CLI | temporal upsampling factor | tuning | preprocess arg registry | `stage preprocess` | yes | pairs with rate/margin/dtype |
| `--temporal-resample-rate-hz` | preprocess | resampling | `cli.py` | preprocessing builder | yes | `AXON_RECON_TEMPORAL_RESAMPLE_RATE_HZ` | env/CLI | explicit target sample rate | tuning | preprocess arg registry | `stage preprocess` | yes | precedence vs factor |
| `--temporal-resample-margin-ms` | preprocess | resampling | `cli.py` | preprocessing builder | yes | `AXON_RECON_TEMPORAL_RESAMPLE_MARGIN_MS` | env/CLI | edge-effect margin | tuning | preprocess arg registry | `stage preprocess` | yes | float parse |
| `--temporal-resample-dtype` | preprocess | resampling | `cli.py` | preprocessing builder | yes | `AXON_RECON_TEMPORAL_RESAMPLE_DTYPE` | env/CLI | output dtype control | tuning | preprocess arg registry | `stage preprocess` | yes | optional |
| `--mea-analysis-repo-root` | spikesort | external dependency path | `cli.py` | spikesort builder | yes | `AXON_RECON_MEA_ANALYSIS_REPO_ROOT` | env/CLI | locates MEA_Analysis repo for stage | runtime | spikesort arg registry | `stage spikesort` | yes | required unless packaged install pathing changes |
| `--sorter` | spikesort/waveforms/analysis | sorter identity | `cli.py` | spikesort/waveforms builders + analysis wrapper BOTM sorter | yes | `AXON_RECON_SORTER`, `AXON_RECON_ANALYSIS_BOTM_SORTER` | env/CLI | selects sorter artifacts/config | runtime | stage-common sorter registry | `stage spikesort|waveforms|analysis` | yes | unify names/source precedence |
| `--docker-image` | spikesort | container selector | `cli.py` | spikesort builder | yes | `AXON_RECON_DOCKER_IMAGE` | env/CLI | selects spikesorter image | runtime | spikesort arg registry | `stage spikesort` | yes | required for current local flow |
| `--torch-threads` | spikesort | perf tuning | `cli.py` | spikesort builder | yes | `AXON_RECON_TORCH_THREADS` | env/CLI | torch intra-op threads | tuning | spikesort perf registry | `stage spikesort` | yes | grouped with thread knobs |
| `--torch-interop-threads` | spikesort | perf tuning | `cli.py` | spikesort builder | yes | `AXON_RECON_TORCH_INTEROP_THREADS` | env/CLI | torch inter-op threads | tuning | spikesort perf registry | `stage spikesort` | yes | grouped with thread knobs |
| `--omp-threads` | spikesort/waveforms | perf tuning | `cli.py` | spikesort/waveforms builders | yes | `AXON_RECON_OMP_THREADS` | env/CLI | BLAS/OpenMP thread cap | tuning | stage-common perf registry | `stage spikesort|waveforms` | yes | unify across stages |
| `--mkl-threads` | spikesort/waveforms | perf tuning | `cli.py` | spikesort/waveforms builders | yes | `AXON_RECON_MKL_THREADS` | env/CLI | MKL thread cap | tuning | stage-common perf registry | `stage spikesort|waveforms` | yes | unify across stages |
| `--openblas-threads` | spikesort/waveforms | perf tuning | `cli.py` | spikesort/waveforms builders | yes | `AXON_RECON_OPENBLAS_THREADS` | env/CLI | OpenBLAS thread cap | tuning | stage-common perf registry | `stage spikesort|waveforms` | yes | unify across stages |
| `--numexpr-threads` | spikesort/waveforms | perf tuning | `cli.py` | spikesort/waveforms builders | yes | `AXON_RECON_NUMEXPR_THREADS` | env/CLI | NumExpr thread cap | tuning | stage-common perf registry | `stage spikesort|waveforms` | yes | unify across stages |
| `--cuda-visible-devices` | spikesort/waveforms/gpu-interact | GPU selection | `cli.py` | spikesort/waveforms builders + runtime hints | yes | `AXON_RECON_CUDA_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES` (gpu-interact fallback source) | env/CLI | controls GPU visibility | tuning | stage-common perf registry | `stage spikesort|waveforms` | yes | also appears outside debug tree |
| `--ks-batch-duration-s` | spikesort | KS4 tuning | `cli.py` | spikesort builder | yes | `AXON_RECON_KS_BATCH_DURATION_S` | env/CLI | derived KS batch size control | tuning | spikesort arg registry | `stage spikesort` | yes | precedence with batch-size |
| `--ks-batch-size` | spikesort | KS4 tuning | `cli.py` | spikesort builder | yes | `AXON_RECON_KS_BATCH_SIZE` | env/CLI | direct KS batch size | tuning | spikesort arg registry | `stage spikesort` | yes | overrides duration option |
| `--curation/--no-curation` | spikesort | post-sort behavior | `cli.py` | spikesort builder | yes | `AXON_RECON_SPIKESORT_CURATION` | env/CLI | enable/disable curation path | tuning | spikesort arg registry | `stage spikesort` | yes | maps to `no_curation` internally |
| `--rerun-analyzer/--no-rerun-analyzer` | spikesort | post-sort behavior | `cli.py` | spikesort builder | yes | `AXON_RECON_SPIKESORT_RERUN_ANALYZER` | env/CLI | force analyzer rerun | diagnostic | spikesort arg registry | `stage spikesort` | yes | |
| `--auto-merge-units/--no-auto-merge-units` | spikesort | post-sort merge | `cli.py` | spikesort builder | yes | `AXON_RECON_SPIKESORT_AUTO_MERGE_UNITS` | env/CLI | toggles SI auto-merge | tuning | spikesort arg registry | `stage spikesort` | yes | |
| `--auto-merge-template-diff-thresh` | spikesort | post-sort merge | `cli.py` | spikesort builder | yes | `AXON_RECON_SPIKESORT_AUTO_MERGE_TEMPLATE_DIFF_THRESH` | env/CLI | threshold sweep values | tuning | spikesort arg registry | `stage spikesort` | yes | comma-separated parse |
| `--force-replot/--no-force-replot` | waveforms | output rewrite behavior | `cli.py` | waveforms builder | yes | `AXON_RECON_FORCE_REPLOT` | env/CLI | regenerate panels while reusing extracts | diagnostic | waveforms arg registry | `stage waveforms` | yes | |
| `--ms-before` | waveforms | extraction window | `cli.py` | waveforms builder | yes | `AXON_RECON_WF_MS_BEFORE` | env/CLI | waveform pre-window | tuning | waveforms arg registry | `stage waveforms` | yes | |
| `--ms-after` | waveforms | extraction window | `cli.py` | waveforms builder | yes | `AXON_RECON_WF_MS_AFTER` | env/CLI | waveform post-window | tuning | waveforms arg registry | `stage waveforms` | yes | |
| `--max-spikes-per-unit` | waveforms | extraction cap | `cli.py` | waveforms builder | yes | `AXON_RECON_WF_MAX_SPIKES_PER_UNIT` | env/CLI | spike subsampling cap | tuning | waveforms arg registry | `stage waveforms` | yes | |
| `--per-segment/--no-per-segment` | waveforms | extraction mode | `cli.py` | waveforms builder | yes | `AXON_RECON_WF_PER_SEGMENT` | env/CLI | segment-level extraction toggle | tuning | waveforms arg registry | `stage waveforms` | yes | |
| `--filter-by-maxwell-epochs/--no-filter-by-maxwell-epochs` | waveforms | extraction filtering | `cli.py` | waveforms builder | yes | `AXON_RECON_WF_FILTER_BY_MAXWELL_EPOCHS` | env/CLI | epoch boundary filtering | tuning | waveforms arg registry | `stage waveforms` | yes | |
| `--debug-max-units` | waveforms | debug limiting | `cli.py` | waveforms builder | yes | `AXON_RECON_WF_DEBUG_MAX_UNITS` | env/CLI | limit units for debug speed | diagnostic | waveforms arg registry | `axon-reconstructor stage waveforms` | yes (for one release) | canonical stage flag in target model; Gate B: provisional-canonical-only |
| `--debug-max-segments` | waveforms | debug limiting | `cli.py` | waveforms builder | yes | `AXON_RECON_WF_DEBUG_MAX_SEGMENTS` | env/CLI | limit segments for debug speed | diagnostic | waveforms arg registry | `axon-reconstructor stage waveforms` | yes (for one release) | canonical stage flag in target model; Gate B: provisional-canonical-only |
| `--replot-from-disk` | templates | mode selector | `cli.py` | templates builder (`do_replot`) | yes | `AXON_RECON_TEMPLATES_REPLOT_FROM_DISK` | env/CLI | replot-only mode | diagnostic | templates substage registry | `stage templates replot` | yes | good substage candidate |
| `--run-templates` | templates | mode selector | `cli.py` | templates builder | yes | `AXON_RECON_TEMPLATES_REPLOT_FROM_DISK` (inverse interaction) | CLI override, else env-driven mode | force extraction mode (`do_replot=False`) | runtime | templates substage registry | `stage templates extract` | yes | good substage candidate |
| `--include-concat/--no-include-concat` | templates | source inclusion | `cli.py` | templates builder | yes | `AXON_RECON_INCLUDE_CONCAT` | env/CLI | include concat analyzer source | tuning | templates arg registry | `stage templates` | yes | |
| `--include-segments/--no-include-segments` | templates | source inclusion | `cli.py` | templates builder | yes | `AXON_RECON_INCLUDE_SEGMENTS` | env/CLI | include segment analyzers | tuning | templates arg registry | `stage templates` | yes | |
| `--plot-templates-grid-pdf/--no-plot-templates-grid-pdf` | templates | output controls | `cli.py` | templates builder | yes | `AXON_RECON_TEMPLATES_PLOT_GRID_PDF` | env/CLI | write templates grid PDF | tuning | templates arg registry | `stage templates` | yes | |
| `--plot-multi-source-templates-pdf/--no-plot-multi-source-templates-pdf` | templates | output controls | `cli.py` | templates builder | yes | `AXON_RECON_TEMPLATES_PLOT_MULTI_SOURCE_PDF` | env/CLI | write multi-source PDFs | tuning | templates arg registry | `stage templates` | yes | |
| `--require-curated-units/--no-require-curated-units` | templates | unit filtering | `cli.py` | templates builder | yes | `AXON_RECON_TEMPLATES_REQUIRE_CURATED_UNITS` | env/CLI | curated-only gating | tuning | templates arg registry | `stage templates` | yes | |
| `--top-channels-per-template` | templates | plotting/detail | `cli.py` | templates builder | yes | `AXON_RECON_TEMPLATES_TOP_CHANNELS_PER_TEMPLATE` | env/CLI | non-merged top-channel plot cap | tuning | templates arg registry | `stage templates` | yes | |
| `--template-time-upsample-factor` | templates | signal processing | `cli.py` | templates builder | yes | `AXON_RECON_TEMPLATES_TIME_UPSAMPLE_FACTOR` | env/CLI | time upsample factor | tuning | templates arg registry | `stage templates` | yes | |
| `--template-time-upsample-method` | templates | signal processing | `cli.py` | templates builder | yes | `AXON_RECON_TEMPLATES_TIME_UPSAMPLE_METHOD` | env/CLI | upsample method (e.g. sinc) | tuning | templates arg registry | `stage templates` | yes | |
| `--rebuild-full-channels-templates` | templates | replot utility | `cli.py` | templates builder/replot runner | yes | n/a | CLI | regenerate full-channel templates in replot mode | diagnostic | templates substage registry | `stage templates replot` | yes | likely substage-specific |
| `--propagation-top-channels` | templates | replot utility | `cli.py` | templates builder/replot runner | yes | n/a | CLI default (25) | propagation plot channel count | tuning | templates substage registry | `stage templates replot` | yes | |
| `--propagation-channels-per-panel` | templates | replot utility | `cli.py` | templates builder/replot runner | yes | n/a | CLI default (25) | panel layout control | tuning | templates substage registry | `stage templates replot` | yes | |
| `--propagation-channel-overlap` | templates | replot utility | `cli.py` | templates builder/replot runner | yes | n/a | CLI default (5) | panel overlap control | tuning | templates substage registry | `stage templates replot` | yes | |
| `--axon-velocity-repo-root` | reconstruct | dependency path | `cli.py` | reconstruction builder | yes | `AXON_RECON_AXON_VELOCITY_REPO_ROOT` | env/CLI | resolves axon_velocity checkout for import fallback | runtime | reconstruct arg registry | `stage reconstruct` | yes | |
| `--replot-summaries-only/--no-replot-summaries-only` | reconstruct | mode toggle | `cli.py` | reconstruction builder | yes | `AXON_RECON_RECON_REPLOT_SUMMARIES_ONLY` | env/CLI | skip full tracking; redraw summaries | diagnostic | reconstruct substage registry | `stage reconstruct summaries` | yes | substage candidate |
| `--recompute-branches-raw-only/--no-recompute-branches-raw-only` | reconstruct | mode toggle | `cli.py` | reconstruction builder | yes | `AXON_RECON_RECON_RECOMPUTE_BRANCHES_RAW_ONLY` | env/CLI | recompute raw branch JSON only | diagnostic | reconstruct substage registry | `stage reconstruct branches` | yes | substage candidate |
| `--r2-threshold` | reconstruct | AV override | `cli.py` | reconstruction builder | yes | `AXON_RECON_AV_R2_THRESHOLD` (via AV key merge) | env/CLI | convenience AV param override | tuning | reconstruct arg registry | `stage reconstruct` | yes | maps into AV params |
| `--av-params` | reconstruct | AV override | `cli.py` | reconstruction builder | yes | `AXON_RECON_AV_PARAMS_JSON` | env/CLI | inline JSON merge into AV params | tuning | reconstruct arg registry | `stage reconstruct` | yes | unify with `--av-params-json` |
| `--av-params-json` | reconstruct | AV override | `cli.py` | reconstruction builder | yes | `AXON_RECON_AV_PARAMS_JSON_PATH` | env/CLI | file JSON merge into AV params | tuning | reconstruct arg registry | `stage reconstruct` | yes | |
| `--write-unit-pdfs/--no-write-unit-pdfs` | reconstruct | output controls | `cli.py` | reconstruction builder | yes | `AXON_RECON_RECON_WRITE_UNIT_PDFS` | env/CLI | per-unit PDF emit toggle | tuning | reconstruct arg registry | `stage reconstruct` | yes | |
| `--write-all-units-overview-pdf/--no-write-all-units-overview-pdf` | reconstruct | output controls | `cli.py` | reconstruction builder | yes | `AXON_RECON_RECON_WRITE_ALL_UNITS_OVERVIEW_PDF` | env/CLI | overview PDF emit toggle | tuning | reconstruct arg registry | `stage reconstruct` | yes | |
| `--verbose/--no-verbose` | reconstruct | logging detail | `cli.py` | reconstruction builder | yes | `AXON_RECON_RECON_VERBOSE` | env/CLI | reconstruction verbosity | diagnostic | reconstruct arg registry | `stage reconstruct` | yes | |
| `--unit-workers` | reconstruct | intra-well parallelism | `cli.py` | reconstruction builder | yes | `AXON_RECON_RECON_UNIT_WORKERS` | env/CLI | per-unit worker count | tuning | reconstruct arg registry | `stage reconstruct` | yes | |
| `--prefer-curated-waveforms-panels/--no-prefer-curated-waveforms-panels` | analysis | panel source policy | `cli.py` | `debug_analysis_step.run_with_args` | yes | `AXON_RECON_ANALYSIS_PREFER_CURATED_WAVEFORMS_PANELS` | env/CLI | choose curated/uncurated panel preference | tuning | analysis arg registry | `stage analysis` | yes | |
| `--botm-enable/--no-botm-enable` | analysis | BOTM validation | `cli.py` | analysis wrapper | yes | `AXON_RECON_ANALYSIS_BOTM_ENABLE` | env/CLI | toggles BOTM computation | diagnostic | analysis substage registry | `stage analysis botm` | yes | substage candidate |
| `--botm-n-events` | analysis | BOTM validation | `cli.py` | analysis/deck wrappers | yes | `AXON_RECON_ANALYSIS_BOTM_N_SPIKE` | env/CLI | number of spike events | tuning | analysis arg registry | `stage analysis` | yes | naming mismatch (`N_SPIKE` vs arg) |
| `--botm-n-noise-windows` | analysis | BOTM validation | `cli.py` | analysis/deck wrappers | yes | `AXON_RECON_ANALYSIS_BOTM_N_NOISE` | env/CLI | number of noise windows | tuning | analysis arg registry | `stage analysis` | yes | |
| `--botm-seed` | analysis | BOTM validation | `cli.py` | analysis/deck wrappers | yes | `AXON_RECON_ANALYSIS_BOTM_SEED` | env/CLI | RNG seed | tuning | analysis arg registry | `stage analysis` | yes | |
| `--botm-prior-signal` | analysis | BOTM validation | `cli.py` | analysis/deck wrappers | yes | `AXON_RECON_ANALYSIS_BOTM_CHANNEL_MATCH_PRIOR_SIGNAL` | env/CLI | BOTM prior P(signal) | tuning | analysis arg registry | `stage analysis` | yes | |
| `--botm-match-fraction-threshold` | analysis | BOTM validation | `cli.py` | analysis/deck wrappers | yes | `AXON_RECON_ANALYSIS_BOTM_CHANNEL_MATCH_FRACTION_THRESHOLD` | env/CLI | good-channel cutoff | tuning | analysis arg registry | `stage analysis` | yes | |
| `--botm-sorter` | analysis | BOTM validation | `cli.py` | analysis/deck wrappers | yes | `AXON_RECON_ANALYSIS_BOTM_SORTER` | env/CLI | sorter for BOTM source loading | runtime | analysis arg registry | `stage analysis` | yes | |
| `--require-complete/--no-require-complete` | analysis-deck | deck filtering | `cli.py` | `debug_analysis_deck.run_with_args` | yes | `AXON_RECON_REQUIRE_COMPLETE` | env/CLI | include only complete panel sets in deck | diagnostic | analysis-deck substage registry | `stage analysis deck` | yes | substage candidate |
| `--steps` | debug-steps | sequence selection | `cli.py` | `_cmd_debug_steps` | yes | n/a | CLI | select subset of stage steps | wrapper-only | wrapper runner only | wrapper script only | no | map to explicit wrapper orchestration, not canonical stage args; Gate B: provisional-wrapper-only |
| `--stop-on-failure/--no-stop-on-failure` | debug-steps | orchestration policy | `cli.py` | `_cmd_debug_steps` | yes | n/a | CLI default true | fail-fast behavior for wrapper sequencing | wrapper-only | wrapper runner only | wrapper script only | no | Gate B: provisional-wrapper-only |
| `--extra-args` | debug-steps | passthrough | `cli.py` | `_cmd_debug_steps` | yes | n/a | CLI | raw arg passthrough to each invoked step | wrapper-only | wrapper runner only | wrapper script only | no | avoid in canonical stage API; Gate B: provisional-wrapper-only |
| `--stage-kwargs`, `--stage-kwargs-file` | `stage` (existing canonical command) | generic stage extension | `cli.py` (`stage` parser) | `_load_stage_kwargs` | no (debug wrappers bypass) | n/a | CLI | inject stage-specific kwargs via JSON | runtime | canonical stage parser | `axon-reconstructor stage <stage>` | n/a | bridge point for post-debug migration |
| `<data_path>` | `mea-sort-cmd` / `pipeline` / `gpu-interact` | command positional | `cli.py` | command handlers (`_cmd_mea_sort_cmd`, `_cmd_pipeline`, `_cmd_gpu_interact`) | no | n/a | CLI positional | dataset file/dir selector for command entrypoints | runtime | command entrypoint parsers (`mea-sort-cmd`, `pipeline`, `gpu-interact`) | `axon-reconstructor mea-sort-cmd|pipeline|gpu-interact <data_path>` | no | may be normalized to shared `--h5-path`/scope input later |
| `<h5_parent_dir>` | `run` | command positional | `cli.py` | `_cmd_run_reconstruction` | no | n/a | CLI positional | parent directory scan root for raw h5 discovery | runtime | command parser (`run`) | `axon-reconstructor run <h5_parent_dir>` | n/a | command-level input distinct from per-stage `--h5-path` |
| `<stage>` | `stage` | command positional | `cli.py` (`stage` parser) | `_cmd_stage` | no | n/a | CLI positional | selects stage implementation to execute | runtime | canonical stage parser | `axon-reconstructor stage <stage>` | n/a | canonical stage selector |
| `--mea-environment` | `mea-sort-cmd` / `run` / `pipeline` / `gpu-interact` | MEA runtime mode | `_add_mea_common_flags` in `cli.py` | command handlers + MEA launch helpers | no | n/a | CLI default (`nersc`) | chooses execution environment preset (e.g., lab/nersc) | runtime | command-common MEA flag registry | `axon-reconstructor mea-sort-cmd|run|pipeline|gpu-interact` | no | common helper-owned arg |
| `--auto-run-driver` | `run` / `pipeline` | MEA launch policy | `cli.py` | `_cmd_run_reconstruction`, `_cmd_pipeline` | no | n/a | CLI default false | allows invoking MEA_Analysis driver when sorter outputs missing | diagnostic | orchestration-command registry (`run`/`pipeline`) | `axon-reconstructor run|pipeline ...` | no | likely command-level, not stage-level |
| `--no-checkpoint` | `run` / `pipeline` | checkpointing control | `cli.py` | command handlers mapping `enable_checkpointing` | no | n/a | CLI default checkpoint enabled | disables axon_reconstructor checkpoint persistence | diagnostic | orchestration-command registry (`run`/`pipeline`) | `axon-reconstructor run|pipeline ...` | no | `dest=enable_checkpointing` inverse flag |
| `--only-load-sortings` | `run` / `pipeline` | legacy compatibility mode | `cli.py` | command handlers | no | n/a | CLI default false | load-only behavior retained for compatibility | wrapper-only | orchestration-command registry (`run`/`pipeline`) | `axon-reconstructor run|pipeline ...` | yes | candidate deprecation after stage-first migration; Gate B: provisional-deprecate-with-replacement |
| `--no-concatenate`, `--no-waveforms`, `--no-templates`, `--no-reconstruct`, `--no-sort` | `run` / `pipeline` | stage toggles in orchestration commands | `cli.py` | command handlers | no | n/a | CLI default stages enabled | disables selected pipeline stages in command orchestration | diagnostic | orchestration-command registry (`run`/`pipeline`) | `axon-reconstructor run|pipeline ...` | yes | orchestration toggles; may map to explicit stage plan model; Gate B: provisional-deprecate-with-replacement |
| `--list-streams` | `pipeline` | inspection utility | `cli.py` | `_cmd_pipeline` | no | n/a | CLI default false | prints available stream IDs then exits | diagnostic | pipeline command parser | `axon-reconstructor pipeline ...` | no | inspection/helper path |
| `--scratch-dir` | `mea-sort-cmd` | NERSC staging | `cli.py` | `_cmd_mea_sort_cmd` / launch command builder | no | `SLURM_TMPDIR` fallback | CLI then env fallback (`SLURM_TMPDIR` for `nersc`) | staging location for NERSC execution scripts | tuning | NERSC command registry | `axon-reconstructor mea-sort-cmd ...` | no | command-level NERSC concern |
| `--stage-back`, `--stage-back-mode` | `mea-sort-cmd` / `gpu-interact` | stage-back policy | `cli.py` | `_cmd_mea_sort_cmd`, `_cmd_gpu_interact` | no | n/a | CLI default (`sorter`, `copy`) | controls copy/move of outputs from staging area | tuning | NERSC command registry | `axon-reconstructor mea-sort-cmd|gpu-interact ...` | no | command-level transport policy |
| `--require-gpu` | `mea-sort-cmd` | resource requirement | `cli.py` | `_cmd_mea_sort_cmd` | no | n/a | CLI default false | enforces GPU availability when building execution command | runtime | NERSC command registry | `axon-reconstructor mea-sort-cmd ...` | no | command safety guard |
| `--account`, `--qos`, `--constraint`, `--time` | `gpu-interact` | Slurm allocation controls | `cli.py` | `_gpu_interact_defaults`, `_cmd_gpu_interact` | no | `GPU_SMOKE_SALLOC_ACCOUNT`, `GPU_SMOKE_SALLOC_QOS`, `GPU_SMOKE_SALLOC_CONSTRAINT`, `GPU_SMOKE_SALLOC_TIME` | CLI override then env/default fallback | controls salloc resource request parameters | tuning | NERSC command registry | `axon-reconstructor gpu-interact ...` | no | HPC scheduler controls |
| `--shifter-image` | `gpu-interact` | container runtime selector | `cli.py` | `_cmd_gpu_interact` | no | `SHIFTER_IMAGE` | CLI override then env fallback | selects shifter image URI for spikesorting | runtime | NERSC command registry | `axon-reconstructor gpu-interact ...` | no | command-level container override |
| `--dry-run` | `gpu-interact` / `scope-run` | plan-only execution | `cli.py` | `_cmd_gpu_interact`, `_cmd_scope_run` | no | n/a | CLI default false | prints/records plan without running work | diagnostic | command-common dry-run registry | `axon-reconstructor gpu-interact|scope-run ...` | no | command-level dry-run semantics vary by command |
| `--config` | `scope-run` | scope config input | `cli.py` | `_cmd_scope_run` | no | n/a | CLI required | path to scope run config file | runtime | scope-run parser | `axon-reconstructor scope-run --config ...` | no | required for scope runner |
| `--summary-out` | `scope-run` | reporting output control | `cli.py` | `_cmd_scope_run` | no | n/a | default `<mea_output_root>/scope_run_summary.json` | output JSON summary location | diagnostic | scope-run parser | `axon-reconstructor scope-run --summary-out ...` | no | optional override |

## Classification vocabulary

- `runtime` — required execution inputs/identifiers
- `tuning` — performance/resource or algorithm tuning controls
- `diagnostic` — troubleshooting/inspection controls useful in debugger
- `wrapper-only` — convenience flags only for wrapper defaults UX

## Open issues found in v0 inventory

1. (resolved in 3.2g) `analysis-deck` normalization moved into package module `src/axon_reconstructor/pipeline/analysis/analysis_deck.py`.
2. The same conceptual knobs have divergent names/default keys in some places (especially BOTM and unit-limit families).
3. `--force` aliases and debug step orchestration flags should likely move to wrapper-only scope in the target model.
4. Some flags are really substage-specific (`templates` replot controls, reconstruction summary/raw-only modes).

## Next fill steps (toward Gate B)

- Convert provisional ownership approvals below to named reviewer sign-off after team review.
- Resolve open-issue naming drifts (`BOTM_*`, unit-limit family) before parser migration.

## Provenance verification notes (legacy/debug rows)

- Verified in stage debug builders (`CLI > env > code defaults` precedence):
	- `src/axon_reconstructor/pipeline/raw_preprocessing/debug_stage.py`
	- `src/axon_reconstructor/pipeline/spikesorting/debug_stage.py`
	- `src/axon_reconstructor/pipeline/waveforms/debug_stage.py`
	- `src/axon_reconstructor/pipeline/templates/debug_stage.py`
	- `src/axon_reconstructor/pipeline/reconstruction/debug_stage.py`
- Verified in debug analysis wrappers (`CLI > env > module constants` precedence):
	- `src/axon_reconstructor/cli.py` (`stage analysis` env/CLI resolution)
	- `src/axon_reconstructor/pipeline/analysis/analysis_deck.py`
- Verified debug orchestration path (`debug-steps`) is CLI-owned with no env fallback for `--steps`, `--stop-on-failure`, `--extra-args`.

## Owner sign-off markers (legacy/debug rows)

- Provisional sign-off: legacy/debug row provenance verified in this pass (2026-03-01).
- Remaining action: replace provisional sign-off with named reviewer approval during Gate B (ownership approved).

## Gate B ownership decisions (command-family matrix)

| Command family | Scope | Canonical owner | Canonical command path | Keep in canonical stage-first CLI? | Compatibility policy |
|---|---|---|---|---|---|
| Stage-common inputs | `--h5-path`, `--stream-id`, `--mea-output-root`, `--debug`, `--force-restart` | stage-common arg registry | `axon-reconstructor stage <stage>` | yes | keep `debug-*` aliases for one release |
| Stage performance/resource knobs | `--n-jobs`, `--chunk-duration`, thread knobs, GPU visibility | stage-common perf registry + stage-specific perf registries | `axon-reconstructor stage <stage>` | yes | preserve flag names where feasible |
| Spikesort stage-specific | sorter/image/KS/curation/merge knobs | spikesort arg registry | `axon-reconstructor stage spikesort` | yes | keep `debug-spikesort` alias for one release |
| Waveforms stage-specific | extraction windows/filtering/debug caps | waveforms arg registry | `axon-reconstructor stage waveforms` | yes | keep `debug-waveforms` alias for one release; debug-cap flags stay canonical |
| Templates stage-specific | include/plot/curation/time-upsample | templates arg registry | `axon-reconstructor stage templates` | yes | preserve options; move replot utilities under substage |
| Templates replot utilities | `--replot-from-disk`, propagation panel controls | templates substage registry | `axon-reconstructor stage templates replot` | yes | map `debug-templates` compatibility to substage path for one release |
| Reconstruction stage-specific | AV params, output toggles, unit workers, replot/recompute modes | reconstruct arg registry + reconstruct substages | `axon-reconstructor stage reconstruct [substage]` | yes | keep debug aliases for one release |
| Analysis stage-specific | BOTM knobs, curated panel preference | analysis arg registry + analysis substages | `axon-reconstructor stage analysis [substage]` | yes | keep `debug-analysis` and `debug-analysis-deck` aliases for one release |
| Wrapper orchestration-only | `--steps`, `--stop-on-failure`, `--extra-args` | wrapper runner only | wrapper script surface only | no | do not promote to canonical stage API |
| Orchestration command toggles | `--only-load-sortings`, `--no-*` stage toggles | orchestration-command registry (`run`/`pipeline`) | `axon-reconstructor run|pipeline` | no (stage-first target) | deprecate after stage-plan replacement is available |
| HPC/NERSC command controls | `--account`, `--qos`, `--constraint`, `--time`, `--shifter-image`, `--scratch-dir`, `--stage-back*` | NERSC command registry | `axon-reconstructor mea-sort-cmd|gpu-interact` | command-specific yes; stage-first no | keep command-specific surfaces; avoid adding to generic stage command |
| Scope orchestration controls | `--config`, `--summary-out`, `--dry-run` (scope semantics) | scope-run parser | `axon-reconstructor scope-run` | command-specific yes; stage-first no | retain as separate orchestration command |

## Gate B provisional approval checklist

- [x] Classification decisions assigned at command-family level.
- [x] Target owner decisions assigned at command-family level.
- [x] Target command-path decisions assigned at command-family level.
- [x] Row-level decision tags applied (`provisional-approved` default + explicit override tags in exception rows).
- [x] Named reviewer sign-off captured (required to close Gate B).

## Gate B reviewer sign-off (finalization template)

| Reviewer | Date (UTC) | Decision | Scope | Notes |
|---|---|---|---|---|
| adamwea | 2026-03-01 | approve | Gate B ownership + classification + command paths | decisions captured via Copilot clarification Q&A: one-release aliases, debug-focused flags canonical-only, proceed to implementation |

Closure rule:
- Gate B can be marked complete after at least one named `approve` or `approve-with-notes` entry is recorded above and any notes are reflected in this inventory.

## Provenance verification notes (canonical non-debug rows)

- Verified in `src/axon_reconstructor/cli.py` for parser defaults and command handlers:
	- `_add_mea_common_flags` (`--mea-environment` default `nersc`),
	- `_cmd_mea_sort_cmd` (`--scratch-dir` fallback to `SLURM_TMPDIR` for `nersc`),
	- `_gpu_interact_defaults` + `_cmd_gpu_interact` (`GPU_SMOKE_SALLOC_*` and `SHIFTER_IMAGE` env fallbacks),
	- `_cmd_scope_run` (`--summary-out` default resolved at runtime to `<mea_output_root>/scope_run_summary.json`).
- Result: canonical non-debug rows now have verified env/default provenance in this inventory pass.

## Appendix A — Machine-parsed CLI arguments by command (`src/axon_reconstructor/cli.py`)

- Generation method: AST parse of `sub.add_parser(...)` + `.add_argument(...)` + `_add_mea_common_flags(...)` in `src/axon_reconstructor/cli.py`.
- Scope note (v1): includes debug and canonical non-debug command trees from current `main()` parser assembly.

- Commands parsed: **14**
- Unique args/positionals parsed: **101**

| Command | Parsed args/positionals | Count |
|---|---|---:|
| `debug-preprocess` | `--env-file`, `--debug`, `--h5-path`, `--stream-id`, `--n-jobs`, `--mea-output-root`, `--break-before-run`, `--force-restart`, `--temporal-resample-factor`, `--temporal-resample-rate-hz`, `--temporal-resample-margin-ms`, `--temporal-resample-dtype` | 12 |
| `debug-spikesort` | `--env-file`, `--debug`, `--mea-analysis-repo-root`, `--h5-path`, `--stream-id`, `--mea-output-root`, `--sorter`, `--docker-image`, `--force-restart`, `--force`, `--break-before-run`, `--n-jobs`, `--chunk-duration`, `--torch-threads`, `--torch-interop-threads`, `--omp-threads`, `--mkl-threads`, `--openblas-threads`, `--numexpr-threads`, `--cuda-visible-devices`, `--ks-batch-duration-s`, `--ks-batch-size`, `--curation`, `--rerun-analyzer`, `--auto-merge-units`, `--auto-merge-template-diff-thresh` | 26 |
| `debug-waveforms` | `--env-file`, `--debug`, `--force-restart`, `--force-replot`, `--force`, `--h5-path`, `--stream-id`, `--mea-output-root`, `--sorter`, `--ms-before`, `--ms-after`, `--max-spikes-per-unit`, `--n-jobs`, `--chunk-duration`, `--omp-threads`, `--mkl-threads`, `--openblas-threads`, `--numexpr-threads`, `--cuda-visible-devices`, `--per-segment`, `--filter-by-maxwell-epochs`, `--debug-max-units`, `--debug-max-segments` | 23 |
| `debug-templates` | `--env-file`, `--debug`, `--unit-ids`, `--force`, `--h5-path`, `--stream-id`, `--mea-output-root`, `--n-jobs`, `--force-restart`, `--unit-limit`, `--include-concat`, `--include-segments`, `--plot-templates-grid-pdf`, `--plot-multi-source-templates-pdf`, `--require-curated-units`, `--top-channels-per-template`, `--template-time-upsample-factor`, `--template-time-upsample-method`, `--rebuild-full-channels-templates`, `--propagation-top-channels`, `--propagation-channels-per-panel`, `--propagation-channel-overlap` | 22 |
| `debug-reconstruct` | `--env-file`, `--debug`, `--axon-velocity-repo-root`, `--h5-path`, `--stream-id`, `--mea-output-root`, `--force-restart`, `--force`, `--unit-limit`, `--unit-ids`, `--replot-summaries-only`, `--recompute-branches-raw-only`, `--r2-threshold`, `--av-params`, `--av-params-json`, `--write-unit-pdfs`, `--write-all-units-overview-pdf`, `--verbose`, `--unit-workers` | 19 |
| `debug-analysis` | `--env-file`, `--debug`, `--h5-path`, `--stream-id`, `--mea-output-root`, `--force-restart`, `--force`, `--unit-limit`, `--unit-ids`, `--prefer-curated-waveforms-panels`, `--botm-enable`, `--botm-n-events`, `--botm-n-noise-windows`, `--botm-seed`, `--botm-prior-signal`, `--botm-match-fraction-threshold`, `--botm-sorter` | 17 |
| `debug-analysis-deck` | `--env-file`, `--debug`, `--h5-path`, `--stream-id`, `--mea-output-root`, `--force-restart`, `--unit-limit`, `--unit-ids`, `--require-complete` | 9 |
| `debug-steps` | `--env-file`, `--h5-path`, `--stream-id`, `--mea-output-root`, `--force-restart`, `--debug`, `--steps`, `--stop-on-failure`, `--extra-args` | 9 |
| `mea-sort-cmd` | `<data_path>`, `--docker-image`, `--scratch-dir`, `--stage-back`, `--stage-back-mode`, `--cuda-visible-devices`, `--require-gpu`, `--n-jobs`, `--chunk-duration`, `--mea-environment`, `--mea-output-root`, `--mea-analysis-repo-root`, `--sorter` | 13 |
| `run` | `<h5_parent_dir>`, `--docker-image`, `--auto-run-driver`, `--force-restart`, `--no-checkpoint`, `--only-load-sortings`, `--no-concatenate`, `--no-waveforms`, `--no-templates`, `--no-reconstruct`, `--mea-environment`, `--mea-output-root`, `--mea-analysis-repo-root`, `--sorter` | 14 |
| `pipeline` | `<data_path>`, `--docker-image`, `--auto-run-driver`, `--force-restart`, `--no-checkpoint`, `--only-load-sortings`, `--no-concatenate`, `--no-sort`, `--no-waveforms`, `--no-templates`, `--no-reconstruct`, `--stream-id`, `--list-streams`, `--n-jobs`, `--mea-environment`, `--mea-output-root`, `--mea-analysis-repo-root`, `--sorter` | 18 |
| `gpu-interact` | `<data_path>`, `--account`, `--qos`, `--constraint`, `--time`, `--shifter-image`, `--cuda-visible-devices`, `--n-jobs`, `--chunk-duration`, `--stage-back`, `--stage-back-mode`, `--dry-run`, `--mea-environment`, `--mea-output-root`, `--mea-analysis-repo-root`, `--sorter` | 16 |
| `stage` | `<stage>`, `--h5-path`, `--stream-id`, `--mea-output-root`, `--mea-analysis-repo-root`, `--sorter`, `--docker-image`, `--n-jobs`, `--chunk-duration`, `--force-restart`, `--debug`, `--stage-kwargs`, `--stage-kwargs-file` | 13 |
| `scope-run` | `--config`, `--dry-run`, `--summary-out`, `--debug` | 4 |

### Unique parsed args/positionals (deduplicated)

`--env-file`, `--debug`, `--h5-path`, `--stream-id`, `--n-jobs`, `--mea-output-root`, `--break-before-run`, `--force-restart`, `--temporal-resample-factor`, `--temporal-resample-rate-hz`, `--temporal-resample-margin-ms`, `--temporal-resample-dtype`, `--mea-analysis-repo-root`, `--sorter`, `--docker-image`, `--force`, `--chunk-duration`, `--torch-threads`, `--torch-interop-threads`, `--omp-threads`, `--mkl-threads`, `--openblas-threads`, `--numexpr-threads`, `--cuda-visible-devices`, `--ks-batch-duration-s`, `--ks-batch-size`, `--curation`, `--rerun-analyzer`, `--auto-merge-units`, `--auto-merge-template-diff-thresh`, `--force-replot`, `--ms-before`, `--ms-after`, `--max-spikes-per-unit`, `--per-segment`, `--filter-by-maxwell-epochs`, `--debug-max-units`, `--debug-max-segments`, `--unit-ids`, `--unit-limit`, `--include-concat`, `--include-segments`, `--plot-templates-grid-pdf`, `--plot-multi-source-templates-pdf`, `--require-curated-units`, `--top-channels-per-template`, `--template-time-upsample-factor`, `--template-time-upsample-method`, `--rebuild-full-channels-templates`, `--propagation-top-channels`, `--propagation-channels-per-panel`, `--propagation-channel-overlap`, `--axon-velocity-repo-root`, `--replot-summaries-only`, `--recompute-branches-raw-only`, `--r2-threshold`, `--av-params`, `--av-params-json`, `--write-unit-pdfs`, `--write-all-units-overview-pdf`, `--verbose`, `--unit-workers`, `--prefer-curated-waveforms-panels`, `--botm-enable`, `--botm-n-events`, `--botm-n-noise-windows`, `--botm-seed`, `--botm-prior-signal`, `--botm-match-fraction-threshold`, `--botm-sorter`, `--require-complete`, `--steps`, `--stop-on-failure`, `--extra-args`, `<data_path>`, `--scratch-dir`, `--stage-back`, `--stage-back-mode`, `--require-gpu`, `--mea-environment`, `<h5_parent_dir>`, `--auto-run-driver`, `--no-checkpoint`, `--only-load-sortings`, `--no-concatenate`, `--no-waveforms`, `--no-templates`, `--no-reconstruct`, `--no-sort`, `--list-streams`, `--account`, `--qos`, `--constraint`, `--time`, `--shifter-image`, `--dry-run`, `<stage>`, `--stage-kwargs`, `--stage-kwargs-file`, `--config`, `--summary-out`

## Coverage Checklist (Gate A)

- [x] Parser command coverage captured from `cli.py` (`14/14` subcommands represented in Appendix A).
- [x] Parser arg universe captured from `cli.py` (`101` unique args/positionals listed).
- [x] Inventory table row coverage complete for canonical non-debug command-only args.
