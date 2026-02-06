# Channel-Set Nomenclature (Waveforms + Templates)

This document defines the channel/electrode group names used throughout the **waveforms** and **templates** stages (and related preprocessing/spikesorting).

These groups matter because:
- the **concat recording** used for spikesorting contains only the **common channels** (intersection across segments)
- each raw **segment recording** contains many more channels that are absent from the concat recording
- some of those “non-common” channels may overlap between segments (but not across *all* segments)

Unless otherwise noted, “channel ID” here refers to **Maxwell electrode IDs** (integers), i.e. the IDs from `contact_vector['electrode']`.

Two important axes of terminology:

1. **Dataset-level channel sets** (common/segment/all-recorded/etc.)
2. **Unit-level channel sets** (contributing/non-contributing/quiet)

The templates stage combines these: it builds per-unit templates over a channel axis that may include contributing channels and may optionally include additional non-contributing and quiet channels (filled with zeros/NaNs) for plotting/merging.

---

## Definitions

### 1) All recorded channels
- **Meaning**: the union of electrode IDs present across *all* segments in this dataset (i.e., all channels that were recorded at least once).
- **Typical magnitude**: ~10,000 (device/config dependent).
- **Where it exists**: conceptually across the whole dataset; not present in a single SpikeInterface `Recording` in this pipeline.

Notes:
- In templates code/docs this is sometimes referred to as the “recorded electrode universe” (union over sources/analyzers).
- This is **not** the same as the full physical MEA chip (see “Full channels”).

### 2) Common channels
- **Meaning**: electrode IDs present in **every** segment; the set intersection across segments.
- **Typical magnitude**: ~300.
- **Where it exists**:
  - This is exactly the channel set of the **concatenated spikesorting recording**.
  - In code/docs this is sometimes called “concat channels” or “common/intersection channels”.

### 3) Segment channels
- **Meaning**: the full electrode set available in a given segment’s raw Maxwell recording.
- **Typical magnitude**: ~1,000 per segment.
- **Relationship**: contains the common channels plus **non-common segment channels**.

Terminology note:
- In some code/flags/logging, “additional channels” is used as shorthand for “non-common segment channels”.

### 4) Non-common segment channels
- **Meaning**: for a given segment, the set difference:

  $$\text{non-common segment channels} = \text{segment channels} \setminus \text{common channels}$$

- **Typical magnitude**: ~700 per segment.
- **Where it exists**:
  - In the waveforms step, per-segment analyzers can optionally run on *only* this set.

### 5) Unique-segment channels
- **Meaning**: non-common segment channels that appear in **exactly one** segment.

  $$\text{unique-segment channels} = \{c : \mathrm{count}(c)=1\}\setminus \text{common channels}$$

- **Typical magnitude**: <700 (device/config dependent).

### 6) Non-unique (shared) non-common channels
- **Meaning**: channels that appear in **more than one** segment but are **not** in the common intersection.

  $$\text{non-unique non-common} = \{c : 1 < \mathrm{count}(c) < N\}$$

  where $N$ is the number of segments.

- **Typical magnitude**: expected to be small (“a few”), but depends on array configuration.

### 7) Full channels (full MEA chip)
- **Meaning**: the complete physical channel set on the MEA chip, including channels that were never recorded in this dataset.
- **Maxwell/AxonTracking full-chip size**: **26,400 channels** (220 columns × 120 rows).
- **Where it exists**:
  - This is a *physical* chip definition.
  - It does not necessarily correspond to any `Recording` loaded in the pipeline.
  - In the templates stage we may *infer* this geometry (locations + electrode IDs) to:
    - render full-chip maps,
    - represent quiet channels explicitly when needed.

Relationship:

$$\text{full channels} = \text{all recorded channels} \cup \text{quiet channels}$$

---

## Unit-level definitions (templates- and reconstruction-facing)

These definitions are per-unit and are used most heavily in the **templates** stage.

### A) Contributing channels (for a unit)
- **Meaning**: channels that have actual waveform samples for this unit’s template.
- In practice, “contributing” means the channel is present in at least one source analyzer for that unit and is included in the unit’s template channel axis.

Sub-categories:
- **Common contributing channels**: contributing channels that are also in the dataset-level common channel set.
- **Segment contributing channels**: contributing channels that are not common (appear only in some segments), but contribute to this unit’s merged template.

### B) Non-contributing channels (for a unit)
- **Meaning**: channels that are part of the dataset-level **all recorded channels** universe, but do not contribute to this unit.
- These channels may be included explicitly in some outputs (e.g., dense/full-axis templates or full-chip maps) with **zeros or NaNs** so that all units share a consistent channel axis.

### C) Quiet channels (for a unit)
- **Meaning**: channels that are in the dataset-level **full channels** set, but are **not** in the dataset-level all recorded channels universe (i.e., they were never recorded at all in this dataset).
- Quiet channels may be included with **zeros or NaNs** in plotting/merging outputs to represent the full physical chip.

Relationship (per unit, conceptually):

$$\text{full channels} = \text{contributing} \cup \text{non-contributing} \cup \text{quiet}$$

---

## How these sets map to the pipeline

### Spikesorting recording (concat)
- Built by concatenating segments **after slicing each segment down to the common channel intersection**.
- Therefore:
  - `concat_channel_ids` == `common_channel_ids`

### Per-segment waveforms
- Each per-segment analyzer loads the raw segment recording with its full channel set (`segment channels`).
- Optional behavior (`per_segment_only_additional_channels=True`): restrict waveforms to `non-common segment channels`.

Important implementation detail:
- The “common channels” comparisons only make sense if both recordings use **electrode IDs** as channel IDs.
- The code tries to rename channels to electrode IDs using `contact_vector['electrode']`. If that mapping is unavailable, the code falls back to whatever channel IDs SpikeInterface provides, and exclusions/sets are logged as “unknown ID space”.

---

## Waveforms-stage channel-groups log

The waveforms stage writes a JSON file:

- `waveforms_outputs/channel_groups.json`

This is intended to make the above definitions concrete for each run. It includes:
- `common_channel_ids` (concat/common intersection)
- per-segment `segment_electrode_ids` and `non_common_segment_electrode_ids` (when electrode IDs are available)
- `all_recorded_electrode_ids`
- per-segment `waveforms_analyzer_electrode_ids` (the electrode set actually used for waveforms after optional channel selection)
- `unique_segment_electrode_ids_by_source`
- `non_unique_non_common_electrode_ids`
- counts/summaries to sanity-check that “concat channels” truly match the segment intersection

If electrode IDs can’t be recovered for a segment, the JSON will still include the raw `channel_ids` as a fallback but will mark the ID space as unknown.
