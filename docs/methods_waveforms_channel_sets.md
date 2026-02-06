# Waveforms Step — Channel-Set Nomenclature (Maxwell / AxonTracking Arrays)

This document defines the channel/electrode group names used throughout the waveforms stage (and related preprocessing/spikesorting).

These groups matter because:
- the **concat recording** used for spikesorting contains only the **common channels** (intersection across segments)
- each raw **segment recording** contains many more channels that are absent from the concat recording
- some of those “non-common” channels may overlap between segments (but not across *all* segments)

Unless otherwise noted, “channel ID” here refers to **Maxwell electrode IDs** (integers), i.e. the IDs from `contact_vector['electrode']`.

---

## Definitions

### 1) All channels
- **Meaning**: the union of electrode IDs present across *all* concatenated segments.
- **Typical magnitude**: ~10,000 (device/config dependent).
- **Where it exists**: conceptually across the whole dataset; not present in a single SpikeInterface `Recording` in this pipeline.

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
- `all_channels_union_electrode_ids`
- `unique_segment_electrode_ids_by_source`
- `non_unique_non_common_electrode_ids`
- counts/summaries to sanity-check that “concat channels” truly match the segment intersection

If electrode IDs can’t be recovered for a segment, the JSON will still include the raw `channel_ids` as a fallback but will mark the ID space as unknown.
