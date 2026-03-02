"""Debug script: run BOTM validation for one unit.

This is a harness-friendly entrypoint that does NOT require wiring through
Stage 06. It computes BOTM metrics for a single unit and (optionally) writes
analysis_outputs/botm_validation artifacts.

Example:
  /home/adamm/dev/pkgs/axon_reconstructor/.venv/bin/python \
    debug_botm_validation_one_unit.py \
    --h5-path /path/to/data.raw.h5 \
    --stream-id well003 \
    --mea-output-root /path/to/output_root \
    --unit-id 1 \
        --n-events 200 \
        --n-noise-windows 2000 \
    --seed 0

Debugging in VS Code
--------------------
If you run this script with *no* CLI args (e.g. via the VS Code Python debugger),
it will automatically load ./debug.env (next to this file) and resolve required
values from env vars.

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to env file (default: ./debug.env next to this script)",
    )

    p.add_argument("--h5-path", type=Path, default=None)
    p.add_argument("--stream-id", type=str, default=None)
    p.add_argument("--mea-output-root", type=Path, default=None)

    p.add_argument("--unit-id", type=str, default=None, help="Unit id (int-like preferred)")

    # New naming
    p.add_argument("--n-events", type=int, default=None)
    p.add_argument("--n-noise-windows", type=int, default=None)

    # Backwards-compatible aliases (map to n-events/n-noise-windows)
    p.add_argument("--n-spike", type=int, default=None, help="Alias for --n-events")
    p.add_argument("--n-noise", type=int, default=None, help="Alias for --n-noise-windows")

    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--prior-signal", type=float, default=None)
    p.add_argument(
        "--match-fraction-threshold",
        type=float,
        default=None,
        help="Good-channel cutoff: channel match fraction must be > this value (default: 0.70)",
    )
    p.add_argument("--sorter", type=str, default=None)

    p.add_argument(
        "--write",
        action="store_true",
        help="If set, also write under <well_out_dir>/analysis_outputs/botm_validation/",
    )

    args = p.parse_args()

    # Load debug.env by default (makes this script work nicely under VS Code debugger).
    import debug_env

    if args.env_file is not None:
        env_files = [Path(args.env_file)]
    else:
        env_files = debug_env.default_env_paths(script_path=__file__)
    debug_env.load_env_files_into_os(env_files=env_files, override_existing=False)

    # Precedence: CLI > env > defaults.
    h5_path = Path(args.h5_path) if args.h5_path is not None else debug_env.env_required_path("AXON_RECON_H5_PATH")
    stream_id = str(args.stream_id) if args.stream_id is not None else debug_env.env_required_str("AXON_RECON_STREAM_ID")
    mea_output_root = (
        Path(args.mea_output_root)
        if args.mea_output_root is not None
        else debug_env.env_required_path("AXON_RECON_MEA_OUTPUT_ROOT")
    )

    unit_id_raw: str | None
    if args.unit_id is not None:
        unit_id_raw = str(args.unit_id)
    else:
        unit_id_raw = debug_env.env_str("AXON_RECON_BOTM_UNIT_ID", default=None)
        if unit_id_raw is None:
            unit_ids = debug_env.env_int_list("AXON_RECON_UNIT_IDS")
            if unit_ids:
                unit_id_raw = str(unit_ids[0])
    if unit_id_raw is None or str(unit_id_raw).strip() == "":
        raise SystemExit(
            "Missing unit id. Provide --unit-id, or set AXON_RECON_BOTM_UNIT_ID, "
            "or set AXON_RECON_UNIT_IDS (first id is used)."
        )

    seed = (
        int(args.seed)
        if args.seed is not None
        else int(debug_env.env_int("AXON_RECON_ANALYSIS_BOTM_SEED", default=0) or 0)
    )

    n_events = (
        int(args.n_events)
        if args.n_events is not None
        else (int(args.n_spike) if args.n_spike is not None else int(debug_env.env_int("AXON_RECON_ANALYSIS_BOTM_N_SPIKE", default=200) or 200))
    )
    n_noise_windows = (
        int(args.n_noise_windows)
        if args.n_noise_windows is not None
        else (int(args.n_noise) if args.n_noise is not None else int(debug_env.env_int("AXON_RECON_ANALYSIS_BOTM_N_NOISE", default=2000) or 2000))
    )
    prior_signal = (
        float(args.prior_signal)
        if args.prior_signal is not None
        else float(debug_env.env_float("AXON_RECON_ANALYSIS_BOTM_CHANNEL_MATCH_PRIOR_SIGNAL", default=0.5) or 0.5)
    )
    match_fraction_threshold = (
        float(args.match_fraction_threshold)
        if args.match_fraction_threshold is not None
        else float(debug_env.env_float("AXON_RECON_ANALYSIS_BOTM_CHANNEL_MATCH_FRACTION_THRESHOLD", default=0.70) or 0.70)
    )
    sorter = (
        str(args.sorter)
        if args.sorter is not None
        else str(debug_env.env_str("AXON_RECON_ANALYSIS_BOTM_SORTER", default="kilosort4") or "kilosort4")
    )

    try:
        uid: Any = int(unit_id_raw)
    except Exception:
        uid = unit_id_raw

    from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir

    well_out_dir = compute_mea_analysis_output_dir(
        output_root=mea_output_root,
        data_file=h5_path,
        well=stream_id,
    )

    print(f"well_out_dir: {well_out_dir}")

    from axon_reconstructor.pipeline.analysis.botm_validation import (
        BotmValidationInputs,
        compute_botm_validation_for_unit,
        write_botm_validation_outputs,
    )

    import logging

    logger = logging.getLogger("debug.botm_validation")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setLevel(logging.INFO)
        logger.addHandler(h)

    if args.write:
        out_dir = Path(well_out_dir) / "analysis_outputs" / "botm_validation"
        inputs = BotmValidationInputs(
            well_out_dir=Path(well_out_dir),
            h5_path=Path(h5_path),
            stream_id=str(stream_id),
            unit_ids=[uid],
            n_events=int(n_events),
            n_noise_windows=int(n_noise_windows),
            seed=int(seed),
            prior_signal=float(prior_signal),
            match_fraction_threshold=float(match_fraction_threshold),
            sorter=str(sorter),
            out_dir=out_dir,
            force_restart=True,
        )
        summary = write_botm_validation_outputs(inputs=inputs, logger=logger)
        print(f"wrote: {out_dir / 'summary.json'}")
        print("metrics_json:", summary["units"][0]["artifacts"]["metrics_json"])
        return 0

    inputs = BotmValidationInputs(
        well_out_dir=Path(well_out_dir),
        h5_path=Path(h5_path),
        stream_id=str(stream_id),
        unit_ids=[uid],
        n_events=int(n_events),
        n_noise_windows=int(n_noise_windows),
        seed=int(seed),
        prior_signal=float(prior_signal),
        match_fraction_threshold=float(match_fraction_threshold),
        sorter=str(sorter),
        out_dir=None,
        force_restart=False,
    )
    rec = compute_botm_validation_for_unit(inputs=inputs, uid=uid, logger=logger)

    # Write next to the script for easy inspection.
    out_json = Path(__file__).with_suffix("").with_name(f"_botm_unit_{uid}_metrics.json")
    _write_json(out_json, rec)

    print("status:", rec.get("status"))
    if rec.get("status") != "ok":
        print("error:", rec.get("error"))
    else:
        cm = rec.get("channel_match", {})
        print("n_good_channels:", cm.get("n_good_channels"))
        print("good_channel_ids:", cm.get("good_channel_ids"))
    print("wrote:", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
