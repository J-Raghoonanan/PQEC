"""Matched-schedule analysis tables for the noisy-circuit PQES sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import PurificationSchedule


PAIR_KEYS = [
    "noise",
    "M",
    "p",
    "p_channel",
    "iterations",
    "purification_level",
    "circuit_kind",
    "circuit_seed",
    "p_1q",
    "p_2q",
    "p_cswap",
    "p_memory",
    "p_measure",
    "p_reset",
]

BASELINE_KEYS = [key for key in PAIR_KEYS if key != "purification_level"]

PAIR_METRICS = [
    "fidelity_final",
    "eps_L_final",
    "purity_final",
    "physical_fidelity_final",
    "signed_trace_final",
    "log_abs_signed_trace_final",
    "sampling_overhead_final",
    "log10_sampling_overhead_final",
    "N",
    "max_depth",
    "leaf_copies",
    "swap_gadgets",
    "physical_1q_gate_locations",
    "physical_2q_gate_locations",
    "physical_cswap_locations",
    "physical_measurement_locations",
    "physical_reset_locations",
    "physical_memory_locations",
]


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"finals table is missing required columns: {missing}")


def build_schedule_comparison(finals: pd.DataFrame) -> pd.DataFrame:
    """Return one row per matched interleaved/end-only configuration."""
    _require_columns(
        finals,
        [*PAIR_KEYS, *PAIR_METRICS, "schedule"],
    )
    purified = finals[
        finals["schedule"].isin(
            [
                PurificationSchedule.INTERLEAVED.value,
                PurificationSchedule.END_ONLY.value,
            ]
        )
    ].copy()

    def schedule_frame(schedule: PurificationSchedule) -> pd.DataFrame:
        selected = purified[purified["schedule"] == schedule.value]
        selected = selected[[*PAIR_KEYS, *PAIR_METRICS]].copy()
        return selected.rename(
            columns={
                metric: f"{metric}_{schedule.value}"
                for metric in PAIR_METRICS
            }
        )

    interleaved = schedule_frame(PurificationSchedule.INTERLEAVED)
    end_only = schedule_frame(PurificationSchedule.END_ONLY)
    comparison = interleaved.merge(
        end_only,
        on=PAIR_KEYS,
        how="inner",
        validate="one_to_one",
    )

    baseline = finals[
        finals["schedule"] == PurificationSchedule.NONE.value
    ][[*BASELINE_KEYS, "fidelity_final", "eps_L_final"]].copy()
    baseline = baseline.rename(
        columns={
            "fidelity_final": "fidelity_final_no_pqes",
            "eps_L_final": "eps_L_final_no_pqes",
        }
    )
    comparison = comparison.merge(
        baseline,
        on=BASELINE_KEYS,
        how="left",
        validate="many_to_one",
    )

    delta = (
        comparison["fidelity_final_interleaved"]
        - comparison["fidelity_final_end_only"]
    )
    comparison["delta_fidelity_interleaved_minus_end_only"] = delta
    comparison["fidelity_gain_interleaved_over_no_pqes"] = (
        comparison["fidelity_final_interleaved"]
        - comparison["fidelity_final_no_pqes"]
    )
    comparison["fidelity_gain_end_only_over_no_pqes"] = (
        comparison["fidelity_final_end_only"]
        - comparison["fidelity_final_no_pqes"]
    )
    comparison[
        "delta_log10_sampling_overhead_interleaved_minus_end_only"
    ] = (
        comparison["log10_sampling_overhead_final_interleaved"]
        - comparison["log10_sampling_overhead_final_end_only"]
    )
    comparison["delta_1q_gate_locations_interleaved_minus_end_only"] = (
        comparison["physical_1q_gate_locations_interleaved"]
        - comparison["physical_1q_gate_locations_end_only"]
    )
    comparison["preferred_schedule_by_fidelity"] = np.select(
        [delta > 1e-12, delta < -1e-12],
        [
            PurificationSchedule.INTERLEAVED.value,
            PurificationSchedule.END_ONLY.value,
        ],
        default="tie",
    )
    comparison["matched_leaf_copies"] = comparison[
        "leaf_copies_interleaved"
    ]
    comparison["matched_total_purification_depth"] = comparison[
        "max_depth_interleaved"
    ]
    return comparison.sort_values(
        ["noise", "M", "p", "purification_level"]
    ).reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the matched noisy-circuit schedule comparison table"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/noisy_circuit_simulations"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    finals = pd.read_csv(args.data_dir / "finals.csv")
    comparison = build_schedule_comparison(finals)
    output = args.data_dir / "schedule_comparison.csv"
    comparison.to_csv(output, index=False)
    manifest_path = args.data_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["matched_schedule_pairs"] = len(comparison)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
    print(f"Saved {len(comparison)} matched rows to {output}")


if __name__ == "__main__":
    main()


__all__ = ["build_schedule_comparison"]
