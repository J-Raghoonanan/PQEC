"""Grid sweep for the circuit-level noisy PQES analysis.

Examples
--------
Quick validation grid:

    python -m src.noisy_circuit_simulations.grid_run --quick

Full default grid:

    python -m src.noisy_circuit_simulations.grid_run

Hardware-weighted example, with error rates relative to the swept 1q rate p:

    python -m src.noisy_circuit_simulations.grid_run \
        --p2-multiplier 5 --cswap-multiplier 10 \
        --memory-multiplier 0.2 --measure-multiplier 2
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .analysis import build_schedule_comparison
from .config import (
    CircuitKind,
    CircuitSpec,
    ExperimentSpec,
    HardwareNoiseSpec,
    NoiseFamily,
    PurificationSchedule,
    PurificationSpec,
)
from .runner import run_experiment


logger = logging.getLogger(__name__)

FULL_M_VALUES = [1, 2, 3]
FULL_P_VALUES = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]
FULL_ELL_VALUES = [0, 1, 2, 3]
FULL_CYCLES = 5

QUICK_M_VALUES = [1, 2]
QUICK_P_VALUES = [0.0, 0.01, 0.03]
QUICK_ELL_VALUES = [0, 1, 2]
QUICK_CYCLES = 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Circuit-level PQES grid sweep with noisy gates"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/noisy_circuit_simulations"),
        help="Output directory for steps.csv, finals.csv, and manifest.json",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--M-values", type=int, nargs="+")
    parser.add_argument("--p-values", type=float, nargs="+")
    parser.add_argument("--ell-values", type=int, nargs="+")
    parser.add_argument("--cycles", type=int)
    parser.add_argument(
        "--noise",
        choices=["all"] + [family.value for family in NoiseFamily],
        default="all",
    )
    parser.add_argument(
        "--schedules",
        nargs="+",
        choices=[schedule.value for schedule in PurificationSchedule],
        default=[
            PurificationSchedule.INTERLEAVED.value,
            PurificationSchedule.END_ONLY.value,
        ],
    )
    parser.add_argument(
        "--circuit",
        choices=[kind.value for kind in CircuitKind],
        default=CircuitKind.BRICKWORK.value,
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rotation-scale", type=float, default=1.0)
    parser.add_argument("--p2-multiplier", type=float, default=1.0)
    parser.add_argument("--cswap-multiplier", type=float, default=1.0)
    parser.add_argument("--memory-multiplier", type=float, default=1.0)
    parser.add_argument("--measure-multiplier", type=float, default=1.0)
    parser.add_argument("--reset-multiplier", type=float, default=1.0)
    return parser.parse_args()


def _clipped(rate: float) -> float:
    return max(0.0, min(1.0, float(rate)))


def _hardware_spec(
    family: NoiseFamily,
    p: float,
    args: argparse.Namespace,
) -> HardwareNoiseSpec:
    return HardwareNoiseSpec(
        family=family,
        p_1q=p,
        p_2q=_clipped(p * args.p2_multiplier),
        p_cswap=_clipped(p * args.cswap_multiplier),
        p_memory=_clipped(p * args.memory_multiplier),
        p_measure=_clipped(p * args.measure_multiplier),
        p_reset=_clipped(p * args.reset_multiplier),
    )


def _families(flag: str) -> List[NoiseFamily]:
    if flag == "all":
        return list(NoiseFamily)
    return [NoiseFamily(flag)]


def _append_or_write(
    frame: pd.DataFrame,
    path: Path,
    *,
    append: bool,
) -> None:
    if append and path.exists():
        previous = pd.read_csv(path)
        frame = pd.concat([previous, frame], ignore_index=True)
    frame.to_csv(path, index=False)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.quick:
        M_values = args.M_values or QUICK_M_VALUES
        p_values = args.p_values or QUICK_P_VALUES
        ell_values = args.ell_values or QUICK_ELL_VALUES
        cycles = args.cycles or QUICK_CYCLES
    else:
        M_values = args.M_values or FULL_M_VALUES
        p_values = args.p_values or FULL_P_VALUES
        ell_values = args.ell_values or FULL_ELL_VALUES
        cycles = args.cycles or FULL_CYCLES

    if any(M < 1 or M > 4 for M in M_values):
        raise ValueError("M values must lie in [1, 4]")
    if any(not 0.0 <= p <= 1.0 for p in p_values):
        raise ValueError("p values must lie in [0, 1]")
    if any(ell < 0 for ell in ell_values):
        raise ValueError("ell values must be nonnegative")
    if cycles < 1:
        raise ValueError("cycles must be positive")

    requested_schedules = {
        PurificationSchedule(value) for value in args.schedules
    }
    purified_schedules = [
        schedule
        for schedule in (
            PurificationSchedule.INTERLEAVED,
            PurificationSchedule.END_ONLY,
        )
        if schedule in requested_schedules
    ]
    circuit_kind = CircuitKind(args.circuit)

    specs: List[ExperimentSpec] = []
    for family in _families(args.noise):
        for M in M_values:
            for p in p_values:
                hardware = _hardware_spec(family, p, args)
                circuit = CircuitSpec(
                    M=M,
                    cycles=cycles,
                    kind=circuit_kind,
                    seed=args.seed,
                    rotation_scale=args.rotation_scale,
                )

                # One baseline per (noise, M, p), represented by ell=0.
                specs.append(
                    ExperimentSpec(
                        circuit=circuit,
                        hardware=hardware,
                        purification=PurificationSpec(
                            schedule=PurificationSchedule.NONE,
                            levels_per_cycle=0,
                        ),
                        out_dir=args.out,
                    )
                )

                for ell in sorted(set(ell_values)):
                    if ell == 0:
                        continue
                    for schedule in purified_schedules:
                        specs.append(
                            ExperimentSpec(
                                circuit=circuit,
                                hardware=hardware,
                                purification=PurificationSpec(
                                    schedule=schedule,
                                    levels_per_cycle=ell,
                                ),
                                out_dir=args.out,
                            )
                        )

    logger.info(
        "Running %d experiments: M=%s, p=%s, ell=%s, cycles=%d, "
        "noise=%s, schedules=%s",
        len(specs),
        M_values,
        p_values,
        ell_values,
        cycles,
        [family.value for family in _families(args.noise)],
        [schedule.value for schedule in purified_schedules],
    )

    step_frames: List[pd.DataFrame] = []
    final_frames: List[pd.DataFrame] = []
    errors: List[Dict[str, object]] = []
    started = time.time()

    for index, spec in enumerate(specs, start=1):
        run_started = time.time()
        logger.info(
            "[%d/%d] %s",
            index,
            len(specs),
            spec.synthesize_run_id(),
        )
        try:
            steps, finals = run_experiment(spec)
            step_frames.append(steps)
            final_frames.append(finals)
            logger.info("completed in %.2fs", time.time() - run_started)
        except Exception as exc:  # continue a long sweep and preserve failures
            logger.exception("experiment failed")
            errors.append(
                {
                    "run_id": spec.synthesize_run_id(),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    if not step_frames or not final_frames:
        raise RuntimeError("No experiments completed successfully")

    args.out.mkdir(parents=True, exist_ok=True)
    steps = pd.concat(step_frames, ignore_index=True)
    finals = pd.concat(final_frames, ignore_index=True)
    _append_or_write(steps, args.out / "steps.csv", append=args.append)
    _append_or_write(finals, args.out / "finals.csv", append=args.append)
    comparison = build_schedule_comparison(finals)
    if not comparison.empty:
        _append_or_write(
            comparison,
            args.out / "schedule_comparison.csv",
            append=args.append,
        )
    if errors:
        _append_or_write(
            pd.DataFrame(errors),
            args.out / "errors.csv",
            append=args.append,
        )

    manifest = {
        "model": "branch-resolved noisy SWAP PQES",
        "circuit_kind": circuit_kind.value,
        "M_values": M_values,
        "p_1q_values": p_values,
        "purification_levels_per_cycle": ell_values,
        "cycles": cycles,
        "noise_families": [
            family.value for family in _families(args.noise)
        ],
        "schedules": [
            PurificationSchedule.NONE.value,
            *[schedule.value for schedule in purified_schedules],
        ],
        "seed": args.seed,
        "rate_multipliers": {
            "p_2q": args.p2_multiplier,
            "p_cswap": args.cswap_multiplier,
            "p_memory": args.memory_multiplier,
            "p_measure": args.measure_multiplier,
            "p_reset": args.reset_multiplier,
        },
        "all_swap_branches_retained": True,
        "signed_estimator": True,
        "elapsed_seconds": time.time() - started,
        "completed_runs": len(finals),
        "matched_schedule_pairs": len(comparison),
        "failed_runs": len(errors),
    }
    with (args.out / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    logger.info(
        "Saved %d step rows and %d final rows to %s in %.1fs",
        len(steps),
        len(finals),
        args.out,
        time.time() - started,
    )


if __name__ == "__main__":
    main()
