"""Focused correctness tests for the circuit-level noisy PQES model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
from .pqes import merge_branch_aggregates
from .runner import run_experiment


def _mixed_qubit_state() -> np.ndarray:
    return np.array([[0.70, 0.20], [0.20, 0.30]], dtype=complex)


def test_ideal_swap_record_returns_rho_squared() -> None:
    rho = _mixed_qubit_state()
    hardware = HardwareNoiseSpec.uniform(NoiseFamily.DEPOLARIZING, 0.0)

    unsigned, signed, diagnostics = merge_branch_aggregates(
        rho,
        rho,
        hardware,
    )

    np.testing.assert_allclose(unsigned, rho, atol=1e-12)
    np.testing.assert_allclose(signed, rho @ rho, atol=1e-12)
    assert diagnostics.branch_probability_plus == pytest.approx(0.83)
    assert diagnostics.branch_probability_minus == pytest.approx(0.17)


def test_readout_error_scales_signed_record_not_normalized_state() -> None:
    rho = _mixed_qubit_state()
    ideal = HardwareNoiseSpec.uniform(NoiseFamily.DEPOLARIZING, 0.0)
    readout_noisy = HardwareNoiseSpec(
        family=NoiseFamily.DEPOLARIZING,
        p_measure=0.1,
    )
    _, signed_ideal, _ = merge_branch_aggregates(rho, rho, ideal)
    _, signed_noisy, _ = merge_branch_aggregates(
        rho,
        rho,
        readout_noisy,
    )

    np.testing.assert_allclose(signed_noisy, 0.8 * signed_ideal, atol=1e-12)
    np.testing.assert_allclose(
        signed_noisy / np.trace(signed_noisy),
        signed_ideal / np.trace(signed_ideal),
        atol=1e-12,
    )


@pytest.mark.parametrize("schedule", list(PurificationSchedule))
def test_zero_noise_circuit_has_unit_fidelity(
    schedule: PurificationSchedule,
) -> None:
    levels = 0 if schedule == PurificationSchedule.NONE else 1
    spec = ExperimentSpec(
        circuit=CircuitSpec(
            M=2,
            cycles=2,
            kind=CircuitKind.BRICKWORK,
            seed=3,
        ),
        hardware=HardwareNoiseSpec.uniform(
            NoiseFamily.DEPOLARIZING,
            0.0,
        ),
        purification=PurificationSpec(
            schedule=schedule,
            levels_per_cycle=levels,
        ),
    )

    steps, finals = run_experiment(spec)
    np.testing.assert_allclose(steps["fidelity"], 1.0, atol=1e-12)
    assert finals.iloc[0]["fidelity_final"] == pytest.approx(1.0)


def test_noisy_run_schema_and_resources() -> None:
    spec = ExperimentSpec(
        circuit=CircuitSpec(
            M=1,
            cycles=2,
            kind=CircuitKind.ECHO,
            seed=1,
        ),
        hardware=HardwareNoiseSpec.uniform(
            NoiseFamily.DEPHASING,
            0.01,
        ),
        purification=PurificationSpec(
            schedule=PurificationSchedule.INTERLEAVED,
            levels_per_cycle=1,
        ),
    )

    steps, finals = run_experiment(spec)
    assert len(steps) == 2
    assert {
        "fidelity",
        "physical_fidelity",
        "signed_trace",
        "log_abs_signed_trace",
        "sampling_overhead",
        "log10_sampling_overhead",
        "p_cswap",
        "schedule",
    }.issubset(steps.columns)
    final = finals.iloc[0]
    assert final["leaf_copies"] == 4
    assert final["swap_gadgets"] == 3
    assert final["physical_cswap_locations"] == 3
    assert np.isfinite(final["fidelity_final"])


def test_deep_signed_tree_uses_finite_log_space_metrics() -> None:
    spec = ExperimentSpec(
        circuit=CircuitSpec(
            M=1,
            cycles=5,
            kind=CircuitKind.BRICKWORK,
            seed=7,
        ),
        hardware=HardwareNoiseSpec.uniform(
            NoiseFamily.DEPHASING,
            0.1,
        ),
        purification=PurificationSpec(
            schedule=PurificationSchedule.INTERLEAVED,
            levels_per_cycle=3,
        ),
    )

    steps, finals = run_experiment(spec)
    assert np.isfinite(steps["fidelity"]).all()
    assert np.isfinite(steps["log_abs_signed_trace"]).all()
    assert np.isfinite(steps["log10_sampling_overhead"]).all()
    assert np.isfinite(finals.iloc[0]["fidelity_final"])


def test_matched_schedule_comparison_table() -> None:
    frames = []
    for schedule, levels in (
        (PurificationSchedule.NONE, 0),
        (PurificationSchedule.INTERLEAVED, 1),
        (PurificationSchedule.END_ONLY, 1),
    ):
        spec = ExperimentSpec(
            circuit=CircuitSpec(
                M=1,
                cycles=1,
                kind=CircuitKind.BRICKWORK,
                seed=4,
            ),
            hardware=HardwareNoiseSpec.uniform(
                NoiseFamily.DEPOLARIZING,
                0.01,
            ),
            purification=PurificationSpec(
                schedule=schedule,
                levels_per_cycle=levels,
            ),
        )
        _, final = run_experiment(spec)
        frames.append(final)

    comparison = build_schedule_comparison(pd.concat(frames, ignore_index=True))
    assert len(comparison) == 1
    assert comparison.iloc[0]["matched_leaf_copies"] == 2
    assert {
        "delta_fidelity_interleaved_minus_end_only",
        "preferred_schedule_by_fidelity",
        "fidelity_final_no_pqes",
    }.issubset(comparison.columns)
