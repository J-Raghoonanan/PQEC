"""Trajectory runner for branch-resolved noisy-circuit PQES."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from qiskit.quantum_info import Pauli, Statevector

from .channels import (
    apply_memory_noise,
    apply_noisy_circuit,
    reset_product_state,
)
from .circuits import (
    build_preparation_circuit,
    build_workload_blocks,
    count_workload_gates,
)
from .config import ExperimentSpec, PurificationSchedule
from .pqes import MergeDiagnostics, merge_branch_aggregates


def _normalized_operator(operator: np.ndarray) -> tuple[np.ndarray, float]:
    normalization = float(np.real(np.trace(operator)))
    if normalization == 0.0 or not np.isfinite(normalization):
        return np.full_like(operator, np.nan), normalization
    return operator / normalization, normalization


def _true_normalization_metrics(
    rescaled_normalization: float,
    log_abs_scale: float,
) -> tuple[float, float, float, float]:
    """Recover trace and overhead diagnostics without numerical underflow."""
    if rescaled_normalization == 0.0 or not np.isfinite(
        rescaled_normalization
    ):
        return 0.0, -np.inf, np.inf, np.inf

    log_abs_normalization = (
        float(log_abs_scale) + float(np.log(abs(rescaled_normalization)))
    )
    sign = float(np.sign(rescaled_normalization))
    smallest_log = float(np.log(np.nextafter(0.0, 1.0)))
    largest_log = float(np.log(np.finfo(float).max))
    if log_abs_normalization < smallest_log:
        normalization = 0.0
    elif log_abs_normalization > largest_log:
        normalization = sign * np.inf
    else:
        normalization = sign * float(np.exp(log_abs_normalization))

    log10_sampling_overhead = (
        -2.0 * log_abs_normalization / float(np.log(10.0))
    )
    max_log10 = float(np.log10(np.finfo(float).max))
    if log10_sampling_overhead > max_log10:
        sampling_overhead = np.inf
    else:
        sampling_overhead = float(10.0**log10_sampling_overhead)
    return (
        normalization,
        log_abs_normalization,
        sampling_overhead,
        log10_sampling_overhead,
    )


def _operator_metrics(
    operator: np.ndarray,
    target: Statevector,
    *,
    log_abs_scale: float = 0.0,
) -> Dict[str, float]:
    normalized, rescaled_normalization = _normalized_operator(operator)
    (
        normalization,
        log_abs_normalization,
        sampling_overhead,
        log10_sampling_overhead,
    ) = _true_normalization_metrics(
        rescaled_normalization,
        log_abs_scale,
    )
    if np.isnan(normalized).any():
        return {
            "fidelity": np.nan,
            "eps_L": np.nan,
            "purity": np.nan,
            "normalization": normalization,
            "log_abs_normalization": log_abs_normalization,
            "sampling_overhead": sampling_overhead,
            "log10_sampling_overhead": log10_sampling_overhead,
            "bloch_r": np.nan,
        }

    vector = target.data
    fidelity = float(np.real(np.vdot(vector, normalized @ vector)))
    purity = float(np.real(np.trace(normalized @ normalized)))
    target_density = np.outer(vector, np.conj(vector))
    difference = (normalized - target_density)
    difference = (difference + difference.conj().T) / 2.0
    eps_L = 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(difference))))

    bloch_r = np.nan
    if normalized.shape == (2, 2):
        components = [
            np.real(np.trace(normalized @ Pauli(label).to_matrix()))
            for label in ("X", "Y", "Z")
        ]
        bloch_r = float(np.linalg.norm(components))

    return {
        "fidelity": fidelity,
        "eps_L": eps_L,
        "purity": purity,
        "normalization": normalization,
        "log_abs_normalization": log_abs_normalization,
        "sampling_overhead": sampling_overhead,
        "log10_sampling_overhead": log10_sampling_overhead,
        "bloch_r": bloch_r,
    }


def _rescale_signed_operator(
    operator: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Keep a signed operator numerically O(1), returning log of its scale."""
    scale = float(np.linalg.norm(operator, ord="fro"))
    if scale == 0.0 or not np.isfinite(scale):
        return operator, -np.inf
    return operator / scale, float(np.log(scale))


def _apply_purification_levels(
    unsigned: np.ndarray,
    signed: np.ndarray,
    *,
    signed_log_abs_scale: float,
    levels: int,
    spec: ExperimentSpec,
) -> tuple[np.ndarray, np.ndarray, float, Optional[MergeDiagnostics]]:
    diagnostics: Optional[MergeDiagnostics] = None
    for _ in range(levels):
        unsigned, signed, diagnostics = merge_branch_aggregates(
            unsigned,
            signed,
            spec.hardware,
        )
        signed, level_log_abs_scale = _rescale_signed_operator(signed)
        signed_log_abs_scale = (
            2.0 * signed_log_abs_scale + level_log_abs_scale
        )
    return unsigned, signed, signed_log_abs_scale, diagnostics


def _physical_workload_resources(
    spec: ExperimentSpec,
    workload_gate_counts: List[Tuple[int, int]],
) -> Dict[str, int]:
    """Gate-location counts needed to produce one final estimator output."""
    leaf_copies = spec.leaf_copies
    levels = spec.purification.levels_per_cycle
    schedule = spec.purification.schedule

    preparation_1q = leaf_copies * spec.circuit.M
    workload_1q = 0
    workload_2q = 0
    if schedule == PurificationSchedule.INTERLEAVED:
        total_depth = spec.total_purification_depth
        for cycle, (one_qubit, two_qubit) in enumerate(workload_gate_counts):
            active_registers = 2 ** (total_depth - cycle * levels)
            workload_1q += active_registers * one_qubit
            workload_2q += active_registers * two_qubit
    else:
        for one_qubit, two_qubit in workload_gate_counts:
            workload_1q += leaf_copies * one_qubit
            workload_2q += leaf_copies * two_qubit

    swap_gadgets = max(0, leaf_copies - 1)
    return {
        "leaf_copies": leaf_copies,
        "swap_gadgets": swap_gadgets,
        "physical_1q_gate_locations": (
            preparation_1q + workload_1q + 2 * swap_gadgets
        ),
        "physical_2q_gate_locations": workload_2q,
        "physical_cswap_locations": spec.circuit.M * swap_gadgets,
        "physical_measurement_locations": swap_gadgets,
        "physical_reset_locations": (
            spec.circuit.M * leaf_copies + swap_gadgets
        ),
        "physical_memory_locations": (
            spec.circuit.M
            * (1 + 2 * spec.circuit.M)
            * swap_gadgets
        ),
    }


def run_experiment(spec: ExperimentSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one complete circuit trajectory and return steps/finals tables."""
    spec.validate()
    M = spec.circuit.M
    levels_per_cycle = spec.purification.levels_per_cycle
    run_id = spec.synthesize_run_id()

    preparation = build_preparation_circuit(M)
    workload_blocks = build_workload_blocks(spec.circuit)
    workload_gate_counts = [
        count_workload_gates(block) for block in workload_blocks
    ]

    ideal_target = Statevector.from_label("0" * M).evolve(preparation)
    unsigned = reset_product_state(M, spec.hardware.p_reset)
    unsigned, prep_1q_count, prep_2q_count = apply_noisy_circuit(
        unsigned,
        preparation,
        spec.hardware,
    )
    if prep_2q_count != 0:  # pragma: no cover - construction invariant
        raise AssertionError("The |+> preparation should contain only H gates")
    signed = unsigned.copy()
    signed_log_abs_scale = 0.0

    steps_rows: List[Dict[str, object]] = []
    purification_depth_completed = 0
    cumulative_1q_per_branch = prep_1q_count
    cumulative_2q_per_branch = 0
    cumulative_cswap_per_branch = 0
    cumulative_memory_per_branch = 0
    first_pre_purification_metrics: Optional[Dict[str, float]] = None

    for cycle_index, block in enumerate(workload_blocks, start=1):
        before_metrics = _operator_metrics(
            signed,
            ideal_target,
            log_abs_scale=signed_log_abs_scale,
        )
        before_physical = _operator_metrics(unsigned, ideal_target)

        ideal_target = ideal_target.evolve(block)
        unsigned, one_qubit_count, two_qubit_count = apply_noisy_circuit(
            unsigned,
            block,
            spec.hardware,
        )
        signed, _, _ = apply_noisy_circuit(
            signed,
            block,
            spec.hardware,
        )
        unsigned = apply_memory_noise(
            unsigned,
            qargs=range(M),
            spec=spec.hardware,
        )
        signed = apply_memory_noise(
            signed,
            qargs=range(M),
            spec=spec.hardware,
        )
        cumulative_1q_per_branch += one_qubit_count
        cumulative_2q_per_branch += two_qubit_count
        cumulative_memory_per_branch += M

        after_circuit = _operator_metrics(
            signed,
            ideal_target,
            log_abs_scale=signed_log_abs_scale,
        )
        after_circuit_physical = _operator_metrics(unsigned, ideal_target)
        if first_pre_purification_metrics is None:
            first_pre_purification_metrics = dict(after_circuit)

        levels_now = 0
        if spec.purification.schedule == PurificationSchedule.INTERLEAVED:
            levels_now = levels_per_cycle
        elif (
            spec.purification.schedule == PurificationSchedule.END_ONLY
            and cycle_index == spec.circuit.cycles
        ):
            levels_now = spec.total_purification_depth

        diagnostics: Optional[MergeDiagnostics] = None
        if levels_now:
            (
                unsigned,
                signed,
                signed_log_abs_scale,
                diagnostics,
            ) = _apply_purification_levels(
                unsigned,
                signed,
                signed_log_abs_scale=signed_log_abs_scale,
                levels=levels_now,
                spec=spec,
            )
            purification_depth_completed += levels_now
            cumulative_1q_per_branch += 2 * levels_now
            cumulative_cswap_per_branch += M * levels_now
            cumulative_memory_per_branch += (
                M * (1 + 2 * M) * levels_now
            )

        output_metrics = _operator_metrics(
            signed,
            ideal_target,
            log_abs_scale=signed_log_abs_scale,
        )
        output_physical = _operator_metrics(unsigned, ideal_target)
        copies_used = 2**purification_depth_completed

        steps_rows.append(
            {
                "run_id": run_id,
                "iteration": cycle_index,
                "merge_num": cycle_index - 1,
                "M": M,
                "depth": purification_depth_completed,
                "copies_used": copies_used,
                "N_so_far": copies_used,
                "noise": spec.hardware.family.value,
                "mode": "gate_level",
                "p": spec.hardware.base_error_rate,
                "p_channel": spec.hardware.base_error_rate,
                "P_success": 1.0,
                "grover_iters": 0,
                "twirling_applied": False,
                "purification_level": levels_per_cycle,
                "schedule": spec.purification.schedule.value,
                "circuit_kind": spec.circuit.kind.value,
                "circuit_seed": spec.circuit.seed,
                "fidelity": output_metrics["fidelity"],
                "eps_L": output_metrics["eps_L"],
                "purity": output_metrics["purity"],
                "bloch_r": output_metrics["bloch_r"],
                "physical_fidelity": output_physical["fidelity"],
                "physical_eps_L": output_physical["eps_L"],
                "physical_purity": output_physical["purity"],
                "signed_trace": output_metrics["normalization"],
                "log_abs_signed_trace": output_metrics[
                    "log_abs_normalization"
                ],
                "sampling_overhead": output_metrics["sampling_overhead"],
                "log10_sampling_overhead": output_metrics[
                    "log10_sampling_overhead"
                ],
                "fidelity_before_noise": before_metrics["fidelity"],
                "fidelity_after_noise": after_circuit["fidelity"],
                "eps_L_before_noise": before_metrics["eps_L"],
                "eps_L_after_noise": after_circuit["eps_L"],
                "purity_before_noise": before_metrics["purity"],
                "purity_after_noise": after_circuit["purity"],
                "physical_fidelity_before_circuit": before_physical["fidelity"],
                "physical_fidelity_after_circuit": (
                    after_circuit_physical["fidelity"]
                ),
                "purification_levels_this_cycle": levels_now,
                "branch_probability_plus": (
                    diagnostics.branch_probability_plus
                    if diagnostics is not None
                    else np.nan
                ),
                "branch_probability_minus": (
                    diagnostics.branch_probability_minus
                    if diagnostics is not None
                    else np.nan
                ),
                "p_1q": spec.hardware.p_1q,
                "p_2q": spec.hardware.p_2q,
                "p_cswap": spec.hardware.p_cswap,
                "p_memory": spec.hardware.p_memory,
                "p_measure": spec.hardware.p_measure,
                "p_reset": spec.hardware.p_reset,
                "one_qubit_gates_per_branch": cumulative_1q_per_branch,
                "two_qubit_gates_per_branch": cumulative_2q_per_branch,
                "cswap_gates_per_branch": cumulative_cswap_per_branch,
                "memory_locations_per_branch": cumulative_memory_per_branch,
            }
        )

    if first_pre_purification_metrics is None:  # pragma: no cover
        raise AssertionError("At least one circuit cycle is required")

    final_signed = _operator_metrics(
        signed,
        ideal_target,
        log_abs_scale=signed_log_abs_scale,
    )
    final_physical = _operator_metrics(unsigned, ideal_target)
    initial_eps = float(first_pre_purification_metrics["eps_L"])
    error_reduction_ratio = (
        final_signed["eps_L"] / initial_eps if initial_eps > 0.0 else np.nan
    )
    resources = _physical_workload_resources(spec, workload_gate_counts)

    finals_row: Dict[str, object] = {
        "run_id": run_id,
        "M": M,
        "N": resources["leaf_copies"],
        "noise": spec.hardware.family.value,
        "mode": "gate_level",
        "p": spec.hardware.base_error_rate,
        "p_channel": spec.hardware.base_error_rate,
        "fidelity_init": first_pre_purification_metrics["fidelity"],
        "fidelity_final": final_signed["fidelity"],
        "eps_L_init": first_pre_purification_metrics["eps_L"],
        "eps_L_final": final_signed["eps_L"],
        "purity_init": first_pre_purification_metrics["purity"],
        "purity_final": final_signed["purity"],
        "error_reduction_ratio": error_reduction_ratio,
        "max_depth": spec.total_purification_depth,
        "iterations": spec.circuit.cycles,
        "purification_level": levels_per_cycle,
        "twirling_enabled": False,
        "schedule": spec.purification.schedule.value,
        "circuit_kind": spec.circuit.kind.value,
        "circuit_seed": spec.circuit.seed,
        "physical_fidelity_final": final_physical["fidelity"],
        "physical_eps_L_final": final_physical["eps_L"],
        "physical_purity_final": final_physical["purity"],
        "signed_trace_final": final_signed["normalization"],
        "log_abs_signed_trace_final": final_signed[
            "log_abs_normalization"
        ],
        "sampling_overhead_final": final_signed["sampling_overhead"],
        "log10_sampling_overhead_final": final_signed[
            "log10_sampling_overhead"
        ],
        "p_1q": spec.hardware.p_1q,
        "p_2q": spec.hardware.p_2q,
        "p_cswap": spec.hardware.p_cswap,
        "p_memory": spec.hardware.p_memory,
        "p_measure": spec.hardware.p_measure,
        "p_reset": spec.hardware.p_reset,
        **resources,
    }
    if M == 1:
        finals_row["bloch_r_init"] = first_pre_purification_metrics["bloch_r"]
        finals_row["bloch_r_final"] = final_signed["bloch_r"]

    return pd.DataFrame(steps_rows), pd.DataFrame([finals_row])


def save_experiment(
    spec: ExperimentSpec,
    *,
    steps_name: str = "steps.csv",
    finals_name: str = "finals.csv",
) -> tuple[Path, Path]:
    """Run one experiment and append it to the clean noisy-circuit tables."""
    steps, finals = run_experiment(spec)
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    steps_path = spec.out_dir / steps_name
    finals_path = spec.out_dir / finals_name

    if steps_path.exists():
        previous = pd.read_csv(steps_path)
        steps = pd.concat([previous, steps], ignore_index=True)
    if finals_path.exists():
        previous = pd.read_csv(finals_path)
        finals = pd.concat([previous, finals], ignore_index=True)

    steps.to_csv(steps_path, index=False)
    finals.to_csv(finals_path, index=False)
    return steps_path, finals_path


__all__ = ["run_experiment", "save_experiment"]
