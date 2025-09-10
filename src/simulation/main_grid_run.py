"""
Main entry point to run a grid sweep of circuit-level purification simulations.

This script uses the *density-matrix* Aer simulator and the SWAP-test based
purification implemented in this package. It writes CSVs under
`data/simulations/` that are directly consumable by your figure-generation
scripts.

CHOICES (documented):
- We cap M at 6 by default because the density matrix for the merge uses
  1 + 2M qubits (ancilla + two registers). Memory/time grows quickly with M.
- We run **i.i.d. per-qubit channels** (NoiseMode.iid_p) so every input copy is
  statistically identical. This aligns with the ρ ⊗ ρ model in the manuscript
  and keeps results clean and reproducible.
- Target state defaults to **Hadamard** (H^⊗M |0…0⟩). You can change to Haar
  or random circuits by editing the `TARGET_KIND` constant below.
- Amplitude amplification is **emulated**: we compute and log the required
  Grover iterations but we don't explicitly apply Q^k. The postselected output
  state (ancilla = 0) is identical either way.

You can run this file directly:

    python -m src.simulation.main_grid_run \
        --out data/simulations \
        --max-m 6

It will append to `steps_circuit.csv` and `finals_circuit.csv`.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, List

from .configs import (
    RunSpec,
    TargetSpec,
    NoiseSpec,
    AASpec,
    NoiseType,
    NoiseMode,
    StateKind,
)
from src.simulation.streaming_runner import run_and_save

# -----------------------------
# Defaults for the sweep
# -----------------------------
M_LIST: List[int] = [1, 2, 3, 4, 5, 6]  # keep ≤ 6 for density-matrix practicality
N_LIST: List[int] = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
DELTA_LIST: List[float] = [0.01, 0.1, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
NOISES: List[NoiseType] = [NoiseType.depolarizing, NoiseType.dephase_z, NoiseType.dephase_x]
TARGET_KIND: StateKind = StateKind.hadamard  # change to StateKind.haar for random pure states
BACKEND_METHOD: str = "density_matrix"

# AA configuration (emulated)
AA = AASpec(target_success=0.99, max_iters=32, use_postselection_only=False)


# -----------------------------
# CLI
# -----------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid sweep for SWAP-QEC circuit simulation")
    p.add_argument("--out", type=Path, default=Path("data/simulations"), help="Output directory for CSVs")
    p.add_argument("--max-m", type=int, default=6, help="Maximum M to include (≤ 6 recommended)")
    p.add_argument("--seed", type=int, default=1, help="Seed for target-state generation where applicable")
    p.add_argument(
        "--noise",
        choices=["all", "depol", "z", "x"],
        default="all",
        help="Which noise families to simulate (default: all)",
    )
    p.add_argument(
        "--mode",
        choices=[m.value for m in NoiseMode],
        default=NoiseMode.iid_p.value,
        help="Noise application mode (iid_p is manuscript-consistent)",
    )
    p.add_argument(
        "--target",
        choices=[k.value for k in StateKind],
        default=TARGET_KIND.value,
        help="Target state family for |ψ⟩",
    )
    return p.parse_args()


def _pick_noises(flag: str) -> List[NoiseType]:
    if flag == "all":
        return NOISES
    if flag == "depol":
        return [NoiseType.depolarizing]
    if flag == "z":
        return [NoiseType.dephase_z]
    if flag == "x":
        return [NoiseType.dephase_x]
    raise ValueError(flag)


# -----------------------------
# Main sweep
# -----------------------------

def main() -> None:
    args = _parse_args()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    noises = _pick_noises(args.noise)
    mode = NoiseMode(args.mode)
    target_kind = StateKind(args.target)

    # Respect the M cap explicitly
    Ms = [m for m in M_LIST if m <= args.max_m]

    print(
        "Running grid sweep with:"
        f"\n  Ms           = {Ms}"
        f"\n  Ns           = {N_LIST}"
        f"\n  deltas       = {DELTA_LIST}"
        f"\n  noises       = {[n.value for n in noises]}"
        f"\n  mode         = {mode.value}"
        f"\n  target_kind  = {target_kind.value}"
        f"\n  backend      = {BACKEND_METHOD}"
        f"\n  out_dir      = {out_dir}"
    )

    started = time.time()

    for noise in noises:
        for M in Ms:
            # Target |ψ⟩ spec:
            target = TargetSpec(M=M, kind=target_kind, seed=args.seed)
            for N in N_LIST:
                for delta in DELTA_LIST:
                    spec = RunSpec(
                        target=target,
                        noise=NoiseSpec(noise_type=noise, mode=mode, delta=delta),
                        aa=AA,
                        N=N,
                        backend_method=BACKEND_METHOD,
                        out_dir=out_dir,
                    )
                    tag = spec.synthesize_run_id()
                    t0 = time.time()
                    try:
                        print(f"→ Running {tag} ...", flush=True)
                        run_and_save(spec)
                        dt = time.time() - t0
                        print(f"  done in {dt:.1f}s\n", flush=True)
                    except Exception as e:
                        # Keep sweeping on errors; log and continue
                        print(f"  ERROR during {tag}: {e}\n", flush=True)

    total = time.time() - started
    print(f"Grid sweep finished in {total/60:.1f} min. Data in {out_dir}")


if __name__ == "__main__":
    main()
