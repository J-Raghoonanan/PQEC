"""
Main entry point to run a grid sweep of purification simulations.

This script uses density-matrix simulation and implements the rho2
purification protocol (ρ → ρ²/Tr(ρ²)).  It writes CSVs under the
directory specified by --out, which are consumed by figure-generation scripts.

KEY DIFFERENCES FROM SWAP PURIFICATION:
- Uses ρ → ρ²/Tr(ρ²) instead of SWAP test
- P_success = Tr(ρ²) at each merge (deterministic, no postselection failure)
- C_ℓ = 2^ℓ exactly (no overhead from failed attempts)

SWEEP STRUCTURE:
  M ∈ {2, 3, 4}  →  N_LIST = [2]          (1 cycle only)
                     L_LIST = [0..10]       (full purification-level range)

  M ∈ {1, 5}     →  N_LIST = [2..1024]     (full cycle range)
                     N = 2  : L_LIST = [0..10]   (full)
                     N > 2  : L_LIST = [0..5]    (reduced; higher ℓ at many
                                                   cycles is very slow for M=5)

CHOICES (documented):
- Memory cap at M=6: density matrices scale as 4^M.
- iid_p noise mode: every copy statistically identical, consistent with ρ⊗ρ model.
- Clifford twirling auto-enabled for dephasing noise types.

Usage examples:

    python -m src.simulation.rho2_sims.main_grid_run \
        --out data/rho2_sim \
        --noise depol \
        --m-values 1 2 3 4 5 \
        --iterative

    python -m src.simulation.rho2_sims.main_grid_run \
        --out data/rho2_sim \
        --noise z \
        --m-values 1 2 3 4 5 \
        --iterative

    python -m src.simulation.rho2_sims.main_grid_run \
        --out data/rho2_sim \
        --noise z \
        --m-values 1 2 3 4 5 \
        --iterative \
        --no-twirl

    python -m src.simulation.rho2_sims.main_grid_run \
        --out data/rho2_sim \
        --noise z \
        --m-values 1 2 3 4 5 \
        --iterative \
        --no-twirl \
        --target single_qubit_product

It will append to `steps_rho2_*.csv` and `finals_rho2_*.csv`.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional
import numpy as np

from .configs import (
    RunSpec,
    TargetSpec,
    NoiseSpec,
    AASpec,
    TwirlingSpec,
    NoiseType,
    NoiseMode,
    StateKind,
)
from .streaming_runner import run_and_save

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Sweep parameter constants
# ─────────────────────────────────────────────────────────────────────────────

M_LIST: List[int] = [1, 2, 3, 4, 5]

# M values that receive the full N sweep (many cycles)
M_FULL_SWEEP = {1, 5}
# M values restricted to a single cycle (N=2 only)
M_SINGLE_CYCLE = {2, 3, 4}

# N lists
N_LIST_FULL: List[int]   = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
N_LIST_SINGLE: List[int] = [2]

# L (purification-level) lists
L_LIST_FULL: List[int]  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
L_LIST_SHORT: List[int] = [0, 1, 2, 3, 4, 5]   # used for M∈{1,5} when N > 2

P_LIST: List[float] = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NOISES: List[NoiseType] = [NoiseType.depolarizing, NoiseType.dephase_z]
TARGET_KIND: StateKind  = StateKind.hadamard
BACKEND_METHOD: str     = "density_matrix"

# AA configuration (not used by rho2 but kept for API compatibility)
AA = AASpec(target_success=0.99, max_iters=32, use_postselection_only=False)


# ─────────────────────────────────────────────────────────────────────────────
# Sweep-schedule helpers
# ─────────────────────────────────────────────────────────────────────────────

def _n_list_for(M: int) -> List[int]:
    """Return the N sweep list appropriate for qubit count M."""
    return N_LIST_FULL if M in M_FULL_SWEEP else N_LIST_SINGLE


def _l_list_for(M: int, N: int) -> List[int]:
    """Return the L sweep list appropriate for (M, N).

    M ∈ {2,3,4}             →  L_LIST_FULL   [0..10]
    M ∈ {1,5},  N = 2       →  L_LIST_FULL   [0..10]
    M ∈ {1,5},  N > 2       →  L_LIST_SHORT  [0..5]
    """
    if M in M_FULL_SWEEP and N > 2:
        return L_LIST_SHORT
    return L_LIST_FULL


def _count_total_runs(
    noises: List[NoiseType],
    Ms: List[int],
    ps: List[float],
) -> int:
    """Compute total run count given M-dependent N and L schedules."""
    count = 0
    for _noise in noises:
        for M in Ms:
            for N in _n_list_for(M):
                count += len(ps) * len(_l_list_for(M, N))
    return count


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid sweep for rho2 purification")
    p.add_argument("--out", type=Path, default=Path("data/rho2_sim"),
                   help="Output directory for CSVs")
    p.add_argument("--max-m", type=int, default=6,
                   help="Maximum M to include (≤ 6 recommended)")
    p.add_argument("--m-values", type=int, nargs='+',
                   help="Specific M values to run (e.g., --m-values 1 3 5)")
    p.add_argument("--seed", type=int, default=1,
                   help="Seed for target-state generation")
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
        help="Noise application mode (iid_p recommended)",
    )
    p.add_argument(
        "--target",
        choices=[k.value for k in StateKind],
        default=TARGET_KIND.value,
        help="Target state family for |ψ⟩",
    )
    p.add_argument(
        "--no-twirl",
        action="store_true",
        help="Disable Clifford twirling even for dephasing noise",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke-test: reduced M, N, p, L parameter space",
    )
    p.add_argument(
        "--iterative",
        action="store_true",
        help="Apply fresh noise before each iteration round (recommended)",
    )
    p.add_argument(
        "--purification-level",
        type=int,
        default=None,
        help="Pin ℓ to a single value rather than sweeping L_LIST "
             "(overrides the M/N-dependent L schedule)",
    )
    p.add_argument(
        "--theta",
        type=float,
        default=np.pi/3,
        help="Theta for single_qubit_product state (default: π/3)",
    )
    p.add_argument(
        "--phi",
        type=float,
        default=np.pi/4,
        help="Phi for single_qubit_product state (default: π/4)",
    )
    return p.parse_args()


def _pick_noises(flag: str) -> List[NoiseType]:
    if flag == "all":    return NOISES
    if flag == "depol":  return [NoiseType.depolarizing]
    if flag == "z":      return [NoiseType.dephase_z]
    if flag == "x":      return [NoiseType.dephase_x]
    raise ValueError(flag)


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    noises      = _pick_noises(args.noise)
    mode        = NoiseMode(args.mode)
    target_kind = StateKind(args.target)

    if args.m_values:
        Ms = [m for m in args.m_values if m <= 6]
        if not Ms:
            raise ValueError("No valid M values provided (must be ≤ 6)")
        logger.info(f"Using specific M values: {Ms}")
    else:
        Ms = [m for m in M_LIST if m <= args.max_m]
        logger.info(f"Using M range: 1 to {args.max_m}")

    # --purification-level pins ℓ to a single value, bypassing the schedule
    fixed_ell: Optional[int] = args.purification_level

    # Quick mode uses small fixed lists regardless of M/N schedule
    QUICK_N = [2, 4]
    QUICK_L = [0, 1]
    QUICK_P = [0.1, 0.5, 0.9]

    ps = QUICK_P if args.quick else P_LIST
    if args.quick:
        Ms = [m for m in [1, 2] if m in Ms] or Ms[:2]
        logger.info("QUICK TEST MODE: reduced parameter space")

    twirling = TwirlingSpec(enabled=not args.no_twirl, mode="cyclic", seed=args.seed)

    # ── Summary log ──────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("rho2 grid sweep — N and L schedules:")
    logger.info(f"  M ∈ {{1,5}}, N=2   → L = {L_LIST_FULL}")
    logger.info(f"  M ∈ {{1,5}}, N>2   → L = {L_LIST_SHORT}")
    logger.info(f"  M ∈ {{2,3,4}}      → N = {N_LIST_SINGLE},  L = {L_LIST_FULL}")
    if fixed_ell is not None:
        logger.info(f"  ℓ override        = {fixed_ell}  (--purification-level)")
    logger.info(f"  Ms       = {Ms}")
    logger.info(f"  ps       = {ps}")
    logger.info(f"  noises   = {[n.value for n in noises]}")
    logger.info(f"  target   = {target_kind.value}")
    logger.info(f"  twirling = {'enabled' if twirling.enabled else 'disabled'}")
    logger.info(f"  iterative= {args.iterative}")
    logger.info(f"  out_dir  = {out_dir}")
    logger.info("=" * 70)

    # Compute total runs accurately given the variable schedule
    if args.quick:
        total_runs = len(noises) * len(Ms) * len(QUICK_N) * len(ps) * len(QUICK_L)
    elif fixed_ell is not None:
        total_runs = len(noises) * len(ps) * sum(len(_n_list_for(M)) for M in Ms)
    else:
        total_runs = _count_total_runs(noises, Ms, ps)

    started     = time.time()
    current_run = 0

    for noise in noises:
        for M in Ms:
            target = TargetSpec(
                M=M,
                kind=target_kind,
                seed=args.seed,
                product_theta=args.theta,
                product_phi=args.phi,
            )

            Ns_for_M = QUICK_N if args.quick else _n_list_for(M)

            for N in Ns_for_M:
                if fixed_ell is not None:
                    Ls_for_N = [fixed_ell]
                elif args.quick:
                    Ls_for_N = QUICK_L
                else:
                    Ls_for_N = _l_list_for(M, N)

                for p in ps:
                    for ell in Ls_for_N:
                        current_run += 1

                        spec = RunSpec(
                            target=target,
                            noise=NoiseSpec(noise_type=noise, mode=mode, p=p),
                            aa=AA,
                            twirling=twirling,
                            N=N,
                            backend_method=BACKEND_METHOD,
                            out_dir=out_dir,
                            verbose=args.verbose,
                            iterative_noise=args.iterative,
                            purification_level=ell,
                        )

                        tag = spec.synthesize_run_id()

                        logger.info(f"\n{'=' * 70}")
                        logger.info(
                            f"Run {current_run}/{total_runs}: {tag}  "
                            f"(M={M}, N={N}, ℓ={ell})"
                        )
                        logger.info(f"{'=' * 70}")

                        t0 = time.time()
                        try:
                            run_and_save(spec)
                            logger.info(f"✓ Completed in {time.time() - t0:.1f}s\n")
                        except Exception as e:
                            logger.error(
                                f"✗ ERROR during {tag}: {e}\n", exc_info=True
                            )

    total = time.time() - started
    logger.info(f"\n{'=' * 70}")
    logger.info(f"rho2 grid sweep complete!")
    logger.info(f"  Total time : {total / 60:.1f} min")
    logger.info(f"  Runs       : {current_run}/{total_runs}")
    logger.info(f"  Data saved : {out_dir}")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()