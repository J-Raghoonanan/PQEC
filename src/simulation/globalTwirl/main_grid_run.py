"""
Main entry point to run a grid sweep of circuit-level purification simulations
(GlobalTwirl variant).

Differences vs simulations_moreNoise:
- Iterative mode uses CHEAP global/parallel twirling (cycle one gate applied to all qubits)
  rather than exact local deterministic twirl over 3^M combinations.
- Automatically sweeps purification_level ℓ over {0,1,2,3} (no CLI flag).
- Defaults output directory to data/globalTwirl_simulations

python -m src.simulation.globalTwirl.main_grid_run \
  --out data/globalTwirl_simulations \
  --noise z \
  --m-values 1 5 \
  --iterative
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import List

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

# -----------------------------
# Defaults for the sweep
# -----------------------------
M_LIST: List[int] = [1, 2, 3, 4, 5]  # keep ≤ 6 for density-matrix practicality
N_LIST: List[int] = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
P_LIST: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.9]

NOISES: List[NoiseType] = [NoiseType.depolarizing, NoiseType.dephase_z]
TARGET_KIND: StateKind = StateKind.hadamard
BACKEND_METHOD: str = "density_matrix"

# Automatically sweep ℓ in {0,1,2,3}
L_LIST: List[int] = [0, 1, 2, 3]

# AA configuration (emulated)
AA = AASpec(target_success=0.99, max_iters=32, use_postselection_only=False)

# Twirling configuration (auto-enabled for dephasing)
TWIRLING = TwirlingSpec(enabled=True, mode="cyclic", seed=None)


# -----------------------------
# CLI
# -----------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid sweep for SWAP-QEC circuit simulation (GlobalTwirl)")
    p.add_argument("--out", type=Path, default=Path("data/globalTwirl_simulations"), help="Output directory for CSVs")
    p.add_argument("--max-m", type=int, default=6, help="Maximum M to include (≤ 6 recommended)")
    p.add_argument("--m-values", type=int, nargs='+', help="Specific M values to run (e.g., --m-values 1 3 5)")
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
        help="Run a quick test with reduced parameter space",
    )
    p.add_argument(
        "--iterative",
        action="store_true",
        help="Enable iterative noise mode (GlobalTwirl uses cheap global cycling in iterative mode)",
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

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    noises = _pick_noises(args.noise)
    mode = NoiseMode(args.mode)
    target_kind = StateKind(args.target)

    # Respect the M cap explicitly, or use specific values if provided
    if args.m_values:
        Ms = [m for m in args.m_values if m <= 6]
        if not Ms:
            raise ValueError("No valid M values provided (must be ≤ 6)")
        logger.info(f"Using specific M values: {Ms}")
    else:
        Ms = [m for m in M_LIST if m <= args.max_m]
        logger.info(f"Using M range: 1 to {args.max_m}")

    # Quick test mode: reduce parameter space
    if args.quick:
        logger.info("QUICK TEST MODE: Using reduced parameter space")
        Ms = Ms[:2] if len(Ms) > 2 else Ms
        Ns = [4, 16, 64]
        ps = [0.1, 0.5, 0.9]
        Ls = [0, 1]  # keep quick test short
    else:
        Ns = N_LIST
        ps = P_LIST
        Ls = L_LIST

    # Twirling config
    twirling = TwirlingSpec(enabled=not args.no_twirl, mode="cyclic", seed=args.seed)

    logger.info(
        "="*70 + "\n"
        "Running GLOBAL-TWIRL grid sweep with:\n"
        f"  Ms           = {Ms}\n"
        f"  Ns           = {Ns}\n"
        f"  ps           = {ps}\n"
        f"  ℓ values      = {Ls}\n"
        f"  noises       = {[n.value for n in noises]}\n"
        f"  mode         = {mode.value}\n"
        f"  target_kind  = {target_kind.value}\n"
        f"  backend      = {BACKEND_METHOD}\n"
        f"  twirling     = {'enabled' if twirling.enabled else 'disabled'}\n"
        f"  iterative    = {bool(args.iterative)}\n"
        f"  out_dir      = {out_dir}\n" +
        "="*70
    )

    started = time.time()
    total_runs = len(noises) * len(Ms) * len(Ns) * len(ps) * len(Ls)
    current_run = 0

    for noise in noises:
        for M in Ms:
            target = TargetSpec(M=M, kind=target_kind, seed=args.seed)
            for N in Ns:
                for p in ps:
                    for ell in Ls:
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

                        logger.info(f"\n{'='*70}")
                        logger.info(f"Run {current_run}/{total_runs}: {tag} (ℓ={ell})")
                        logger.info(f"{'='*70}")

                        t0 = time.time()
                        try:
                            run_and_save(spec)
                            dt = time.time() - t0
                            logger.info(f"✓ Completed in {dt:.1f}s\n")
                        except Exception as e:
                            logger.error(f"✗ ERROR during {tag}: {e}\n", exc_info=True)

    total = time.time() - started
    logger.info(f"\n{'='*70}")
    logger.info(f"Grid sweep complete!")
    logger.info(f"  Total time: {total/60:.1f} min")
    logger.info(f"  Runs: {current_run}/{total_runs}")
    logger.info(f"  Data saved to: {out_dir}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
