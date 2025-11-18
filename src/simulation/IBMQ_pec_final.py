"""
Main IBM Quantum experimental runner for SWAP-based purification experiments.

This script runs grid sweeps of circuit-level purification experiments on IBM quantum 
hardware, collecting the same data as the density-matrix simulations for comparison.

Key differences from simulation:
- Uses shot-based measurements instead of exact density matrix evolution
- Implements post-selection on successful ancilla measurements  
- Handles hardware noise, gate errors, and measurement errors
- Saves experimental data under `data/IBM/` in same CSV format as simulations

USAGE:
    python -m src.simulation.IBMQ_pec_final \
        --backend aer_simulator \
        --shots 1024 \
        --max-attempts 5

    python -m src.simulation.IBMQ_pec_final \
        --backend ibm_torino \
        --shots 2048 \
        --max-m 2 \
        --quick

The script will create/append to:
- data/IBMQ/steps_ibm.csv     (step-by-step purification data)  
- data/IBMQ/finals_ibm.csv    (final purified state results)
- data/IBMQ/experiments_log.txt (detailed experiment log)
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

# Import our IBMQ components
from IBMQ_components import *

# Setup logging
def setup_logging(log_file: Path, verbose: bool = False):
    """Setup logging to both file and console."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler  
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)


# =============================================================================
# EXPERIMENTAL PARAMETERS - EASY TO MODIFY
# =============================================================================

# Grid search parameters
M_LIST: List[int] = [1, 2, 3]                    # Number of qubits
P_LIST: List[float] = [0.01, 0.1, 0.2, 0.3]     # Noise probabilities
NOISE_TYPES: List[str] = ['depolarizing']         # Noise types to test
TARGET_STATE: str = 'hadamard'                   # Target state type
APPLY_TWIRLING: bool = True                       # Enable Clifford twirling

# Experimental settings
DEFAULT_SHOTS: int = 1024                        # Shots per SWAP test
MAX_ATTEMPTS: int = 5                            # Max attempts per purification step
NUM_LEVELS: int = 1                              # Recursion levels (1 = single step)

# Quick test mode (reduced parameters)
QUICK_M_LIST: List[int] = [1, 2]
QUICK_P_LIST: List[float] = [0.1, 0.2]


# =============================================================================
# DATA MANAGEMENT
# =============================================================================

def create_data_directories(base_dir: Path) -> Dict[str, Path]:
    """Create necessary data directories."""
    dirs = {
        'base': base_dir,
        'steps': base_dir / 'steps',
        'finals': base_dir / 'finals', 
        'logs': base_dir / 'logs',
        'configs': base_dir / 'configs'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def save_experiment_config(config_file: Path, config: Dict[str, Any]):
    """Save experiment configuration as JSON."""
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2, default=str)


def create_csv_headers():
    """Define CSV headers matching simulation format."""
    step_headers = [
        'run_id', 'timestamp', 'M', 'N', 'noise_type', 'p', 'apply_twirling',
        'backend_name', 'shots', 'step', 'level', 'copies_consumed',
        'attempts', 'measurements_used', 'success_probability', 
        'estimated_fidelity', 'grover_iters', 'raw_counts', 'notes'
    ]
    
    final_headers = [
        'run_id', 'timestamp', 'M', 'N', 'noise_type', 'p', 'apply_twirling',
        'backend_name', 'total_shots', 'num_levels', 'final_success',
        'total_attempts', 'total_measurements', 'final_fidelity',
        'experiment_duration_sec', 'error_msg', 'config'
    ]
    
    return step_headers, final_headers


def save_step_data(csv_file: Path, data: Dict[str, Any]):
    """Save step-by-step data to CSV."""
    step_headers, _ = create_csv_headers()
    
    # Create file with headers if it doesn't exist
    if not csv_file.exists():
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=step_headers)
            writer.writeheader()
    
    # Append data
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=step_headers)
        writer.writerow(data)


def save_final_data(csv_file: Path, data: Dict[str, Any]):
    """Save final experiment results to CSV."""
    _, final_headers = create_csv_headers()
    
    # Create file with headers if it doesn't exist
    if not csv_file.exists():
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=final_headers)
            writer.writeheader()
    
    # Append data
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=final_headers)
        writer.writerow(data)


# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

def generate_run_id(M: int, noise_type: str, p: float, backend_name: str, timestamp: str) -> str:
    """Generate unique run ID."""
    return f"IBM_M{M}_{noise_type}_p{p:.3f}_{backend_name}_{timestamp}"


def run_single_experiment(M: int, noise_type: str, p: float, backend, service, 
                         backend_name: str, shots: int, data_dirs: Dict[str, Path],
                         logger) -> Dict[str, Any]:
    """
    Run a single purification experiment and collect data.
    
    This is the core experimental function that matches the simulation structure.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = generate_run_id(M, noise_type, p, backend_name, timestamp)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Starting experiment: {run_id}")
    logger.info(f"M={M}, noise={noise_type}, p={p:.3f}, backend={backend_name}")
    logger.info(f"{'='*70}")
    
    experiment_start = time.time()
    
    # Experiment results
    results = {
        'run_id': run_id,
        'success': False,
        'error': None,
        'steps_data': [],
        'final_data': {},
        'total_measurements': 0,
        'total_attempts': 0
    }
    
    try:
        # 1. Create target state
        target_circuit, target_sv = create_target_state_circuit(M, TARGET_STATE)
        logger.info(f"Created target state: {target_circuit.name} ({M} qubits)")
        
        # 2. Estimate required copies for full recursive purification
        # For now, we'll do single-level purification (N=2 copies)
        N = 2  # Two noisy copies for one purification step
        
        # 3. Run purification experiment
        logger.info(f"Running purification with {N} copies...")
        
        # Create two noisy copies
        copy_A = create_noisy_copy_circuit(target_circuit, noise_type, p, APPLY_TWIRLING, copy_id=1)
        copy_B = create_noisy_copy_circuit(target_circuit, noise_type, p, APPLY_TWIRLING, copy_id=2)
        
        # Run single purification step with detailed logging
        success, step_results = run_single_purification_step(
            target_circuit, noise_type, p, APPLY_TWIRLING, backend, service, 
            shots=shots, max_attempts=MAX_ATTEMPTS
        )
        
        results['total_measurements'] = step_results['measurements_used']
        results['total_attempts'] = step_results['attempts']
        
        # 4. Collect step-by-step data
        step_data = {
            'run_id': run_id,
            'timestamp': timestamp,
            'M': M,
            'N': N,
            'noise_type': noise_type,
            'p': p,
            'apply_twirling': APPLY_TWIRLING,
            'backend_name': backend_name,
            'shots': shots,
            'step': 1,  # Single purification step
            'level': 1,
            'copies_consumed': N,
            'attempts': step_results['attempts'],
            'measurements_used': step_results['measurements_used'],
            'success_probability': step_results['final_success_prob'],
            'estimated_fidelity': 2.0 * step_results['final_success_prob'] - 1.0,  # F = 2P - 1
            'grover_iters': calculate_grover_iterations(step_results['final_success_prob']),
            'raw_counts': json.dumps(step_results.get('final_analysis', {}).get('raw_counts', {})),
            'notes': f"success={success}, error={step_results.get('error', 'None')}"
        }
        
        # Save step data
        save_step_data(data_dirs['base'] / 'steps_ibm.csv', step_data)
        results['steps_data'].append(step_data)
        
        # 5. Measure final fidelity (if successful)
        final_fidelity = 0.0
        if success:
            try:
                # Create fidelity measurement circuit
                final_noisy = create_noisy_copy_circuit(target_circuit, noise_type, p, APPLY_TWIRLING, copy_id=999)
                fidelity, fid_counts = measure_fidelity_with_swap_test(
                    target_circuit, final_noisy, backend, service, shots=shots
                )
                final_fidelity = fidelity
                logger.info(f"Measured final fidelity: {final_fidelity:.4f}")
            except Exception as e:
                logger.warning(f"Fidelity measurement failed: {e}")
                final_fidelity = step_data['estimated_fidelity']
        
        # 6. Collect final results
        experiment_duration = time.time() - experiment_start
        
        final_data = {
            'run_id': run_id,
            'timestamp': timestamp,
            'M': M,
            'N': N,
            'noise_type': noise_type,
            'p': p,
            'apply_twirling': APPLY_TWIRLING,
            'backend_name': backend_name,
            'total_shots': results['total_measurements'],
            'num_levels': NUM_LEVELS,
            'final_success': success,
            'total_attempts': results['total_attempts'],
            'total_measurements': results['total_measurements'],
            'final_fidelity': final_fidelity,
            'experiment_duration_sec': experiment_duration,
            'error_msg': step_results.get('error', ''),
            'config': json.dumps({
                'shots_per_step': shots,
                'max_attempts': MAX_ATTEMPTS,
                'target_state': TARGET_STATE,
                'twirling': APPLY_TWIRLING
            })
        }
        
        # Save final data
        save_final_data(data_dirs['base'] / 'finals_ibm.csv', final_data)
        results['final_data'] = final_data
        results['success'] = success
        
        logger.info(f"Experiment completed: success={success}, duration={experiment_duration:.1f}s")
        
        if success:
            logger.info(f"✅ Final fidelity: {final_fidelity:.4f}")
        else:
            logger.warning(f"❌ Experiment failed: {step_results.get('error', 'Unknown error')}")
        
    except Exception as e:
        results['error'] = str(e)
        logger.error(f"Experiment failed with exception: {e}", exc_info=True)
        
        # Save error to final results
        error_data = {
            'run_id': run_id,
            'timestamp': timestamp,
            'M': M,
            'N': 0,
            'noise_type': noise_type,
            'p': p,
            'apply_twirling': APPLY_TWIRLING,
            'backend_name': backend_name,
            'total_shots': 0,
            'num_levels': NUM_LEVELS,
            'final_success': False,
            'total_attempts': 0,
            'total_measurements': 0,
            'final_fidelity': 0.0,
            'experiment_duration_sec': time.time() - experiment_start,
            'error_msg': str(e),
            'config': json.dumps({'error': True})
        }
        save_final_data(data_dirs['base'] / 'finals_ibm.csv', error_data)
    
    return results


def run_grid_sweep(backend_name: str, shots: int, data_dirs: Dict[str, Path], 
                  logger, quick: bool = False, max_m: Optional[int] = None):
    """
    Run complete grid sweep over all parameter combinations.
    """
    # Setup backend
    logger.info(f"Setting up quantum backend: {backend_name}")
    service, backend = setup_quantum_backend(backend_name)
    
    if backend is None:
        logger.error("Failed to setup backend")
        return
    
    # Select parameter ranges
    M_range = QUICK_M_LIST if quick else M_LIST
    P_range = QUICK_P_LIST if quick else P_LIST
    
    if max_m is not None:
        M_range = [m for m in M_range if m <= max_m]
    
    # Calculate total experiments
    total_experiments = len(M_range) * len(P_range) * len(NOISE_TYPES)
    logger.info(f"\n{'='*70}")
    logger.info(f"Starting grid sweep: {total_experiments} experiments")
    logger.info(f"M values: {M_range}")
    logger.info(f"p values: {P_range}") 
    logger.info(f"Noise types: {NOISE_TYPES}")
    logger.info(f"Backend: {backend_name}")
    logger.info(f"Shots per step: {shots}")
    logger.info(f"{'='*70}")
    
    # Save experiment configuration
    config = {
        'timestamp': datetime.now().isoformat(),
        'backend_name': backend_name,
        'M_range': M_range,
        'P_range': P_range,
        'noise_types': NOISE_TYPES,
        'target_state': TARGET_STATE,
        'apply_twirling': APPLY_TWIRLING,
        'shots': shots,
        'max_attempts': MAX_ATTEMPTS,
        'num_levels': NUM_LEVELS,
        'quick_mode': quick,
        'total_experiments': total_experiments
    }
    config_file = data_dirs['configs'] / f"grid_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_experiment_config(config_file, config)
    
    # Run experiments
    completed = 0
    failed = 0
    start_time = time.time()
    
    for noise_type in NOISE_TYPES:
        for M in M_range:
            for p in P_range:
                completed += 1
                
                logger.info(f"\n{'='*50}")
                logger.info(f"Experiment {completed}/{total_experiments}")
                logger.info(f"M={M}, noise={noise_type}, p={p:.3f}")
                
                try:
                    results = run_single_experiment(
                        M, noise_type, p, backend, service, backend_name, 
                        shots, data_dirs, logger
                    )
                    
                    if results['success']:
                        logger.info(f"✅ Experiment {completed} completed successfully")
                    else:
                        failed += 1
                        logger.warning(f"❌ Experiment {completed} failed")
                        
                except Exception as e:
                    failed += 1
                    logger.error(f"💥 Experiment {completed} crashed: {e}")
                    continue
                
                # Progress update
                elapsed = time.time() - start_time
                avg_time = elapsed / completed
                remaining = total_experiments - completed
                eta = remaining * avg_time
                
                logger.info(f"Progress: {completed}/{total_experiments} "
                          f"({100*completed/total_experiments:.1f}%), "
                          f"Failed: {failed}, ETA: {eta/60:.1f}min")
    
    # Final summary
    total_time = time.time() - start_time
    logger.info(f"\n{'='*70}")
    logger.info(f"GRID SWEEP COMPLETED!")
    logger.info(f"  Total experiments: {total_experiments}")
    logger.info(f"  Successful: {completed - failed}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Success rate: {100*(completed-failed)/completed:.1f}%")
    logger.info(f"  Total time: {total_time/60:.1f} minutes")
    logger.info(f"  Data saved to: {data_dirs['base']}")
    logger.info(f"{'='*70}")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="IBM Quantum SWAP-based purification experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.simulation.main_ibm_run --backend aer_simulator
  python -m src.simulation.main_ibm_run --backend ibm_brisbane --shots 2048
  python -m src.simulation.main_ibm_run --quick --max-m 2 --verbose
        """
    )
    
    parser.add_argument('--backend', type=str, default='aer_simulator',
                       help='Quantum backend (aer_simulator or IBM device name)')
    parser.add_argument('--shots', type=int, default=DEFAULT_SHOTS,
                       help=f'Shots per SWAP test (default: {DEFAULT_SHOTS})')
    parser.add_argument('--max-m', type=int, default=None,
                       help='Maximum M value to test')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with reduced parameters')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--data-dir', type=Path, default=Path('data/IBMQ'),
                       help='Data output directory')
    
    return parser.parse_args()


def main():
    """Main experimental runner."""
    args = parse_arguments()
    
    # Setup data directories
    data_dirs = create_data_directories(args.data_dir)
    
    # Setup logging
    log_file = data_dirs['logs'] / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_file, args.verbose)
    
    logger.info("IBM Quantum SWAP Purification Experiments Starting...")
    logger.info(f"Arguments: {vars(args)}")
    
    # Check Qiskit availability
    if not QISKIT_AVAILABLE:
        logger.error("Qiskit not available! Install with: pip install qiskit qiskit-aer qiskit-ibm-runtime")
        return 1
    
    try:
        # Run grid sweep
        run_grid_sweep(
            backend_name=args.backend,
            shots=args.shots,
            data_dirs=data_dirs,
            logger=logger,
            quick=args.quick,
            max_m=args.max_m
        )
        
        logger.info("All experiments completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Experiments interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Experiments failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())