#!/usr/bin/env python3
"""
Fallback version using Qiskit Aer simulator when hardware transpilation issues occur.
This allows testing the SWAP purification logic while hardware issues are resolved.
"""

import logging
import csv
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from IBMQ_components import (
    PurificationConfig,
    create_batch_purification_circuit,
    add_measurements,
    analyze_results,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Parameter ranges
TARGET_M_VALUES = [1, 2]
TARGET_P_VALUES = [0.01, 0.1, 0.2, 0.3]
TARGET_N_VALUES = [2, 4, 8]

def run_experiment_with_simulator(config: PurificationConfig) -> Dict[str, Any]:
    """
    Run SWAP purification experiment using Qiskit Aer simulator.
    This avoids transpilation issues with real hardware.
    """
    try:
        from qiskit_aer import AerSimulator
        from qiskit import transpile
        
        logger.info(f"Running {config.synthesize_run_id()} with Aer simulator")
        
        # Create circuit
        purification_circuit = create_batch_purification_circuit(config)
        measured_circuit = add_measurements(purification_circuit, config)
        
        logger.info(f"Circuit: {measured_circuit.num_qubits} qubits, depth {measured_circuit.depth()}")
        
        # Setup Aer simulator
        simulator = AerSimulator()
        
        # Transpile for simulator (should be much cleaner)
        transpiled = transpile(measured_circuit, simulator, optimization_level=1)
        logger.info(f"Transpiled for simulator: {transpiled.num_qubits} qubits")
        
        # Run simulation
        job = simulator.run(transpiled, shots=config.shots)
        result = job.result()
        counts = result.get_counts(0)  # Aer uses different result structure
        
        # Analyze results
        fidelity, success_prob = analyze_results(counts, config)
        
        return {
            'run_id': config.synthesize_run_id(),
            'M': config.M,
            'N': config.N, 
            'p': config.p,
            'purification_rounds': int(np.log2(config.N)),
            'estimated_qubits': measured_circuit.num_qubits,
            'final_fidelity': fidelity,
            'swap_success_probability': success_prob,
            'total_shots': config.shots,
            'backend_name': 'aer_simulator',
            'circuit_depth': transpiled.depth(),
            'circuit_qubits': transpiled.num_qubits,
            'error_message': ''
        }
        
    except ImportError:
        logger.error("Qiskit Aer not available. Install with: pip install qiskit-aer")
        return {
            'run_id': config.synthesize_run_id(),
            'error_message': 'Qiskit Aer not available'
        }
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return {
            'run_id': config.synthesize_run_id(),
            'error_message': str(e)
        }

def run_simulator_grid_sweep():
    """Run the complete parameter sweep using simulators."""
    
    print("🔬 SWAP Purification Simulator Test")
    print("="*60)
    print("This tests your implementation using Qiskit Aer simulator")
    print("to validate the logic while hardware issues are resolved.")
    print("="*60)
    
    # Generate all configurations
    configs = []
    for M in TARGET_M_VALUES:
        for N in TARGET_N_VALUES:
            for p in TARGET_P_VALUES:
                config = PurificationConfig(
                    M=M, N=N, p=p,
                    backend_name="aer_simulator",
                    shots=8192,  # High stats for clean results
                    max_retry_attempts=1,
                    min_success_rate=0.0
                )
                configs.append(config)
    
    print(f"📊 Running {len(configs)} experiments")
    
    results = []
    start_time = time.time()
    
    for i, config in enumerate(configs):
        qubits = config.N * config.M + int(np.log2(config.N))
        rounds = int(np.log2(config.N))
        
        print(f"\n🧪 [{i+1:2d}/{len(configs)}] M={config.M}, N={config.N} ({rounds} rounds), p={config.p}")
        print(f"    Expected qubits: {qubits}")
        
        result = run_experiment_with_simulator(config)
        results.append(result)
        
        if 'error_message' in result and result['error_message']:
            print(f"    ❌ Error: {result['error_message']}")
        else:
            fid = result.get('final_fidelity', -1)
            success = result.get('swap_success_probability', -1)
            print(f"    ✅ Fidelity: {fid:.4f}, Success: {success:.4f}")
    
    # Save results
    output_file = Path("data/SWAP_simulator_results.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        'run_id', 'M', 'N', 'p', 'purification_rounds', 'estimated_qubits',
        'final_fidelity', 'swap_success_probability', 'total_shots',
        'backend_name', 'circuit_depth', 'circuit_qubits', 'error_message'
    ]
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            clean_result = {field: result.get(field, '') for field in fieldnames}
            writer.writerow(clean_result)
    
    # Summary
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r.get('final_fidelity', -1) >= 0)
    
    print(f"\n{'='*60}")
    print(f"📋 SIMULATOR TEST COMPLETED")
    print(f"{'='*60}")
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print(f"Runtime: {total_time:.1f} seconds")
    print(f"Results saved to: {output_file}")
    
    # Show some results
    if successful > 0:
        print(f"\n📊 Sample Results:")
        for result in results[:6]:  # Show first 6
            if result.get('final_fidelity', -1) >= 0:
                print(f"  M={result['M']}, N={result['N']}, p={result['p']}: "
                      f"fidelity={result['final_fidelity']:.3f}, "
                      f"success={result['swap_success_probability']:.3f}")
    
    print(f"\n💡 If simulator results look good, the issue is only with")
    print(f"   hardware transpilation, not your implementation logic!")

if __name__ == "__main__":
    run_simulator_grid_sweep()