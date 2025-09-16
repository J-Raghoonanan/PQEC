import numpy as np
from qiskit.quantum_info import Statevector, DensityMatrix, random_statevector, state_fidelity
from typing import Optional, Tuple, List, Dict, Union
import pickle
import json
import time
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ExperimentResult:
    """Store results from a single experiment."""
    M: int
    N: int
    delta: float
    noise_type: str
    trial_number: int
    metrics_by_level: Dict[int, Dict[str, float]]  # level -> {'fidelity': float, 'logical_error': float}
    final_logical_error: float
    final_fidelity: float

class QuantumState:
    """Manages M-qubit quantum states for streaming purification protocol."""
    
    def __init__(self, density_matrix: DensityMatrix, target_state: Statevector, purification_level: int = 0):
        self.density_matrix = density_matrix
        self.target_state = target_state
        self.purification_level = purification_level
        # Fix: Use shape instead of dims() to get correct M value
        total_dim = density_matrix.data.shape[0]
        self.M = int(np.log2(total_dim))
    
    @classmethod
    def from_statevector(cls, statevector: Statevector, purification_level: int = 0):
        """Create QuantumState from pure state vector."""
        density_matrix = DensityMatrix(statevector)
        return cls(density_matrix, statevector, purification_level)
    
    def copy(self):
        """Create deep copy of quantum state."""
        return QuantumState(
            self.density_matrix.copy(),
            self.target_state.copy(),
            self.purification_level
        )

def generate_random_state(M: int, seed: Optional[int] = None) -> QuantumState:
    """Generate random M-qubit pure state using Haar measure."""
    if seed is not None:
        np.random.seed(seed)
    
    statevector = random_statevector(2**M, seed=seed)
    return QuantumState.from_statevector(statevector)

def manual_state(amplitudes: np.ndarray) -> QuantumState:
    """Create quantum state from specified amplitudes."""
    amplitudes = np.array(amplitudes, dtype=complex)
    amplitudes = amplitudes / np.linalg.norm(amplitudes)
    statevector = Statevector(amplitudes)
    return QuantumState.from_statevector(statevector)

def apply_single_qubit_depolarizing(rho: np.ndarray, target_qubit: int, M: int, p: float) -> np.ndarray:
    """Apply depolarizing channel to single qubit using Kraus operators from manuscript."""
    # Kraus operators: E₀ = √(1-p)I, E₁,₂,₃ = √(p/3)σₓ,ᵧ,ᵤ
    I_single = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def build_full_operator(single_qubit_op):
        full_op = np.eye(1, dtype=complex)
        for i in range(M):
            if i == target_qubit:
                full_op = np.kron(full_op, single_qubit_op)
            else:
                full_op = np.kron(full_op, I_single)
        return full_op
    
    I_full = build_full_operator(I_single)
    X_full = build_full_operator(X)
    Y_full = build_full_operator(Y)
    Z_full = build_full_operator(Z)
    
    # Apply depolarizing channel: ρ → (1-p)ρ + (p/3)(XρX† + YρY† + ZρZ†)
    rho_out = (1 - p) * rho
    rho_out += (p/3) * (X_full @ rho @ X_full.conj().T)
    rho_out += (p/3) * (Y_full @ rho @ Y_full.conj().T)
    rho_out += (p/3) * (Z_full @ rho @ Z_full.conj().T)
    
    return rho_out

def apply_depolarizing_noise(state: QuantumState, p: float, mode: str = 'probabilistic', 
                           num_errors: Optional[int] = None) -> QuantumState:
    """Apply depolarizing noise. Manuscript relation: δ_loc = 4p/3."""
    M = state.M
    rho = state.density_matrix.data.copy()
    
    if mode == 'probabilistic':
        for qubit in range(M):
            if np.random.random() < p:
                rho = apply_single_qubit_depolarizing(rho, qubit, M, p)
    
    elif mode == 'fixed':
        if num_errors is None:
            raise ValueError("num_errors must be specified when mode='fixed'")
        error_qubits = np.random.choice(M, size=min(num_errors, M), replace=False)
        for qubit in error_qubits:
            rho = apply_single_qubit_depolarizing(rho, qubit, M, p)
    
    noisy_state = state.copy()
    noisy_state.density_matrix = DensityMatrix(rho)
    return noisy_state

def apply_pauli_error(rho: np.ndarray, target_qubit: int, M: int, pauli: str) -> np.ndarray:
    """Apply single Pauli error to target qubit."""
    if pauli == 'X':
        pauli_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
    elif pauli == 'Z':
        pauli_matrix = np.array([[1, 0], [0, -1]], dtype=complex)
    else:
        raise ValueError("pauli must be 'X' or 'Z'")
    
    # Build M-qubit Pauli operator
    I_single = np.eye(2, dtype=complex)
    operators = []
    for i in range(M):
        if i == target_qubit:
            operators.append(pauli_matrix)
        else:
            operators.append(I_single)
    
    # Compute tensor product
    pauli_op = operators[0]
    for i in range(1, M):
        pauli_op = np.kron(pauli_op, operators[i])
    
    return pauli_op @ rho @ pauli_op.conj().T

def apply_dephasing_noise(state: QuantumState, p_z: float, p_x: float = 0, 
                         mode: str = 'probabilistic', num_errors: Optional[int] = None) -> QuantumState:
    """Apply Pauli dephasing noise (Z and/or X errors)."""
    M = state.M
    rho = state.density_matrix.data.copy()
    
    if mode == 'probabilistic':
        for qubit in range(M):
            rand = np.random.random()
            if rand < p_z:
                rho = apply_pauli_error(rho, qubit, M, 'Z')
            elif rand < p_z + p_x:
                rho = apply_pauli_error(rho, qubit, M, 'X')
    
    elif mode == 'fixed':
        if num_errors is None:
            raise ValueError("num_errors must be specified when mode='fixed'")
        error_qubits = np.random.choice(M, size=min(num_errors, M), replace=False)
        for qubit in error_qubits:
            total_p = p_z + p_x
            if total_p > 0:
                if np.random.random() < p_z / total_p:
                    rho = apply_pauli_error(rho, qubit, M, 'Z')
                else:
                    rho = apply_pauli_error(rho, qubit, M, 'X')
    
    noisy_state = state.copy()
    noisy_state.density_matrix = DensityMatrix(rho)
    return noisy_state

def execute_swap_test(state1: QuantumState, state2: QuantumState) -> Tuple[QuantumState, float]:
    """Execute SWAP test using theoretical calculation from manuscript Eq. 22."""
    rho = state1.density_matrix.data
    
    # Success probability: P_success = 1/2(1 + Tr(ρ²)) [manuscript Eq. 21]
    rho_squared = rho @ rho
    trace_rho_squared = np.trace(rho_squared).real
    success_prob = 0.5 * (1 + trace_rho_squared)
    
    # Purified density matrix: ρ_out = (ρ + ρ²) / (1 + Tr(ρ²)) [manuscript Eq. 22]
    rho_out_data = (rho + rho_squared) / (1 + trace_rho_squared)
    
    purified_state = state1.copy()
    purified_state.density_matrix = DensityMatrix(rho_out_data)
    purified_state.purification_level = max(state1.purification_level, state2.purification_level) + 1
    
    return purified_state, success_prob

def calculate_logical_error(state: QuantumState) -> float:
    """Calculate logical error: ε_L = 1/2 ||ρ - |ψ⟩⟨ψ||₁ [manuscript Eq. 40]."""
    rho = state.density_matrix
    target_dm = DensityMatrix(state.target_state)
    
    diff_matrix = rho.data - target_dm.data
    eigenvalues = np.linalg.eigvals(diff_matrix)
    trace_norm = np.sum(np.abs(eigenvalues))
    
    return 0.5 * trace_norm.real

def calculate_fidelity(state: QuantumState) -> float:
    """Calculate state fidelity: F = ⟨ψ|ρ|ψ⟩ [manuscript]."""
    return state_fidelity(state.density_matrix, state.target_state)

def calculate_amplitude_amplification_iterations(success_prob: float) -> int:
    """Calculate iterations: N_iter = ⌊π/(4 arcsin√P_success) - 1/2⌋ [manuscript Eq. 38]."""
    if success_prob <= 0 or success_prob >= 1:
        return 0
    iterations = np.pi / (4 * np.arcsin(np.sqrt(success_prob))) - 0.5
    return max(0, int(np.floor(iterations)))

class StreamingPurifier:
    """Streaming purification with O(log N) memory scaling [manuscript Section III]."""
    
    def __init__(self, max_levels: int):
        self.stack = [None] * max_levels  # Each element is QuantumState or None
        self.max_levels = max_levels
    
    def process_new_state(self, noisy_state: QuantumState) -> int:
        """Process new state through streaming stack. Returns final purification level."""
        level = 0
        current_state = noisy_state
        
        # Cascade through occupied levels [manuscript Algorithm 1]
        while level < len(self.stack) and self.stack[level] is not None:
            partner_state = self.stack[level]
            self.stack[level] = None
            
            # Apply SWAP test purification
            current_state, _ = execute_swap_test(current_state, partner_state)
            level += 1
        
        # Store at empty level
        if level < len(self.stack):
            self.stack[level] = current_state
        
        return level
    
    def get_state_at_level(self, level: int) -> Optional[QuantumState]:
        """Get state at specified purification level."""
        if 0 <= level < len(self.stack):
            return self.stack[level]
        return None
    
    def get_all_states(self) -> Dict[int, QuantumState]:
        """Get all stored states by purification level."""
        states = {}
        for level, state in enumerate(self.stack):
            if state is not None:
                states[level] = state
        return states
    
    def clear(self):
        """Clear the purification stack."""
        self.stack = [None] * self.max_levels

class StreamingQECSimulation:
    """Main simulation class for streaming QEC protocol."""
    
    def __init__(self):
        self.results = []
    
    def run_single_experiment(self, M: int, N: int, delta: float, noise_type: str = 'depolarizing',
                            trial_number: int = 0, use_random_state: bool = True, 
                            manual_amplitudes: Optional[np.ndarray] = None) -> ExperimentResult:
        """Run single experiment with specified parameters."""
        
        # print(f"DEBUG: Starting experiment M={M}, N={N}, delta={delta}, trial={trial_number}")
        
        # Generate or create target state
        if use_random_state:
            target_state = generate_random_state(M, seed=trial_number * 1000 + M * 100 + N)
        else:
            if manual_amplitudes is None:
                # Default to |+⟩^⊗M state
                single_plus = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
                amplitudes = single_plus
                for _ in range(M-1):
                    amplitudes = np.kron(amplitudes, single_plus)
            else:
                amplitudes = manual_amplitudes
            target_state = manual_state(amplitudes)
        
        # print(f"DEBUG: Target state created, M={target_state.M}, shape={target_state.density_matrix.data.shape}")
        
        # Initialize streaming purifier with O(log N) levels
        max_levels = int(np.log2(N)) + 1
        purifier = StreamingPurifier(max_levels)
        
        # Convert delta to p using manuscript relation: δ_loc = 4p/3
        p = 3 * delta / 4
        # print(f"DEBUG: Converted delta={delta} to p={p}")
        
        # Process N noisy copies
        for i in range(N):
            # print(f"DEBUG: Processing copy {i+1}/{N}")
            
            # Create noisy copy
            if noise_type == 'depolarizing':
                noisy_copy = apply_depolarizing_noise(target_state, p=p, mode='probabilistic')
            elif noise_type == 'dephasing':
                noisy_copy = apply_dephasing_noise(target_state, p_z=p, mode='probabilistic')
            else:
                raise ValueError(f"Unknown noise type: {noise_type}")
            
            # print(f"DEBUG: Created noisy copy {i+1}, shape={noisy_copy.density_matrix.data.shape}")
            
            # Process through streaming purifier
            level = purifier.process_new_state(noisy_copy)
            # print(f"DEBUG: Processed copy {i+1}, final level={level}")
        
        # Calculate metrics at each purification level
        metrics_by_level = {}
        all_states = purifier.get_all_states()
        
        for level, state in all_states.items():
            fidelity = calculate_fidelity(state)
            logical_error = calculate_logical_error(state)
            metrics_by_level[level] = {
                'fidelity': fidelity,
                'logical_error': logical_error
            }
        
        # Get final metrics (highest purification level)
        if metrics_by_level:
            final_level = max(metrics_by_level.keys())
            final_fidelity = metrics_by_level[final_level]['fidelity']
            final_logical_error = metrics_by_level[final_level]['logical_error']
        else:
            final_fidelity = 0.0
            final_logical_error = 1.0
        
        # print(f"DEBUG: Experiment completed successfully")
        
        return ExperimentResult(
            M=M,
            N=N,
            delta=delta,
            noise_type=noise_type,
            trial_number=trial_number,
            metrics_by_level=metrics_by_level,
            final_logical_error=final_logical_error,
            final_fidelity=final_fidelity
        )
    
    def run_parameter_sweep(self, M_values: List[int], N_values: List[int], 
                          delta_values: List[float], noise_types: List[str] = ['depolarizing'],
                          num_trials: int = 100, use_random_states: bool = True) -> List[ExperimentResult]:
        """Run complete parameter sweep as specified in manuscript."""
        
        all_results = []
        total_experiments = len(M_values) * len(N_values) * len(delta_values) * len(noise_types) * num_trials
        experiment_count = 0
        
        print(f"Starting parameter sweep: {total_experiments} total experiments")
        print(f"M: {M_values}")
        print(f"N: {N_values}")
        print(f"delta: {delta_values}")
        print(f"noise_types: {noise_types}")
        print(f"trials per condition: {num_trials}")
        
        start_time = time.time()
        
        for M in M_values:
            for N in N_values:
                for delta in delta_values:
                    for noise_type in noise_types:
                        
                        print(f"Running M={M}, N={N}, delta={delta}, noise={noise_type}")
                        condition_start = time.time()
                        
                        for trial in range(num_trials):
                            result = self.run_single_experiment(
                                M=M, N=N, delta=delta, noise_type=noise_type,
                                trial_number=trial, use_random_state=use_random_states
                            )
                            all_results.append(result)
                            experiment_count += 1
                            
                            if experiment_count % 100 == 0:
                                elapsed = time.time() - start_time
                                rate = experiment_count / elapsed
                                remaining = (total_experiments - experiment_count) / rate
                                print(f"  Progress: {experiment_count}/{total_experiments} "
                                      f"({100*experiment_count/total_experiments:.1f}%) "
                                      f"Rate: {rate:.1f} exp/sec, ETA: {remaining/60:.1f} min")
                        
                        condition_time = time.time() - condition_start
                        print(f"  Completed in {condition_time:.1f}s")
        
        total_time = time.time() - start_time
        print(f"Parameter sweep completed in {total_time/60:.1f} minutes")
        
        self.results = all_results
        return all_results
    
    def aggregate_results(self, results: List[ExperimentResult]) -> Dict:
        """Aggregate results for analysis and plotting."""
        aggregated = {}
        
        # Group by (M, N, delta, noise_type)
        grouped = {}
        for result in results:
            key = (result.M, result.N, result.delta, result.noise_type)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)
        
        # Calculate averages for each condition
        for key, condition_results in grouped.items():
            M, N, delta, noise_type = key
            
            # Aggregate metrics by purification level
            level_metrics = {}
            
            # Find all purification levels present
            all_levels = set()
            for result in condition_results:
                all_levels.update(result.metrics_by_level.keys())
            
            for level in all_levels:
                fidelities = []
                logical_errors = []
                
                for result in condition_results:
                    if level in result.metrics_by_level:
                        fidelities.append(result.metrics_by_level[level]['fidelity'])
                        logical_errors.append(result.metrics_by_level[level]['logical_error'])
                
                if fidelities:  # Only include if we have data
                    level_metrics[level] = {
                        'fidelity_mean': np.mean(fidelities),
                        'fidelity_std': np.std(fidelities),
                        'logical_error_mean': np.mean(logical_errors),
                        'logical_error_std': np.std(logical_errors),
                        'count': len(fidelities)
                    }
            
            # Final metrics
            final_fidelities = [r.final_fidelity for r in condition_results]
            final_logical_errors = [r.final_logical_error for r in condition_results]
            
            aggregated[key] = {
                'M': M,
                'N': N,
                'delta': delta,
                'noise_type': noise_type,
                'num_trials': len(condition_results),
                'level_metrics': level_metrics,
                'final_fidelity_mean': np.mean(final_fidelities),
                'final_fidelity_std': np.std(final_fidelities),
                'final_logical_error_mean': np.mean(final_logical_errors),
                'final_logical_error_std': np.std(final_logical_errors)
            }
        
        return aggregated
    
    def save_results(self, filename_prefix: str = "data/simulations/streaming_qec_results"):
        """Save results in multiple formats."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save raw results as pickle
        pickle_filename = f"{filename_prefix}_raw_{timestamp}.pkl"
        with open(pickle_filename, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Raw results saved to {pickle_filename}")
        
        # Save aggregated results as JSON
        aggregated = self.aggregate_results(self.results)
        
        # Convert to JSON-serializable format
        json_data = {}
        for key, data in aggregated.items():
            json_key = f"M{key[0]}_N{key[1]}_delta{key[2]:.3f}_{key[3]}"
            json_data[json_key] = data
        
        json_filename = f"{filename_prefix}_aggregated_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Aggregated results saved to {json_filename}")
        
        return pickle_filename, json_filename

def test_noise_application():
    """Test noise application in isolation to debug dimension issues."""
    print("=== Testing Noise Application ===")
    
    # Test 1: Single qubit
    print("Testing M=1...")
    state1 = generate_random_state(1, seed=42)
    print(f"M=1 density matrix shape: {state1.density_matrix.data.shape}")
    
    try:
        noisy1 = apply_depolarizing_noise(state1, p=0.1, mode='probabilistic')
        print("M=1 noise application: SUCCESS")
    except Exception as e:
        print(f"M=1 noise application: FAILED - {e}")
    
    # Test 2: Two qubits  
    print("Testing M=2...")
    state2 = generate_random_state(2, seed=42)
    print(f"M=2 density matrix shape: {state2.density_matrix.data.shape}")
    
    try:
        noisy2 = apply_depolarizing_noise(state2, p=0.1, mode='probabilistic')
        print("M=2 noise application: SUCCESS")
    except Exception as e:
        print(f"M=2 noise application: FAILED - {e}")
    
    return True

def main():
    """Main function to run the complete simulation."""
    
    # First run debugging test
    # test_noise_application()
    
    # If that works, proceed with reduced parameter sweep for testing
    M_values = [1, 2, 3, 4, 5, 6]
    N_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    delta_values = [0.01, 0.1, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    
    # Initialize simulation
    simulation = StreamingQECSimulation()
    
    # Run parameter sweep
    results = simulation.run_parameter_sweep(
        M_values=M_values,
        N_values=N_values,
        delta_values=delta_values,
        noise_types=['depolarizing'],
        num_trials=5,  # Very small for debugging
        use_random_states=True
    )
    
    # Save results
    simulation.save_results("streaming_qec_debug_data")
    
    print(f"Simulation completed. Total results: {len(results)}")
    
    return simulation, results, {}

if __name__ == "__main__":
    simulation, results, aggregated = main()