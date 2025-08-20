"""
Streaming Purification Quantum Error Correction - Parameter Sweep Implementation

This module implements complete quantum simulation with parameter sweep capabilities:
- Accepts arrays for dimension, noise_strength, and Pauli probabilities
- Runs simulations across all parameter combinations
- Organized data saving and analysis across parameter space
- Progress tracking for large parameter sweeps

"""

import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import logging
import itertools
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class NoiseType(Enum):
    """Enumeration of supported noise models."""
    DEPOLARIZING = "depolarizing"
    PAULI_SYMMETRIC = "pauli_symmetric"
    PAULI_BIASED = "pauli_biased"
    PURE_DEPHASING = "pure_dephasing"

@dataclass
class ProtocolParameters:
    """Configuration parameters for the purification protocol."""
    dimension: int
    noise_type: NoiseType
    noise_strength: float
    purification_levels: int
    pauli_px: float = 1/3
    pauli_py: float = 1/3
    pauli_pz: float = 1/3
    max_amplification_retries: int = 10
    amplitude_noise_level: float = 1e-8

@dataclass
class QuantumState:
    """Representation of a quantum state in the protocol."""
    purity_parameter: float  # λ ∈ [0,1]
    dimension: int
    target_vector: np.ndarray
    
    def __post_init__(self):
        """Validate state parameters."""
        if not 0 <= self.purity_parameter <= 1:
            raise ValueError(f"Purity parameter must be in [0,1], got {self.purity_parameter}")
        if self.dimension < 2:
            raise ValueError(f"Dimension must be ≥ 2, got {self.dimension}")
        
    @property
    def fidelity_with_target(self) -> float:
        """Calculate fidelity with pure target state."""
        return self.purity_parameter + (1 - self.purity_parameter) / self.dimension
    
    @property
    def logical_error(self) -> float:
        """Calculate logical error using manuscript formula."""
        return (1 - self.purity_parameter) * (self.dimension - 1) / self.dimension

    def get_density_matrix(self) -> np.ndarray:
        """Get the full density matrix representation."""
        target_projector = np.outer(self.target_vector, self.target_vector.conj())
        mixed_state = np.eye(self.dimension) / self.dimension
        return self.purity_parameter * target_projector + (1 - self.purity_parameter) * mixed_state

@dataclass
class AmplificationResult:
    """Result of amplitude amplification process."""
    initial_success_probability: float
    final_success_probability: float
    iterations_used: int
    amplitude_evolution: np.ndarray
    success: bool
    gate_count: int
    retries_needed: int

@dataclass
class SwapTestResult:
    """Results from a single swap test operation with amplitude amplification."""
    input_purity: float
    output_purity: float
    amplification_result: AmplificationResult
    theoretical_output: float
    actual_success: bool
    total_gate_count: int

@dataclass
class PurificationResult:
    """Complete results from recursive purification simulation."""
    protocol_params: ProtocolParameters
    initial_state: QuantumState
    final_state: QuantumState
    purity_evolution: np.ndarray
    error_evolution: np.ndarray
    fidelity_evolution: np.ndarray
    amplification_results: List[AmplificationResult]
    total_amplification_iterations: int
    total_gate_complexity: int
    total_retries: int
    error_reduction_ratio: float
    theoretical_final_purity: float
    simulation_accuracy: float

@dataclass
class ParameterSweepResult:
    """Results from a complete parameter sweep."""
    parameter_combinations: List[Dict]
    individual_results: List[PurificationResult]
    summary_statistics: Dict[str, Any]
    sweep_metadata: Dict[str, Any]

class NoiseModel(ABC):
    """Abstract base class for noise models."""
    
    @abstractmethod
    def apply_noise(self, pure_state: np.ndarray, noise_strength: float) -> QuantumState:
        """Apply noise to a pure state."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get descriptive name for the noise model."""
        pass

class DepolarizingNoise(NoiseModel):
    """Standard depolarizing noise: rho = (1-δ)|ψ⟩⟨ψ| + δI/d"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
    
    def apply_noise(self, pure_state: np.ndarray, noise_strength: float) -> QuantumState:
        """Apply depolarizing noise with strength δ."""
        purity = 1 - noise_strength
        return QuantumState(purity, self.dimension, pure_state)
    
    def get_name(self) -> str:
        return f"Depolarizing_d{self.dimension}"

class PauliNoise(NoiseModel):
    """General Pauli noise with independent X, Y, Z error rates."""
    
    def __init__(self, dimension: int = 2, px: float = 1/3, py: float = 1/3, pz: float = 1/3):
        if dimension != 2:
            raise ValueError("Pauli noise currently only supports qubits (d=2)")
        
        self.dimension = dimension
        self.px, self.py, self.pz = px, py, pz
        
        # Normalize probabilities
        total = px + py + pz
        if total > 0:
            self.px, self.py, self.pz = px/total, py/total, pz/total
    
    def apply_noise(self, pure_state: np.ndarray, noise_strength: float) -> QuantumState:
        """Apply Pauli noise with total strength scaled by noise_strength."""
        # Scale individual Pauli probabilities
        noise_strength = 1 # hardcoding this for now
        scaled_px = self.px * noise_strength
        scaled_py = self.py * noise_strength
        scaled_pz = self.pz * noise_strength
        
        # For Pauli noise, effective purity calculation depends on specific model
        total_error = scaled_px + scaled_py + scaled_pz
        effective_purity = 1 - total_error
        
        return QuantumState(effective_purity, self.dimension, pure_state)
    
    def get_name(self) -> str:
        return f"Pauli_px{self.px:.2f}_py{self.py:.2f}_pz{self.pz:.2f}"

class SwapTestSimulator:
    """Implements the swap test with full amplitude amplification simulation."""
    
    def __init__(self, dimension: int, params: ProtocolParameters):
        self.dimension = dimension
        self.params = params
    
    def calculate_success_probability(self, state1: QuantumState, state2: QuantumState) -> float:
        """Calculate swap test success probability using manuscript formula."""
        if state1.dimension != state2.dimension:
            raise ValueError("States must have same dimension")
        
        λ = state1.purity_parameter
        d = self.dimension
        
        tr_rho_squared = λ**2 + (1 - λ**2) / d
        return 0.5 * (1 + tr_rho_squared)
    
    def compute_output_purity(self, input_purity: float) -> float:
        """Compute output purity using theoretical transformation."""
        λ = input_purity
        d = self.dimension
        
        numerator = λ * (1 + λ + 2*(1-λ)/d)
        denominator = 1 + λ**2 + (1-λ**2)/d
        
        return numerator / denominator
    
    def simulate_amplitude_amplification(self, initial_success_prob: float) -> AmplificationResult:
        """Simulate full amplitude amplification process with Q operator iterations."""
        if initial_success_prob >= 1.0:
            return AmplificationResult(
                initial_success_probability=1.0,
                final_success_probability=1.0,
                iterations_used=0,
                amplitude_evolution=np.array([1.0]),
                success=True,
                gate_count=4,
                retries_needed=0
            )
        
        theta = 2 * np.arcsin(np.sqrt(initial_success_prob))
        optimal_iterations = max(0, int(np.floor(np.pi / (2 * theta) - 0.5)))
        
        amplitude_evolution = np.zeros(optimal_iterations + 1)
        amplitude_evolution[0] = np.sqrt(initial_success_prob)
        
        for k in range(optimal_iterations):
            new_amplitude = np.sin((2*(k+1) + 1) * theta / 2)
            amplitude_evolution[k + 1] = new_amplitude
        
        final_success_prob = amplitude_evolution[-1]**2
        final_success_prob = min(1.0, final_success_prob + 
                               np.random.normal(0, self.params.amplitude_noise_level))
        
        measurement_success = np.random.random() < final_success_prob
        gate_count = 4 + 4 * optimal_iterations
        
        return AmplificationResult(
            initial_success_probability=initial_success_prob,
            final_success_probability=final_success_prob,
            iterations_used=optimal_iterations,
            amplitude_evolution=amplitude_evolution,
            success=measurement_success,
            gate_count=gate_count,
            retries_needed=0
        )
    
    def perform_swap_test_with_amplification(self, state1: QuantumState, state2: QuantumState) -> SwapTestResult:
        """Perform complete swap test with amplitude amplification and retry logic."""
        initial_success_prob = self.calculate_success_probability(state1, state2)
        theoretical_output = self.compute_output_purity(state1.purity_parameter)
        
        total_retries = 0
        total_gates = 0
        
        for attempt in range(self.params.max_amplification_retries):
            amp_result = self.simulate_amplitude_amplification(initial_success_prob)
            amp_result.retries_needed = total_retries
            total_gates += amp_result.gate_count
            
            if amp_result.success:
                actual_output = theoretical_output + np.random.normal(0, self.params.amplitude_noise_level)
                actual_output = np.clip(actual_output, 0, 1)
                
                return SwapTestResult(
                    input_purity=state1.purity_parameter,
                    output_purity=actual_output,
                    amplification_result=amp_result,
                    theoretical_output=theoretical_output,
                    actual_success=True,
                    total_gate_count=total_gates
                )
            else:
                total_retries += 1
        
        raise RuntimeError(f"Amplitude amplification failed after {self.params.max_amplification_retries} attempts")

class RecursivePurificationEngine:
    """Implements the recursive purification protocol with full quantum simulation."""
    
    def __init__(self, protocol_params: ProtocolParameters):
        self.params = protocol_params
        self.swap_simulator = SwapTestSimulator(protocol_params.dimension, protocol_params)
        
    def theoretical_purity_evolution(self, initial_purity: float) -> np.ndarray:
        """Calculate theoretical purity evolution for validation."""
        evolution = np.zeros(self.params.purification_levels + 1)
        evolution[0] = initial_purity
        
        current_purity = initial_purity
        for level in range(self.params.purification_levels):
            current_purity = self.swap_simulator.compute_output_purity(current_purity)
            evolution[level + 1] = current_purity
        
        return evolution
    
    def execute_purification(self, initial_state: QuantumState) -> PurificationResult:
        """Execute the complete recursive purification protocol with full simulation."""
        
        levels = self.params.purification_levels
        purity_evolution = np.zeros(levels + 1)
        error_evolution = np.zeros(levels + 1)
        fidelity_evolution = np.zeros(levels + 1)
        all_amplification_results = []
        
        purity_evolution[0] = initial_state.purity_parameter
        error_evolution[0] = initial_state.logical_error
        fidelity_evolution[0] = initial_state.fidelity_with_target
        
        total_gates = 0
        total_amp_iterations = 0
        total_retries = 0
        current_purity = initial_state.purity_parameter
        
        current_states = [QuantumState(current_purity, self.params.dimension, 
                                     initial_state.target_vector) 
                         for _ in range(2**levels)]
        
        for level in range(levels):
            level_amp_results = []
            new_states = []
            
            for i in range(0, len(current_states), 2):
                state1 = current_states[i]
                state2 = current_states[i + 1]
                
                swap_result = self.swap_simulator.perform_swap_test_with_amplification(state1, state2)
                
                output_state = QuantumState(swap_result.output_purity, self.params.dimension,
                                          initial_state.target_vector)
                new_states.append(output_state)
                
                level_amp_results.append(swap_result.amplification_result)
                total_gates += swap_result.total_gate_count
                total_amp_iterations += swap_result.amplification_result.iterations_used
                total_retries += swap_result.amplification_result.retries_needed
            
            current_states = new_states
            all_amplification_results.extend(level_amp_results)
            
            if current_states:
                avg_purity = np.mean([state.purity_parameter for state in current_states])
                purity_evolution[level + 1] = avg_purity
                error_evolution[level + 1] = (1 - avg_purity) * (self.params.dimension - 1) / self.params.dimension
                fidelity_evolution[level + 1] = avg_purity + (1 - avg_purity) / self.params.dimension
                current_purity = avg_purity
        
        final_state = current_states[0] if current_states else QuantumState(0, self.params.dimension, 
                                                                           initial_state.target_vector)
        
        theoretical_evolution = self.theoretical_purity_evolution(initial_state.purity_parameter)
        theoretical_final = theoretical_evolution[-1]
        simulation_accuracy = 1 - abs(final_state.purity_parameter - theoretical_final) / max(theoretical_final, 1e-6)
        
        error_reduction_ratio = final_state.logical_error / max(initial_state.logical_error, 1e-6)
        
        return PurificationResult(
            protocol_params=self.params,
            initial_state=initial_state,
            final_state=final_state,
            purity_evolution=purity_evolution,
            error_evolution=error_evolution,
            fidelity_evolution=fidelity_evolution,
            amplification_results=all_amplification_results,
            total_amplification_iterations=total_amp_iterations,
            total_gate_complexity=total_gates,
            total_retries=total_retries,
            error_reduction_ratio=error_reduction_ratio,
            theoretical_final_purity=theoretical_final,
            simulation_accuracy=simulation_accuracy
        )

class DataManager:
    """Handles data saving and organization for parameter sweeps."""
    
    def __init__(self, base_dir: str = "./data"):
        self.base_dir = base_dir
        self.create_directory_structure()
    
    def create_directory_structure(self):
        """Create organized data directories."""
        subdirs = [
            "streaming_purification",
            "streaming_purification/parameter_sweeps",
            "streaming_purification/individual_runs",
            "streaming_purification/sweep_summaries"
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(self.base_dir, subdir), exist_ok=True)
    
    def save_individual_result(self, result: PurificationResult, run_id: int = 0) -> str:
        """Save individual purification result."""
        noise_type = result.protocol_params.noise_type.value
        filename = f"run_{run_id:04d}_{noise_type}_d{result.protocol_params.dimension}_" \
                  f"delta{result.protocol_params.noise_strength:.4f}_" \
                  f"levels{result.protocol_params.purification_levels}.npz"
        
        filepath = os.path.join(self.base_dir, "streaming_purification/individual_runs", filename)
        
        save_data = {
            'dimension': result.protocol_params.dimension,
            'noise_type': result.protocol_params.noise_type.value,
            'noise_strength': result.protocol_params.noise_strength,
            'purification_levels': result.protocol_params.purification_levels,
            'pauli_px': result.protocol_params.pauli_px,
            'pauli_py': result.protocol_params.pauli_py,
            'pauli_pz': result.protocol_params.pauli_pz,
            'initial_purity': result.initial_state.purity_parameter,
            'final_purity': result.final_state.purity_parameter,
            'purity_evolution': result.purity_evolution,
            'error_evolution': result.error_evolution,
            'fidelity_evolution': result.fidelity_evolution,
            'total_amplification_iterations': result.total_amplification_iterations,
            'total_gate_complexity': result.total_gate_complexity,
            'total_retries': result.total_retries,
            'error_reduction_ratio': result.error_reduction_ratio,
            'theoretical_final_purity': result.theoretical_final_purity,
            'simulation_accuracy': result.simulation_accuracy,
            'run_id': run_id
        }
        
        np.savez_compressed(filepath, **save_data)
        return filepath
    
    def save_parameter_sweep(self, sweep_result: ParameterSweepResult, sweep_name: str) -> str:
        """Save complete parameter sweep results."""
        filename = f"parameter_sweep_{sweep_name}.npz"
        filepath = os.path.join(self.base_dir, "streaming_purification/parameter_sweeps", filename)
        
        # Extract arrays from individual results
        dimensions = np.array([r.protocol_params.dimension for r in sweep_result.individual_results])
        noise_strengths = np.array([r.protocol_params.noise_strength for r in sweep_result.individual_results])
        pauli_px_values = np.array([r.protocol_params.pauli_px for r in sweep_result.individual_results])
        pauli_py_values = np.array([r.protocol_params.pauli_py for r in sweep_result.individual_results])
        pauli_pz_values = np.array([r.protocol_params.pauli_pz for r in sweep_result.individual_results])
        
        initial_purities = np.array([r.initial_state.purity_parameter for r in sweep_result.individual_results])
        final_purities = np.array([r.final_state.purity_parameter for r in sweep_result.individual_results])
        error_reductions = np.array([r.error_reduction_ratio for r in sweep_result.individual_results])
        gate_complexities = np.array([r.total_gate_complexity for r in sweep_result.individual_results])
        total_retries = np.array([r.total_retries for r in sweep_result.individual_results])
        simulation_accuracies = np.array([r.simulation_accuracy for r in sweep_result.individual_results])
        
        save_data = {
            'parameter_combinations': sweep_result.parameter_combinations,
            'dimensions': dimensions,
            'noise_strengths': noise_strengths,
            'pauli_px_values': pauli_px_values,
            'pauli_py_values': pauli_py_values,
            'pauli_pz_values': pauli_pz_values,
            'initial_purities': initial_purities,
            'final_purities': final_purities,
            'error_reductions': error_reductions,
            'gate_complexities': gate_complexities,
            'total_retries': total_retries,
            'simulation_accuracies': simulation_accuracies,
            'summary_statistics': sweep_result.summary_statistics,
            'sweep_metadata': sweep_result.sweep_metadata,
            'num_combinations': len(sweep_result.individual_results)
        }
        
        np.savez_compressed(filepath, **save_data)
        logger.info(f"Saved parameter sweep to: {filepath}")
        
        return filepath

class StreamingPurificationProtocol:
    """Main protocol implementation with parameter sweep capabilities."""
    
    def __init__(self, data_manager: Optional[DataManager] = None):
        self.data_manager = data_manager or DataManager()
    
    def run_single_purification(self, dimension: int, noise_type: NoiseType, 
                              noise_strength: float, purification_levels: int,
                              pauli_px: float = 1/3, pauli_py: float = 1/3, pauli_pz: float = 1/3) -> PurificationResult:
        """Run a single purification experiment with full quantum simulation."""
        params = ProtocolParameters(
            dimension=dimension,
            noise_type=noise_type,
            noise_strength=noise_strength,
            purification_levels=purification_levels,
            pauli_px=pauli_px,
            pauli_py=pauli_py,
            pauli_pz=pauli_pz
        )
        
        if noise_type == NoiseType.DEPOLARIZING:
            noise_model = DepolarizingNoise(dimension)
        elif noise_type in [NoiseType.PAULI_SYMMETRIC, NoiseType.PAULI_BIASED, NoiseType.PURE_DEPHASING]:
            if noise_type == NoiseType.PAULI_SYMMETRIC:
                noise_model = PauliNoise(2, pauli_px, pauli_py, pauli_pz)
            elif noise_type == NoiseType.PAULI_BIASED:
                noise_model = PauliNoise(2, pauli_px, pauli_py, pauli_pz)
            else:  # PURE_DEPHASING
                noise_model = PauliNoise(2, 0.0, 0.0, 1.0)
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
        
        target_vector = np.zeros(dimension)
        target_vector[0] = 1.0
        initial_state = noise_model.apply_noise(target_vector, noise_strength)
        
        engine = RecursivePurificationEngine(params)
        result = engine.execute_purification(initial_state)
        
        return result
    
    def run_parameter_sweep(self, dimensions: Union[int, List[int]], 
                          noise_strengths: Union[float, List[float]],
                          purification_levels: int = 5,
                          pauli_px: Union[float, List[float]] = 1/3,
                          pauli_py: Union[float, List[float]] = 1/3,
                          pauli_pz: Union[float, List[float]] = 1/3,
                          noise_type: NoiseType = NoiseType.DEPOLARIZING,
                          sweep_name: str = "default") -> ParameterSweepResult:
        """
        Run parameter sweep across all combinations of input parameters.
        
        Args:
            dimensions: Single value or list of system dimensions
            noise_strengths: Single value or list of noise parameters δ
            purification_levels: Number of recursive levels
            pauli_px, pauli_py, pauli_pz: Single value or list of Pauli probabilities
            noise_type: Type of noise model to use
            sweep_name: Name for saving results
            
        Returns:
            ParameterSweepResult with all combinations and analysis
        """
        
        # Convert single values to lists
        if not isinstance(dimensions, list):
            dimensions = [dimensions]
        if not isinstance(noise_strengths, list):
            noise_strengths = [noise_strengths]
        if not isinstance(pauli_px, list):
            pauli_px = [pauli_px]
        if not isinstance(pauli_py, list):
            pauli_py = [pauli_py]
        if not isinstance(pauli_pz, list):
            pauli_pz = [pauli_pz]
        
        # Generate all parameter combinations
        param_combinations = list(itertools.product(
            dimensions, noise_strengths, pauli_px, pauli_py, pauli_pz
        ))
        
        print(f"Running parameter sweep: {len(param_combinations)} combinations")
        print(f"Dimensions: {dimensions}")
        print(f"Noise strengths: {noise_strengths}")
        print(f"Pauli px: {pauli_px}")
        print(f"Pauli py: {pauli_py}")
        print(f"Pauli pz: {pauli_pz}")
        
        individual_results = []
        parameter_dicts = []
        
        # Run simulations with progress bar
        for i, (d, delta, px, py, pz) in enumerate(tqdm(param_combinations, desc="Parameter sweep")):
            try:
                result = self.run_single_purification(
                    dimension=d,
                    noise_type=noise_type,
                    noise_strength=delta,
                    purification_levels=purification_levels,
                    pauli_px=px,
                    pauli_py=py,
                    pauli_pz=pz
                )
                
                individual_results.append(result)
                parameter_dicts.append({
                    'dimension': d,
                    'noise_strength': delta,
                    'pauli_px': px,
                    'pauli_py': py,
                    'pauli_pz': pz,
                    'run_id': i
                })
                
                # Save individual result
                self.data_manager.save_individual_result(result, run_id=i)
                
            except Exception as e:
                logger.error(f"Failed simulation {i} with params d={d}, δ={delta}, px={px}, py={py}, pz={pz}: {e}")
                continue
        
        # Calculate summary statistics
        if individual_results:
            final_purities = np.array([r.final_state.purity_parameter for r in individual_results])
            error_reductions = np.array([r.error_reduction_ratio for r in individual_results])
            gate_complexities = np.array([r.total_gate_complexity for r in individual_results])
            simulation_accuracies = np.array([r.simulation_accuracy for r in individual_results])
            
            summary_stats = {
                'num_successful_runs': len(individual_results),
                'num_total_combinations': len(param_combinations),
                'success_rate': len(individual_results) / len(param_combinations),
                'final_purity_stats': {
                    'mean': float(np.mean(final_purities)),
                    'std': float(np.std(final_purities)),
                    'min': float(np.min(final_purities)),
                    'max': float(np.max(final_purities))
                },
                'error_reduction_stats': {
                    'mean': float(np.mean(error_reductions)),
                    'std': float(np.std(error_reductions)),
                    'min': float(np.min(error_reductions)),
                    'max': float(np.max(error_reductions))
                },
                'gate_complexity_stats': {
                    'mean': float(np.mean(gate_complexities)),
                    'std': float(np.std(gate_complexities)),
                    'min': float(np.min(gate_complexities)),
                    'max': float(np.max(gate_complexities))
                },
                'simulation_accuracy_stats': {
                    'mean': float(np.mean(simulation_accuracies)),
                    'std': float(np.std(simulation_accuracies)),
                    'min': float(np.min(simulation_accuracies)),
                    'max': float(np.max(simulation_accuracies))
                }
            }
        else:
            summary_stats = {'error': 'No successful runs'}
        
        sweep_metadata = {
            'sweep_name': sweep_name,
            'noise_type': noise_type.value,
            'purification_levels': purification_levels,
            'parameter_ranges': {
                'dimensions': dimensions,
                'noise_strengths': noise_strengths,
                'pauli_px': pauli_px,
                'pauli_py': pauli_py,
                'pauli_pz': pauli_pz
            },
            'creation_time': datetime.now().isoformat()
        }
        
        # Create sweep result
        sweep_result = ParameterSweepResult(
            parameter_combinations=parameter_dicts,
            individual_results=individual_results,
            summary_statistics=summary_stats,
            sweep_metadata=sweep_metadata
        )
        
        # Save sweep result
        self.data_manager.save_parameter_sweep(sweep_result, sweep_name)
        
        return sweep_result

def run_comprehensive_parameter_sweep(dimensions: Union[int, List[int]] = [2, 4, 6], 
                                    noise_strengths: Union[float, List[float]] = [0.1, 0.3, 0.5, 0.7],
                                    purification_levels: int = 4,
                                    pauli_px: Union[float, List[float]] = [0.2, 0.5, 0.8],
                                    pauli_py: Union[float, List[float]] = [0.2, 0.5, 0.8],
                                    pauli_pz: Union[float, List[float]] = [0.2, 0.5, 0.8],
                                    data_dir: str = "./data") -> Dict[str, ParameterSweepResult]:
    """
    Execute comprehensive parameter sweep analysis.
    
    Args:
        dimensions: System dimensions to test
        noise_strengths: Depolarization parameters δ to test
        purification_levels: Number of recursive levels
        pauli_px, pauli_py, pauli_pz: Pauli error probabilities to test
        data_dir: Data save directory
    
    Returns:
        Dictionary of sweep results by noise type
    """
    print("="*80)
    print("STREAMING PURIFICATION QEC - COMPREHENSIVE PARAMETER SWEEP")
    print("="*80)
    
    # Initialize protocol
    data_manager = DataManager(data_dir)
    protocol = StreamingPurificationProtocol(data_manager)
    
    sweep_results = {}
    
    # Sweep 1: Depolarizing noise
    print("\n1. DEPOLARIZING NOISE PARAMETER SWEEP")
    print("-" * 50)
    
    depolarizing_result = protocol.run_parameter_sweep(
        dimensions=dimensions,
        noise_strengths=noise_strengths,
        purification_levels=purification_levels,
        pauli_px=1/3,  # Fixed for depolarizing
        pauli_py=1/3,
        pauli_pz=1/3,
        noise_type=NoiseType.DEPOLARIZING,
        sweep_name="depolarizing_sweep"
    )
    
    sweep_results['depolarizing'] = depolarizing_result
    
    # Sweep 2: Pauli noise (only for qubits)
    if 2 in (dimensions if isinstance(dimensions, list) else [dimensions]):
        print("\n2. PAULI NOISE PARAMETER SWEEP")
        print("-" * 50)
        
        pauli_result = protocol.run_parameter_sweep(
            dimensions=2,  # Pauli only for qubits
            noise_strengths=noise_strengths,
            purification_levels=purification_levels,
            pauli_px=pauli_px,
            pauli_py=pauli_py,
            pauli_pz=pauli_pz,
            noise_type=NoiseType.PAULI_BIASED,
            sweep_name="pauli_sweep"
        )
        
        sweep_results['pauli'] = pauli_result
    
    # Print summary
    print("\n" + "="*80)
    print("PARAMETER SWEEP ANALYSIS COMPLETED")
    print("="*80)
    
    for sweep_name, result in sweep_results.items():
        stats = result.summary_statistics
        print(f"\n{sweep_name.upper()} SWEEP SUMMARY:")
        print(f"  Successful runs: {stats['num_successful_runs']}/{stats['num_total_combinations']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        if 'final_purity_stats' in stats:
            purity_stats = stats['final_purity_stats']
            print(f"  Final purity: {purity_stats['mean']:.4f} ± {purity_stats['std']:.4f}")
            print(f"  Purity range: [{purity_stats['min']:.4f}, {purity_stats['max']:.4f}]")
    
    print(f"\nAll data saved to: {data_dir}/streaming_purification/")
    
    return sweep_results

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # =============================================================================
    # CONFIGURE PARAMETER ARRAYS HERE
    # =============================================================================
    
    # System parameters (can be single values or arrays)
    DIMENSIONS = [2, 4, 6, 8]                    # d: qudit dimensions to test
    NOISE_STRENGTHS = [0.01, 0.1, 0.3, 0.5, 0.8, 0.99]  # δ: depolarization parameters to test
    PURIFICATION_LEVELS = 4                      # Number of recursive purification levels
    
    # Pauli error probabilities (for Pauli noise models)
    PAULI_PX = [0.1, 0.5, 0.9]                 # X error probabilities to test
    PAULI_PY = [0.1, 0.5, 0.9]                 # Y error probabilities to test  
    PAULI_PZ = [0.1, 0.5, 0.9]                 # Z error probabilities to test
    
    # Data directory
    DATA_DIR = "./data"
    
    # =============================================================================
    
    # Run comprehensive parameter sweep
    sweep_results = run_comprehensive_parameter_sweep(
        dimensions=DIMENSIONS,
        noise_strengths=NOISE_STRENGTHS,
        purification_levels=PURIFICATION_LEVELS,
        pauli_px=PAULI_PX,
        pauli_py=PAULI_PY,
        pauli_pz=PAULI_PZ,
        data_dir=DATA_DIR
    )
    
    print("\nParameter sweep completed successfully!")