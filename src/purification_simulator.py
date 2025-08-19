"""
Streaming Purification Quantum Error Correction - Numerical Simulator

This module implements systematic parameter studies for streaming purification QEC.
Data is automatically saved to data directory.
"""

import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings

@dataclass
class PurificationResult:
    """Data structure to store purification simulation results."""
    initial_purity: float
    final_purity: float
    purity_evolution: np.ndarray
    error_evolution: np.ndarray
    success_probabilities: np.ndarray
    amplification_iterations: np.ndarray
    total_cost: int
    dimension: int
    levels: int

@dataclass
class ThresholdResult:
    """Data structure to store threshold analysis results."""
    noise_levels: np.ndarray
    final_purities: np.ndarray
    error_reductions: np.ndarray
    resource_costs: np.ndarray
    dimension: int

@dataclass
class StudyParameters:
    """Configuration for systematic parameter studies."""
    # Dimension studies
    dimensions: List[int]
    
    # Noise level studies  
    noise_fine: np.ndarray      # Fine resolution for detailed analysis
    noise_coarse: np.ndarray    # Coarse resolution for quick surveys
    
    # Recursion depth studies
    max_levels: int
    convergence_levels: List[int]
    
    # Specific test points
    test_deltas: List[float]    # Specific noise levels of interest
    
    # Resource analysis
    resource_levels: int

class PurificationSimulator:
    """
    Comprehensive simulator for streaming purification quantum error correction
    with organized data saving and systematic parameter studies.
    """
    
    def __init__(self, dimension: int = 2, data_dir: str = "./data/", verbose: bool = False):
        """
        Initialize the purification simulator with data management.
        
        Args:
            dimension: Dimension of the quantum system (d=2 for qubits)
            data_dir: Directory to save all simulation data
            verbose: Enable detailed logging of simulation progress
        """
        self.dimension = dimension
        self.data_dir = data_dir
        self.verbose = verbose
        
        # Create data directory structure
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(f"{data_dir}/threshold_studies", exist_ok=True)
        os.makedirs(f"{data_dir}/convergence_studies", exist_ok=True)
        os.makedirs(f"{data_dir}/dimension_scaling", exist_ok=True)
        os.makedirs(f"{data_dir}/parameter_sweeps", exist_ok=True)
        
        # Validate dimension
        if dimension < 2:
            raise ValueError("Dimension must be at least 2")
            
        if self.verbose:
            print(f"Initialized PurificationSimulator for d={dimension} systems")
            print(f"Data will be saved to: {data_dir}")
    
    def purity_transformation(self, lambda_in: float) -> Tuple[float, float, int]:
        """
        Compute the effect of a single swap test on purity parameter.
        
        Args:
            lambda_in: Input purity parameter (0 ≤ λ ≤ 1)
            
        Returns:
            Tuple of (output_purity, success_probability, amplification_iterations)
        """
        if not 0 <= lambda_in <= 1:
            raise ValueError(f"Purity parameter must be in [0,1], got {lambda_in}")
        
        d = self.dimension
        delta = 1 - lambda_in  # Convert to depolarization parameter
        
        # Calculate success probability
        success_prob = 0.5 * (1 + (1-delta)**2 + delta*(2-delta)/d)
        
        # Calculate output purity using derived formula
        numerator = lambda_in * (1 + lambda_in + 2*(1-lambda_in)/d)
        denominator = 1 + lambda_in**2 + (1-lambda_in**2)/d
        lambda_out = numerator / denominator
        
        # Calculate amplitude amplification iterations needed
        if success_prob >= 1.0:
            amp_iterations = 0
        else:
            # Fixed-point quantum search formula
            theta = 2 * np.arcsin(np.sqrt(success_prob))
            amp_iterations = max(0, int(np.floor(np.pi / (4*np.arcsin(np.sqrt(success_prob))) - 0.5)))
        
        return lambda_out, success_prob, amp_iterations
    
    def recursive_purification(self, initial_delta: float, num_levels: int) -> PurificationResult:
        """
        Simulate complete recursive purification through binary tree.
        
        Args:
            initial_delta: Initial depolarization parameter (0 ≤ δ ≤ 1)
            num_levels: Number of recursion levels (depth of binary tree)
            
        Returns:
            PurificationResult containing complete simulation data
        """
        if not 0 <= initial_delta <= 1:
            raise ValueError(f"Depolarization parameter must be in [0,1], got {initial_delta}")
        
        if num_levels < 1:
            raise ValueError(f"Number of levels must be positive, got {num_levels}")
        
        # Initialize arrays to store evolution
        purity_evolution = np.zeros(num_levels + 1)
        error_evolution = np.zeros(num_levels + 1)
        success_probabilities = np.zeros(num_levels)
        amplification_iterations = np.zeros(num_levels)
        
        # Initial state
        initial_purity = 1 - initial_delta
        purity_evolution[0] = initial_purity
        error_evolution[0] = self._logical_error(initial_purity)
        
        current_purity = initial_purity
        total_cost = 0
        
        if self.verbose:
            print(f"Starting recursive purification:")
            print(f"  Initial δ={initial_delta:.3f}, λ={current_purity:.3f}")
            print(f"  Target levels: {num_levels}")
        
        # Recursive purification through each level
        for level in range(num_levels):
            # Number of parallel swap operations at this level
            num_swaps = 2**(num_levels - level - 1)
            
            # Apply purity transformation
            new_purity, success_prob, amp_iter = self.purity_transformation(current_purity)
            
            # Store results
            purity_evolution[level + 1] = new_purity
            error_evolution[level + 1] = self._logical_error(new_purity)
            success_probabilities[level] = success_prob
            amplification_iterations[level] = amp_iter
            
            # Calculate cost for this level
            level_cost = num_swaps * amp_iter
            total_cost += level_cost
            
            if self.verbose:
                print(f"  Level {level}: λ={new_purity:.4f}, P_success={success_prob:.4f}, "
                      f"amp_iter={amp_iter}, cost={level_cost}")
            
            current_purity = new_purity
        
        return PurificationResult(
            initial_purity=initial_purity,
            final_purity=current_purity,
            purity_evolution=purity_evolution,
            error_evolution=error_evolution,
            success_probabilities=success_probabilities,
            amplification_iterations=amplification_iterations,
            total_cost=total_cost,
            dimension=self.dimension,
            levels=num_levels
        )
    
    def _logical_error(self, purity: float) -> float:
        """Calculate logical error using generalized Grafe metric."""
        return (1 - purity) * (self.dimension - 1) / self.dimension
    
    def save_result(self, result: any, filename: str, metadata: Dict = None) -> str:
        """
        Save simulation result with metadata.
        
        Args:
            result: Result object to save
            filename: Base filename (without extension)
            metadata: Additional metadata to save
            
        Returns:
            Full path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = f"{filename}_{timestamp}"
        filepath = os.path.join(self.data_dir, f"{full_filename}.npz")
        
        # Prepare data for saving
        if isinstance(result, PurificationResult):
            save_data = asdict(result)
            save_data['purity_evolution'] = result.purity_evolution
            save_data['error_evolution'] = result.error_evolution
            save_data['success_probabilities'] = result.success_probabilities
            save_data['amplification_iterations'] = result.amplification_iterations
        elif isinstance(result, ThresholdResult):
            save_data = asdict(result)
            save_data['noise_levels'] = result.noise_levels
            save_data['final_purities'] = result.final_purities
            save_data['error_reductions'] = result.error_reductions
            save_data['resource_costs'] = result.resource_costs
        else:
            save_data = result
        
        # Add metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'timestamp': timestamp,
            'dimension': self.dimension,
            'simulator_version': '1.0',
            'save_date': datetime.now().isoformat()
        })
        
        # Save data and metadata
        np.savez_compressed(filepath, **save_data, metadata=metadata)
        
        # Save metadata as separate JSON for easy reading
        with open(filepath.replace('.npz', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.verbose:
            print(f"Results saved to: {filepath}")
        
        return filepath

def get_study_parameters() -> StudyParameters:
    """
    Define comprehensive parameter sets for systematic studies.
    
    Returns:
        StudyParameters object with all test configurations
    """
    return StudyParameters(
        # Dimension studies: Focus on computationally feasible range
        dimensions=[2, 3, 4, 5, 8, 10, 16],
        
        # Noise level studies
        noise_fine=np.linspace(0.01, 0.99, 99),        # High resolution for detailed threshold
        noise_coarse=np.linspace(0.05, 0.95, 19),      # Quick survey
        
        # Recursion depth studies  
        max_levels=12,
        convergence_levels=[1, 2, 3, 4, 5, 6, 8, 10],
        
        # Specific test points of interest
        test_deltas=[
            0.01,   # Very low noise
            0.1,    # Low noise  
            0.25,   # Moderate noise
            0.5,    # High noise
            0.75,   # Very high noise
            0.9,    # Extreme noise
            0.99    # Near-maximal noise
        ],
        
        # Resource analysis depth
        resource_levels=8
    )

def run_systematic_studies(data_dir: str = "./data/") -> Dict[str, str]:
    """
    Execute comprehensive systematic parameter studies.
    
    Args:
        data_dir: Directory to save all results
        
    Returns:
        Dictionary mapping study name to saved file path
    """
    params = get_study_parameters()
    saved_files = {}
    
    print("="*60)
    print("SYSTEMATIC PARAMETER STUDIES FOR STREAMING PURIFICATION QEC")
    print("="*60)
    
    # Study 1: Detailed Qubit Analysis
    print("\n1. Detailed Qubit Analysis (d=2)")
    print("-" * 40)
    
    qubit_sim = PurificationSimulator(dimension=2, data_dir=data_dir, verbose=True)
    
    # High-resolution threshold analysis
    threshold_result = ThresholdResult(
        noise_levels=params.noise_fine,
        final_purities=np.zeros_like(params.noise_fine),
        error_reductions=np.zeros_like(params.noise_fine),
        resource_costs=np.zeros_like(params.noise_fine),
        dimension=2
    )
    
    for i, delta in enumerate(params.noise_fine):
        try:
            result = qubit_sim.recursive_purification(delta, params.resource_levels)
            threshold_result.final_purities[i] = result.final_purity
            threshold_result.resource_costs[i] = result.total_cost
            
            initial_error = qubit_sim._logical_error(result.initial_purity)
            final_error = qubit_sim._logical_error(result.final_purity)
            threshold_result.error_reductions[i] = final_error / initial_error if initial_error > 0 else 0
            
        except Exception as e:
            print(f"Warning: Failed at δ={delta:.3f}: {e}")
            threshold_result.final_purities[i] = np.nan
            threshold_result.error_reductions[i] = np.nan
            threshold_result.resource_costs[i] = np.nan
    
    saved_files['qubit_detailed'] = qubit_sim.save_result(
        threshold_result, 
        "threshold_studies/qubit_detailed_threshold",
        {"study_type": "detailed_qubit_threshold", "resolution": "high"}
    )
    
    # Study 2: Dimension Scaling Analysis
    print("\n2. Dimension Scaling Analysis")
    print("-" * 40)
    
    dimension_results = {}
    
    for d in params.dimensions:
        print(f"  Testing dimension d={d}")
        sim = PurificationSimulator(dimension=d, data_dir=data_dir, verbose=False)
        
        # Threshold analysis for this dimension
        threshold_result = ThresholdResult(
            noise_levels=params.noise_coarse,
            final_purities=np.zeros_like(params.noise_coarse),
            error_reductions=np.zeros_like(params.noise_coarse),
            resource_costs=np.zeros_like(params.noise_coarse),
            dimension=d
        )
        
        for i, delta in enumerate(params.noise_coarse):
            try:
                result = sim.recursive_purification(delta, 6)  # Moderate depth for survey
                threshold_result.final_purities[i] = result.final_purity
                threshold_result.resource_costs[i] = result.total_cost
                
                initial_error = sim._logical_error(result.initial_purity)
                final_error = sim._logical_error(result.final_purity)
                threshold_result.error_reductions[i] = final_error / initial_error if initial_error > 0 else 0
                
            except Exception as e:
                threshold_result.final_purities[i] = np.nan
                threshold_result.error_reductions[i] = np.nan
                threshold_result.resource_costs[i] = np.nan
        
        dimension_results[d] = threshold_result
        
        # Save individual dimension result
        saved_files[f'dimension_{d}'] = sim.save_result(
            threshold_result,
            f"dimension_scaling/threshold_d{d}",
            {"study_type": "dimension_scaling", "dimension": d}
        )
    
    # Study 3: Convergence Analysis  
    print("\n3. Convergence Analysis")
    print("-" * 40)
    
    convergence_results = {}
    
    for d in [2, 4, 8]:  # Selected dimensions for detailed convergence study
        print(f"  Convergence study for d={d}")
        sim = PurificationSimulator(dimension=d, data_dir=data_dir, verbose=False)
        
        convergence_data = {
            'dimension': d,
            'test_deltas': params.test_deltas,
            'levels': params.convergence_levels,
            'results': {}
        }
        
        for delta in params.test_deltas:
            level_results = {
                'final_purities': [],
                'total_costs': [],
                'error_reductions': []
            }
            
            for levels in params.convergence_levels:
                try:
                    result = sim.recursive_purification(delta, levels)
                    level_results['final_purities'].append(result.final_purity)
                    level_results['total_costs'].append(result.total_cost)
                    
                    initial_error = sim._logical_error(result.initial_purity)
                    final_error = sim._logical_error(result.final_purity)
                    level_results['error_reductions'].append(
                        final_error / initial_error if initial_error > 0 else 0
                    )
                    
                except Exception as e:
                    level_results['final_purities'].append(np.nan)
                    level_results['total_costs'].append(np.nan)
                    level_results['error_reductions'].append(np.nan)
            
            convergence_data['results'][delta] = level_results
        
        convergence_results[d] = convergence_data
        
        # Save convergence result
        saved_files[f'convergence_d{d}'] = sim.save_result(
            convergence_data,
            f"convergence_studies/convergence_d{d}",
            {"study_type": "convergence_analysis", "dimension": d}
        )
    
    # Study 4: Resource Scaling Deep Dive
    print("\n4. Resource Scaling Analysis")
    print("-" * 40)
    
    resource_study = {
        'dimensions': params.dimensions,
        'test_delta': 0.3,  # Fixed moderate noise level
        'max_levels': params.max_levels,
        'results': {}
    }
    
    for d in params.dimensions:
        print(f"  Resource scaling for d={d}")
        sim = PurificationSimulator(dimension=d, data_dir=data_dir, verbose=False)
        
        level_data = {
            'levels': list(range(1, params.max_levels + 1)),
            'final_purities': [],
            'total_costs': [],
            'cost_per_level': []
        }
        
        for levels in range(1, params.max_levels + 1):
            try:
                result = sim.recursive_purification(0.3, levels)
                level_data['final_purities'].append(result.final_purity)
                level_data['total_costs'].append(result.total_cost)
                level_data['cost_per_level'].append(result.total_cost / levels)
                
            except Exception as e:
                level_data['final_purities'].append(np.nan)
                level_data['total_costs'].append(np.nan)
                level_data['cost_per_level'].append(np.nan)
        
        resource_study['results'][d] = level_data
    
    saved_files['resource_scaling'] = PurificationSimulator(
        dimension=2, data_dir=data_dir
    ).save_result(
        resource_study,
        "parameter_sweeps/resource_scaling",
        {"study_type": "resource_scaling", "fixed_delta": 0.3}
    )
    
    # Summary
    print("\n" + "="*60)
    print("SYSTEMATIC STUDIES COMPLETED")
    print("="*60)
    print(f"Total studies conducted: {len(saved_files)}")
    print(f"Data saved to: {data_dir}")
    print("\nSaved files:")
    for study, filepath in saved_files.items():
        print(f"  {study}: {os.path.basename(filepath)}")
    
    return saved_files

if __name__ == "__main__":
    # Run all systematic studies
    run_systematic_studies()
    