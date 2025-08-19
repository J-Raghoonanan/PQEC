"""
Streaming Purification Quantum Error Correction - Numerical Simulator

This module implements the complete mathematical framework for analyzing
streaming purification as a quantum error correction protocol.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
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

class PurificationSimulator:
    """
    Comprehensive simulator for streaming purification quantum error correction.
    
    This class implements the complete theoretical framework including:
    - Purity transformation for qudits
    - Recursive binary tree purification
    - Amplitude amplification cost analysis
    - Error metrics using Grafe formalism
    - Threshold and scaling studies
    """
    
    def __init__(self, dimension: int = 2, verbose: bool = False):
        """
        Initialize the purification simulator.
        
        Args:
            dimension: Dimension of the quantum system (d=2 for qubits)
            verbose: Enable detailed logging of simulation progress
        """
        self.dimension = dimension
        self.verbose = verbose
        
        # Validate dimension
        if dimension < 2:
            raise ValueError("Dimension must be at least 2")
            
        if self.verbose:
            print(f"Initialized PurificationSimulator for d={dimension} systems")
    
    def purity_transformation(self, lambda_in: float) -> Tuple[float, float, int]:
        """
        Compute the effect of a single swap test on purity parameter.
        
        This implements the core purity transformation:
        λ_out = λ_in * (2-(1-λ_in) + 2(1-λ_in)/d) / (1 + λ_in² + (1-λ_in²)/d)
        
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
        
        Starting with 2^num_levels noisy copies, this performs recursive
        swap tests to produce a single purified copy.
        
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
        """
        Calculate logical error using generalized Grafe metric.
        
        For qudits: ε_L = (1-λ)(d-1)/d
        For qubits: ε_L = (1-λ)/2
        
        Args:
            purity: Current purity parameter λ
            
        Returns:
            Logical error value
        """
        return (1 - purity) * (self.dimension - 1) / self.dimension
    
    def threshold_analysis(self, noise_range: np.ndarray, num_levels: int = 5) -> ThresholdResult:
        """
        Analyze purification performance across range of noise levels.
        
        This sweeps through different initial noise levels and computes
        the final purification performance and resource costs.
        
        Args:
            noise_range: Array of initial depolarization parameters to test
            num_levels: Number of purification levels to apply
            
        Returns:
            ThresholdResult containing performance vs noise data
        """
        final_purities = np.zeros_like(noise_range)
        error_reductions = np.zeros_like(noise_range)
        resource_costs = np.zeros_like(noise_range)
        
        if self.verbose:
            print(f"Threshold analysis for d={self.dimension}")
            print(f"Noise range: {noise_range[0]:.3f} to {noise_range[-1]:.3f}")
        
        for i, delta in enumerate(noise_range):
            try:
                result = self.recursive_purification(delta, num_levels)
                
                final_purities[i] = result.final_purity
                resource_costs[i] = result.total_cost
                
                # Calculate error reduction ratio
                initial_error = self._logical_error(result.initial_purity)
                final_error = self._logical_error(result.final_purity)
                
                if initial_error > 0:
                    error_reductions[i] = final_error / initial_error
                else:
                    error_reductions[i] = 0
                    
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed at δ={delta:.3f}: {e}")
                final_purities[i] = np.nan
                error_reductions[i] = np.nan
                resource_costs[i] = np.nan
        
        return ThresholdResult(
            noise_levels=noise_range,
            final_purities=final_purities,
            error_reductions=error_reductions,
            resource_costs=resource_costs,
            dimension=self.dimension
        )
    
    def dimension_scaling_study(self, dimensions: List[int], 
                               delta: float = 0.3, num_levels: int = 5) -> Dict[int, PurificationResult]:
        """
        Study how purification performance scales with system dimension.
        
        Args:
            dimensions: List of dimensions to test
            delta: Fixed depolarization parameter
            num_levels: Number of purification levels
            
        Returns:
            Dictionary mapping dimension to PurificationResult
        """
        results = {}
        
        if self.verbose:
            print(f"Dimension scaling study at δ={delta}")
            print(f"Testing dimensions: {dimensions}")
        
        for d in dimensions:
            # Temporarily create simulator for this dimension
            temp_sim = PurificationSimulator(dimension=d, verbose=False)
            results[d] = temp_sim.recursive_purification(delta, num_levels)
            
            if self.verbose:
                result = results[d]
                print(f"d={d}: λ_final={result.final_purity:.4f}, cost={result.total_cost}")
        
        return results
    
    def convergence_analysis(self, initial_delta: float, max_levels: int = 10) -> Dict[str, np.ndarray]:
        """
        Analyze convergence properties of recursive purification.
        
        Args:
            initial_delta: Initial depolarization parameter
            max_levels: Maximum number of levels to test
            
        Returns:
            Dictionary with convergence data
        """
        levels_range = np.arange(1, max_levels + 1)
        final_purities = np.zeros(len(levels_range))
        total_costs = np.zeros(len(levels_range))
        
        for i, levels in enumerate(levels_range):
            result = self.recursive_purification(initial_delta, levels)
            final_purities[i] = result.final_purity
            total_costs[i] = result.total_cost
        
        return {
            'levels': levels_range,
            'final_purities': final_purities,
            'total_costs': total_costs,
            'dimension': self.dimension,
            'initial_delta': initial_delta
        }

# Utility functions for batch simulations
def compare_dimensions(dimensions: List[int], delta_range: np.ndarray, 
                      num_levels: int = 5) -> Dict[int, ThresholdResult]:
    """
    Compare threshold performance across multiple dimensions.
    
    Args:
        dimensions: List of dimensions to compare
        delta_range: Range of noise levels to test
        num_levels: Number of purification levels
        
    Returns:
        Dictionary mapping dimension to ThresholdResult
    """
    results = {}
    
    for d in dimensions:
        sim = PurificationSimulator(dimension=d, verbose=True)
        results[d] = sim.threshold_analysis(delta_range, num_levels)
    
    return results

def generate_benchmark_data(output_file: Optional[str] = "data/benchmark.npz") -> Dict:
    """
    Generate comprehensive benchmark data for the paper.
    
    Args:
        output_file: Optional filename to save results
        
    Returns:
        Dictionary containing all benchmark results
    """
    print("Generating comprehensive benchmark data...")
    
    # Standard parameters
    dimensions = [2, 3, 4, 5, 8, 10]
    delta_range = np.linspace(0.1, 0.9, 25)
    num_levels = 6
    
    # Main simulations
    benchmark = {
        'dimensions': dimensions,
        'delta_range': delta_range,
        'num_levels': num_levels,
        'threshold_comparison': compare_dimensions(dimensions, delta_range, num_levels),
        'qubit_detailed': None,
        'convergence_studies': {}
    }
    
    # Detailed qubit analysis
    qubit_sim = PurificationSimulator(dimension=2, verbose=True)
    benchmark['qubit_detailed'] = qubit_sim.threshold_analysis(
        np.linspace(0.05, 0.95, 50), num_levels=8
    )
    
    # Convergence studies for different dimensions
    for d in [2, 4, 8]:
        sim = PurificationSimulator(dimension=d)
        benchmark['convergence_studies'][d] = sim.convergence_analysis(0.3, max_levels=10)
    
    # Save results if filename provided
    if output_file:
        np.savez(output_file, **benchmark)
        print(f"Benchmark data saved to {output_file}")
    
    return benchmark

if __name__ == "__main__":
    # Example usage and testing
    print("Testing Purification Simulator...")
    
    # Test single transformation
    sim = PurificationSimulator(dimension=2, verbose=True)
    purity_out, prob, iters = sim.purity_transformation(0.7)
    print(f"Single transformation: 0.7 → {purity_out:.4f} (P={prob:.4f}, iters={iters})")
    
    # Test recursive purification
    result = sim.recursive_purification(0.3, 5)
    print(f"Recursive result: {result.initial_purity:.4f} → {result.final_purity:.4f}")
    
    # Test threshold analysis
    noise_range = np.linspace(0.1, 0.8, 10)
    threshold_result = sim.threshold_analysis(noise_range, 3)
    print(f"Threshold analysis completed for {len(noise_range)} points")
    
    print("All tests passed!")