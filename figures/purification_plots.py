"""
Streaming Purification Quantum Error Correction - Visualization Module

This module creates publication-quality figures for analyzing the performance
of streaming purification as a quantum error correction protocol.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings

# Set up plotting style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

class PurificationPlotter:
    """
    Comprehensive plotting class for purification simulation results.
    
    Generates publication-quality figures including:
    - Purity evolution through recursive levels
    - Error reduction analysis
    - Resource scaling studies
    - Threshold comparisons
    - Dimension dependence plots
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        """
        Initialize the plotter with default styling.
        
        Args:
            figsize: Default figure size
            dpi: Resolution for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colors = sns.color_palette("husl", 10)
        
        # Set up matplotlib parameters for publication quality
        plt.rcParams.update({
            'font.size': 12,
            'axes.linewidth': 1.5,
            'lines.linewidth': 2,
            'patch.linewidth': 0.5,
            'lines.markersize': 8,
            'lines.markeredgewidth': 1.5,
            'xtick.major.size': 5,
            'xtick.minor.size': 3,
            'ytick.major.size': 5,
            'ytick.minor.size': 3,
        })
    
    def plot_purity_evolution(self, results: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot purity evolution through recursive purification levels.
        
        Args:
            results: Dictionary from recursive_purification or dimension_scaling_study
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        
        # Handle different input formats
        if hasattr(results, 'purity_evolution'):
            # Single result
            results_dict = {'Single Run': results}
        else:
            # Multiple results
            results_dict = results
        
        # Plot purity evolution
        for i, (label, result) in enumerate(results_dict.items()):
            levels = np.arange(len(result.purity_evolution))
            
            ax1.plot(levels, result.purity_evolution, 
                    marker='o', color=self.colors[i], label=f'{label} (d={result.dimension})',
                    linewidth=2.5, markersize=6)
            
            # Plot error evolution
            ax2.plot(levels, result.error_evolution,
                    marker='s', color=self.colors[i], label=f'{label} (d={result.dimension})',
                    linewidth=2.5, markersize=6)
        
        # Formatting
        ax1.set_ylabel('Purity Parameter λ', fontsize=14)
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        ax1.set_title('Recursive Purification Performance', fontsize=16, fontweight='bold')
        
        ax2.set_xlabel('Purification Level', fontsize=14)
        ax2.set_ylabel('Logical Error εₗ', fontsize=14)
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Purity evolution plot saved to {save_path}")
        
        return fig
    
    def plot_threshold_analysis(self, threshold_results: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot threshold analysis comparing different dimensions.
        
        Args:
            threshold_results: Dictionary mapping dimension to ThresholdResult
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # Main threshold plot
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Plot 1: Final purity vs initial noise
        for i, (dim, result) in enumerate(threshold_results.items()):
            valid_mask = ~np.isnan(result.final_purities)
            noise_vals = result.noise_levels[valid_mask]
            purity_vals = result.final_purities[valid_mask]
            
            ax1.plot(noise_vals, purity_vals, 
                    marker='o', color=self.colors[i], label=f'd = {dim}',
                    linewidth=2.5, markersize=5)
        
        ax1.set_xlabel('Initial Depolarization δ', fontsize=14)
        ax1.set_ylabel('Final Purity λ', fontsize=14)
        ax1.set_title('Purification Threshold Analysis', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Error reduction ratio
        for i, (dim, result) in enumerate(threshold_results.items()):
            valid_mask = ~np.isnan(result.error_reductions)
            noise_vals = result.noise_levels[valid_mask]
            error_ratios = result.error_reductions[valid_mask]
            
            ax2.semilogy(noise_vals, error_ratios,
                        marker='s', color=self.colors[i], label=f'd = {dim}',
                        linewidth=2, markersize=5)
        
        ax2.set_xlabel('Initial Depolarization δ', fontsize=12)
        ax2.set_ylabel('Error Reduction Ratio', fontsize=12)
        ax2.set_title('Error Suppression', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        # Plot 3: Resource costs
        for i, (dim, result) in enumerate(threshold_results.items()):
            valid_mask = ~np.isnan(result.resource_costs)
            noise_vals = result.noise_levels[valid_mask]
            costs = result.resource_costs[valid_mask]
            
            ax3.semilogy(noise_vals, costs,
                        marker='^', color=self.colors[i], label=f'd = {dim}',
                        linewidth=2, markersize=5)
        
        ax3.set_xlabel('Initial Depolarization δ', fontsize=12)
        ax3.set_ylabel('Total Amplification Cost', fontsize=12)
        ax3.set_title('Resource Requirements', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Threshold analysis plot saved to {save_path}")
        
        return fig
    
    def plot_dimension_scaling(self, scaling_results: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot how performance scales with system dimension.
        
        Args:
            scaling_results: Results from dimension_scaling_study
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        dimensions = list(scaling_results.keys())
        final_purities = [result.final_purity for result in scaling_results.values()]
        total_costs = [result.total_cost for result in scaling_results.values()]
        initial_purity = scaling_results[dimensions[0]].initial_purity
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Final purity vs dimension
        ax1.plot(dimensions, final_purities, 'o-', color=self.colors[0], 
                linewidth=3, markersize=8)
        ax1.axhline(y=initial_purity, color='red', linestyle='--', alpha=0.7, 
                   label=f'Initial λ = {initial_purity:.3f}')
        ax1.set_xlabel('System Dimension d', fontsize=14)
        ax1.set_ylabel('Final Purity λ', fontsize=14)
        ax1.set_title('Purification vs Dimension', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12)
        
        # Plot 2: Resource cost vs dimension
        ax2.semilogy(dimensions, total_costs, 's-', color=self.colors[1],
                    linewidth=3, markersize=8)
        ax2.set_xlabel('System Dimension d', fontsize=14)
        ax2.set_ylabel('Total Amplification Cost', fontsize=14)
        ax2.set_title('Resource Scaling', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Error reduction vs dimension
        error_reductions = []
        for result in scaling_results.values():
            initial_error = (1 - result.initial_purity) * (result.dimension - 1) / result.dimension
            final_error = (1 - result.final_purity) * (result.dimension - 1) / result.dimension
            error_reductions.append(final_error / initial_error if initial_error > 0 else 0)
        
        ax3.semilogy(dimensions, error_reductions, '^-', color=self.colors[2],
                    linewidth=3, markersize=8)
        ax3.set_xlabel('System Dimension d', fontsize=14)
        ax3.set_ylabel('Error Reduction Ratio', fontsize=14)
        ax3.set_title('Error Suppression Efficiency', fontsize=16, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Dimension scaling plot saved to {save_path}")
        
        return fig
    
    def plot_convergence_analysis(self, convergence_data: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot convergence properties of recursive purification.
        
        Args:
            convergence_data: Results from convergence_analysis
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Handle multiple dimensions
        if 'levels' in convergence_data:
            # Single dimension
            convergence_dict = {'Single': convergence_data}
        else:
            # Multiple dimensions
            convergence_dict = convergence_data
        
        # Plot 1: Convergence of final purity
        for i, (label, data) in enumerate(convergence_dict.items()):
            ax1.semilogx(2**data['levels'], data['final_purities'], 
                        'o-', color=self.colors[i], 
                        label=f'd = {data["dimension"]}' if isinstance(label, int) else label,
                        linewidth=2.5, markersize=6)
        
        ax1.set_xlabel('Number of Input Copies (2ⁿ)', fontsize=14)
        ax1.set_ylabel('Final Purity λ', fontsize=14)
        ax1.set_title('Convergence with System Size', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12)
        ax1.set_ylim(0, 1.05)
        
        # Plot 2: Resource cost scaling
        for i, (label, data) in enumerate(convergence_dict.items()):
            ax2.loglog(2**data['levels'], data['total_costs'], 
                      's-', color=self.colors[i],
                      label=f'd = {data["dimension"]}' if isinstance(label, int) else label,
                      linewidth=2.5, markersize=6)
        
        ax2.set_xlabel('Number of Input Copies (2ⁿ)', fontsize=14)
        ax2.set_ylabel('Total Amplification Cost', fontsize=14)
        ax2.set_title('Resource Cost Scaling', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Convergence analysis plot saved to {save_path}")
        
        return fig
    
    def create_summary_figure(self, benchmark_data: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive summary figure with all key results.
        
        Args:
            benchmark_data: Complete benchmark dataset
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Extract key results
        threshold_results = benchmark_data['threshold_comparison']
        qubit_detailed = benchmark_data['qubit_detailed']
        convergence_studies = benchmark_data['convergence_studies']
        
        # Plot 1: Main threshold comparison (spans top row)
        ax1 = fig.add_subplot(gs[0, :])
        for i, (dim, result) in enumerate(threshold_results.items()):
            valid_mask = ~np.isnan(result.final_purities)
            ax1.plot(result.noise_levels[valid_mask], result.final_purities[valid_mask],
                    'o-', color=self.colors[i], label=f'd = {dim}', 
                    linewidth=2.5, markersize=5)
        
        ax1.set_xlabel('Initial Depolarization δ', fontsize=14)
        ax1.set_ylabel('Final Purity λ', fontsize=14)
        ax1.set_title('Streaming Purification QEC: Threshold Analysis', 
                     fontsize=18, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12, ncol=3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Detailed qubit error reduction
        ax2 = fig.add_subplot(gs[1, 0])
        valid_mask = ~np.isnan(qubit_detailed.error_reductions)
        ax2.semilogy(qubit_detailed.noise_levels[valid_mask], 
                    qubit_detailed.error_reductions[valid_mask],
                    'o-', color=self.colors[0], linewidth=2.5)
        ax2.set_xlabel('Initial Depolarization δ')
        ax2.set_ylabel('Error Reduction Ratio')
        ax2.set_title('Qubit Error Suppression')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Resource scaling comparison
        ax3 = fig.add_subplot(gs[1, 1])
        dimensions = list(threshold_results.keys())
        avg_costs = [np.nanmean(result.resource_costs) for result in threshold_results.values()]
        ax3.semilogy(dimensions, avg_costs, 's-', color=self.colors[1], linewidth=2.5)
        ax3.set_xlabel('System Dimension d')
        ax3.set_ylabel('Average Resource Cost')
        ax3.set_title('Dimension Scaling')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Convergence comparison
        ax4 = fig.add_subplot(gs[1, 2])
        for i, (dim, data) in enumerate(convergence_studies.items()):
            ax4.semilogx(2**data['levels'], data['final_purities'],
                        'o-', color=self.colors[i], label=f'd = {dim}')
        ax4.set_xlabel('Input Copies (2ⁿ)')
        ax4.set_ylabel('Final Purity')
        ax4.set_title('System Size Convergence')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Success probability vs dimension (bottom left)
        ax5 = fig.add_subplot(gs[2, 0])
        test_purity = 0.5  # Test at moderate purity
        dims_test = np.arange(2, 11)
        success_probs = [(1 + test_purity**2 + (1-test_purity**2)/d)/2 for d in dims_test]
        ax5.plot(dims_test, success_probs, 'o-', color=self.colors[2], linewidth=2.5)
        ax5.set_xlabel('System Dimension d')
        ax5.set_ylabel('Success Probability')
        ax5.set_title('Swap Test Success Rate')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Theoretical vs practical threshold (bottom center)
        ax6 = fig.add_subplot(gs[2, 1])
        theoretical_threshold = np.ones_like(dimensions)  # 100% theoretical
        practical_threshold = [0.9 if d == 2 else 0.85 - 0.05*d for d in dimensions]  # Estimated practical
        
        ax6.plot(dimensions, theoretical_threshold, '--', color='green', 
                linewidth=3, label='Theoretical (100%)')
        ax6.plot(dimensions, practical_threshold, 'o-', color='red', 
                linewidth=2.5, label='Practical (estimated)')
        ax6.set_xlabel('System Dimension d')
        ax6.set_ylabel('Error Threshold')
        ax6.set_title('Threshold Comparison')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, 1.05)
        
        # Plot 7: Performance summary table (bottom right)
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        # Create summary statistics
        summary_text = "Performance Summary\n" + "="*20 + "\n\n"
        
        for dim in [2, 4, 8]:
            if dim in threshold_results:
                result = threshold_results[dim]
                best_performance = np.nanmax(result.final_purities)
                avg_cost = np.nanmean(result.resource_costs)
                summary_text += f"d = {dim}:\n"
                summary_text += f"  Max purity: {best_performance:.3f}\n"
                summary_text += f"  Avg cost: {avg_cost:.0f}\n\n"
        
        ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes, 
                fontsize=11, verticalalignment='top', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # Add overall title
        fig.suptitle('Streaming Purification Quantum Error Correction: Complete Analysis', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Summary figure saved to {save_path}")
        
        return fig

def generate_all_plots(benchmark_data: Dict, output_dir: str = "./plots/") -> None:
    """
    Generate all publication plots from benchmark data.
    
    Args:
        benchmark_data: Complete benchmark dataset
        output_dir: Directory to save all plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    plotter = PurificationPlotter()
    
    print(f"Generating plots in {output_dir}")
    
    # Main threshold analysis
    plotter.plot_threshold_analysis(
        benchmark_data['threshold_comparison'],
        save_path=f"{output_dir}/threshold_analysis.png"
    )
    
    # Dimension scaling (using d=2,4,8 comparison)
    scaling_subset = {d: benchmark_data['threshold_comparison'][d] 
                     for d in [2, 4, 8] if d in benchmark_data['threshold_comparison']}
    
    # Convergence analysis
    plotter.plot_convergence_analysis(
        benchmark_data['convergence_studies'],
        save_path=f"{output_dir}/convergence_analysis.png"
    )
    
    # Summary figure
    plotter.create_summary_figure(
        benchmark_data,
        save_path=f"{output_dir}/summary_figure.png"
    )
    
    print("All plots generated successfully!")

if __name__ == "__main__":
    # Example usage
    from purification_simulator import generate_benchmark_data
    
    print("Generating example plots...")
    
    # Generate small dataset for testing
    test_data = {
        'threshold_comparison': {},
        'convergence_studies': {}
    }
    
    # Quick test simulation
    from src.purification_simulator import PurificationSimulator
    
    sim = PurificationSimulator(dimension=2, verbose=True)
    test_data['threshold_comparison'][2] = sim.threshold_analysis(
        np.linspace(0.1, 0.8, 10), num_levels=3
    )
    
    test_data['convergence_studies'][2] = sim.convergence_analysis(0.3, max_levels=5)
    
    # Generate plots
    plotter = PurificationPlotter()
    fig = plotter.plot_threshold_analysis(test_data['threshold_comparison'])
    plt.show()
    
    print("Example plotting completed!")