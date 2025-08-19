"""
Streaming Purification Quantum Error Correction - Data Loading and Visualization

This module loads saved simulation data and creates publication-quality figures.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os
import json
import glob
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Set up plotting style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

class DataLoader:
    """
    Data loading utilities for purification simulation results.
    """
    
    def __init__(self, data_dir: str = "./data/"):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing saved simulation data
        """
        self.data_dir = data_dir
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    def load_latest_file(self, pattern: str) -> Tuple[Dict, Dict]:
        """
        Load the most recent file matching a pattern.
        
        Args:
            pattern: File pattern to match (e.g., "qubit_detailed_threshold_*")
            
        Returns:
            Tuple of (data_dict, metadata_dict)
        """
        search_pattern = os.path.join(self.data_dir, "**", f"{pattern}.npz")
        files = glob.glob(search_pattern, recursive=True)
        
        if not files:
            raise FileNotFoundError(f"No files found matching pattern: {pattern}")
        
        # Get most recent file
        latest_file = max(files, key=os.path.getmtime)
        
        # Load data
        data = dict(np.load(latest_file, allow_pickle=True))
        
        # Load metadata
        metadata_file = latest_file.replace('.npz', '_metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        print(f"Loaded: {os.path.basename(latest_file)}")
        return data, metadata
    
    def load_dimension_scaling_data(self) -> Dict[int, Tuple[Dict, Dict]]:
        """
        Load all dimension scaling data files.
        
        Returns:
            Dictionary mapping dimension to (data, metadata)
        """
        dimension_data = {}
        
        # Look for dimension scaling files
        pattern = os.path.join(self.data_dir, "dimension_scaling", "threshold_d*_*.npz")
        files = glob.glob(pattern)
        
        for file in files:
            # Extract dimension from filename
            basename = os.path.basename(file)
            try:
                # Extract dimension number from filename like "threshold_d2_20231201_123456.npz"
                d = int(basename.split('_')[1][1:])  # Extract number after 'd'
                
                data = dict(np.load(file, allow_pickle=True))
                
                metadata_file = file.replace('.npz', '_metadata.json')
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                else:
                    metadata = {}
                
                dimension_data[d] = (data, metadata)
                
            except (ValueError, IndexError):
                print(f"Warning: Could not parse dimension from filename: {basename}")
                continue
        
        print(f"Loaded dimension scaling data for d = {sorted(dimension_data.keys())}")
        return dimension_data
    
    def load_convergence_data(self) -> Dict[int, Tuple[Dict, Dict]]:
        """
        Load all convergence study data files.
        
        Returns:
            Dictionary mapping dimension to (data, metadata)
        """
        convergence_data = {}
        
        pattern = os.path.join(self.data_dir, "convergence_studies", "convergence_d*_*.npz")
        files = glob.glob(pattern)
        
        for file in files:
            basename = os.path.basename(file)
            try:
                d = int(basename.split('_')[1][1:])  # Extract dimension
                
                data = dict(np.load(file, allow_pickle=True))
                
                metadata_file = file.replace('.npz', '_metadata.json')
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                else:
                    metadata = {}
                
                convergence_data[d] = (data, metadata)
                
            except (ValueError, IndexError):
                print(f"Warning: Could not parse dimension from filename: {basename}")
                continue
        
        print(f"Loaded convergence data for d = {sorted(convergence_data.keys())}")
        return convergence_data

class PurificationPlotter:
    """
    Publication-quality plotting class that loads data automatically.
    """
    
    def __init__(self, data_dir: str = "./data/", figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        """
        Initialize plotter with automatic data loading.
        
        Args:
            data_dir: Directory containing saved data
            figsize: Default figure size
            dpi: Resolution for saved figures
        """
        self.data_loader = DataLoader(data_dir)
        self.figsize = figsize
        self.dpi = dpi
        self.colors = sns.color_palette("husl", 10)
        
        # Set up matplotlib parameters
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
        
        print(f"PurificationPlotter initialized with data from: {data_dir}")
    
    def plot_qubit_detailed_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot detailed qubit threshold analysis from saved data.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        # Load qubit detailed data
        data, metadata = self.data_loader.load_latest_file("*/qubit_detailed_threshold_*")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        noise_levels = data['noise_levels']
        final_purities = data['final_purities']
        error_reductions = data['error_reductions']
        resource_costs = data['resource_costs']
        
        # Remove NaN values
        valid_mask = ~(np.isnan(final_purities) | np.isnan(error_reductions) | np.isnan(resource_costs))
        
        # Plot 1: Final purity vs noise
        ax1.plot(noise_levels[valid_mask], final_purities[valid_mask], 
                'o-', color=self.colors[0], linewidth=2.5, markersize=4)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random guess')
        ax1.set_xlabel('Initial Depolarization δ')
        ax1.set_ylabel('Final Purity λ')
        ax1.set_title('Qubit Purification Threshold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Error reduction
        ax2.semilogy(noise_levels[valid_mask], error_reductions[valid_mask],
                    's-', color=self.colors[1], linewidth=2.5, markersize=4)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No improvement')
        ax2.set_xlabel('Initial Depolarization δ')
        ax2.set_ylabel('Error Reduction Ratio')
        ax2.set_title('Error Suppression Performance')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Resource costs
        ax3.semilogy(noise_levels[valid_mask], resource_costs[valid_mask],
                    '^-', color=self.colors[2], linewidth=2.5, markersize=4)
        ax3.set_xlabel('Initial Depolarization δ')
        ax3.set_ylabel('Total Amplification Cost')
        ax3.set_title('Resource Requirements')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Success probability analysis
        success_probs = 0.5 * (1 + (1-noise_levels)**2 + noise_levels*(2-noise_levels)/2)
        ax4.plot(noise_levels, success_probs, 'o-', color=self.colors[3], linewidth=2.5, markersize=4)
        ax4.set_xlabel('Initial Depolarization δ')
        ax4.set_ylabel('Swap Test Success Probability')
        ax4.set_title('Single-Level Success Rate')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.suptitle('Detailed Qubit Purification Analysis', fontsize=16, fontweight='bold', y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Qubit analysis plot saved to {save_path}")
        
        return fig
    
    def plot_dimension_comparison(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison across different dimensions from saved data.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        # Load dimension scaling data
        dimension_data = self.data_loader.load_dimension_scaling_data()
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot threshold curves for each dimension
        for i, (d, (data, metadata)) in enumerate(sorted(dimension_data.items())):
            noise_levels = data['noise_levels']
            final_purities = data['final_purities']
            error_reductions = data['error_reductions']
            resource_costs = data['resource_costs']
            
            valid_mask = ~np.isnan(final_purities)
            
            # Plot 1: Threshold comparison
            ax1.plot(noise_levels[valid_mask], final_purities[valid_mask],
                    'o-', color=self.colors[i], label=f'd = {d}', 
                    linewidth=2.5, markersize=5)
            
            # Plot 2: Error reduction comparison
            valid_error_mask = valid_mask & ~np.isnan(error_reductions)
            if np.any(valid_error_mask):
                ax2.semilogy(noise_levels[valid_error_mask], error_reductions[valid_error_mask],
                            's-', color=self.colors[i], label=f'd = {d}',
                            linewidth=2, markersize=4)
            
            # Plot 3: Resource cost comparison
            valid_cost_mask = valid_mask & ~np.isnan(resource_costs)
            if np.any(valid_cost_mask):
                ax3.semilogy(noise_levels[valid_cost_mask], resource_costs[valid_cost_mask],
                            '^-', color=self.colors[i], label=f'd = {d}',
                            linewidth=2, markersize=4)
        
        # Formatting
        ax1.set_xlabel('Initial Depolarization δ', fontsize=14)
        ax1.set_ylabel('Final Purity λ', fontsize=14)
        ax1.set_title('Threshold vs Dimension', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        ax2.set_xlabel('Initial Depolarization δ', fontsize=14)
        ax2.set_ylabel('Error Reduction Ratio', fontsize=14)
        ax2.set_title('Error Suppression vs Dimension', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        ax3.set_xlabel('Initial Depolarization δ', fontsize=14)
        ax3.set_ylabel('Total Amplification Cost', fontsize=14)
        ax3.set_title('Resource Cost vs Dimension', fontsize=16, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Dimension comparison plot saved to {save_path}")
        
        return fig
    
    def plot_convergence_studies(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot convergence analysis from saved data.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        # Load convergence data
        convergence_data = self.data_loader.load_convergence_data()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Test delta for detailed analysis
        test_delta = 0.3
        
        # Plot convergence for different dimensions
        for i, (d, (data, metadata)) in enumerate(sorted(convergence_data.items())):
            results = data['results'].item()  # Extract from numpy array
            
            if test_delta in results:
                levels = data['levels']
                delta_results = results[test_delta]
                
                final_purities = np.array(delta_results['final_purities'])
                total_costs = np.array(delta_results['total_costs'])
                error_reductions = np.array(delta_results['error_reductions'])
                
                valid_mask = ~np.isnan(final_purities)
                
                # Plot 1: Purity convergence
                ax1.semilogx(2**np.array(levels)[valid_mask], final_purities[valid_mask],
                            'o-', color=self.colors[i], label=f'd = {d}',
                            linewidth=2.5, markersize=6)
                
                # Plot 2: Cost scaling
                ax2.loglog(2**np.array(levels)[valid_mask], total_costs[valid_mask],
                          's-', color=self.colors[i], label=f'd = {d}',
                          linewidth=2.5, markersize=6)
        
        # Plot 3 & 4: Multi-delta analysis for d=2
        if 2 in convergence_data:
            data, _ = convergence_data[2]
            results = data['results'].item()
            levels = data['levels']
            test_deltas = data['test_deltas']
            
            for j, delta in enumerate([0.1, 0.3, 0.5, 0.75]):
                if delta in results:
                    delta_results = results[delta]
                    final_purities = np.array(delta_results['final_purities'])
                    error_reductions = np.array(delta_results['error_reductions'])
                    
                    valid_mask = ~np.isnan(final_purities)
                    
                    ax3.semilogx(2**np.array(levels)[valid_mask], final_purities[valid_mask],
                                'o-', color=self.colors[j], label=f'δ = {delta}',
                                linewidth=2, markersize=5)
                    
                    ax4.loglog(2**np.array(levels)[valid_mask], error_reductions[valid_mask],
                              's-', color=self.colors[j], label=f'δ = {delta}',
                              linewidth=2, markersize=5)
        
        # Formatting
        ax1.set_xlabel('Number of Input Copies (2ⁿ)')
        ax1.set_ylabel('Final Purity λ')
        ax1.set_title(f'Convergence vs Dimension (δ = {test_delta})')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        ax2.set_xlabel('Number of Input Copies (2ⁿ)')
        ax2.set_ylabel('Total Amplification Cost')
        ax2.set_title('Resource Scaling vs Dimension')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        ax3.set_xlabel('Number of Input Copies (2ⁿ)')
        ax3.set_ylabel('Final Purity λ')
        ax3.set_title('Convergence vs Noise Level (d = 2)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_ylim(0, 1)
        
        ax4.set_xlabel('Number of Input Copies (2ⁿ)')
        ax4.set_ylabel('Error Reduction Ratio')
        ax4.set_title('Error Suppression vs Noise Level')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.suptitle('Convergence Analysis', fontsize=16, fontweight='bold', y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Convergence analysis plot saved to {save_path}")
        
        return fig
    
    def create_publication_summary(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive publication-ready summary figure.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # Load all data
        try:
            qubit_data, _ = self.data_loader.load_latest_file("*/qubit_detailed_threshold_*")
            dimension_data = self.data_loader.load_dimension_scaling_data()
            convergence_data = self.data_loader.load_convergence_data()
            
            # Main threshold plot (top row, full width)
            ax_main = fig.add_subplot(gs[0, :])
            
            # Plot qubit detailed threshold
            noise_levels = qubit_data['noise_levels']
            final_purities = qubit_data['final_purities']
            valid_mask = ~np.isnan(final_purities)
            
            ax_main.plot(noise_levels[valid_mask], final_purities[valid_mask],
                        'o-', color=self.colors[0], linewidth=3, markersize=6,
                        label='d = 2 (detailed)', zorder=10)
            
            # Add other dimensions
            for i, (d, (data, _)) in enumerate(sorted(dimension_data.items())):
                if d != 2:  # Skip d=2 since we already plotted detailed version
                    noise = data['noise_levels']
                    purity = data['final_purities']
                    valid = ~np.isnan(purity)
                    ax_main.plot(noise[valid], purity[valid],
                                's-', color=self.colors[i+1], linewidth=2.5, markersize=5,
                                label=f'd = {d}', alpha=0.8)
            
            ax_main.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random state')
            ax_main.set_xlabel('Initial Depolarization δ', fontsize=14)
            ax_main.set_ylabel('Final Purity λ', fontsize=14)
            ax_main.set_title('Streaming Purification QEC: Universal Threshold Analysis', 
                             fontsize=18, fontweight='bold')
            ax_main.grid(True, alpha=0.3)
            ax_main.legend(fontsize=12, ncol=4)
            ax_main.set_xlim(0, 1)
            ax_main.set_ylim(0, 1)
            
            # Subplot arrangement for remaining plots
            subplot_positions = [
                (gs[1, 0], "Error Reduction (d=2)"),
                (gs[1, 1], "Resource Scaling"),
                (gs[1, 2], "Success Probability"),
                (gs[2, 0], "Convergence Rate"),
                (gs[2, 1], "Dimension Comparison"),
                (gs[2, 2], "Performance Summary")
            ]
            
            # Error reduction plot
            ax1 = fig.add_subplot(gs[1, 0])
            error_reductions = qubit_data['error_reductions']
            valid_error = valid_mask & ~np.isnan(error_reductions)
            ax1.semilogy(noise_levels[valid_error], error_reductions[valid_error],
                        'o-', color=self.colors[0], linewidth=2.5)
            ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7)
            ax1.set_xlabel('Initial Depolarization δ')
            ax1.set_ylabel('Error Reduction Ratio')
            ax1.set_title('Error Suppression (d=2)')
            ax1.grid(True, alpha=0.3)
            
            # Resource scaling plot
            ax2 = fig.add_subplot(gs[1, 1])
            dims = sorted(dimension_data.keys())
            avg_costs = []
            for d in dims:
                data, _ = dimension_data[d]
                costs = data['resource_costs']
                avg_costs.append(np.nanmean(costs))
            
            ax2.semilogy(dims, avg_costs, 's-', color=self.colors[1], linewidth=2.5, markersize=8)
            ax2.set_xlabel('System Dimension d')
            ax2.set_ylabel('Average Resource Cost')
            ax2.set_title('Dimension Scaling')
            ax2.grid(True, alpha=0.3)
            
            # Success probability
            ax3 = fig.add_subplot(gs[1, 2])
            test_noise = np.linspace(0, 1, 100)
            for i, d in enumerate([2, 4, 8]):
                success_prob = 0.5 * (1 + (1-test_noise)**2 + test_noise*(2-test_noise)/d)
                ax3.plot(test_noise, success_prob, '-', color=self.colors[i], 
                        linewidth=2.5, label=f'd = {d}')
            ax3.set_xlabel('Depolarization δ')
            ax3.set_ylabel('Success Probability')
            ax3.set_title('Swap Test Success Rate')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            ax3.set_ylim(0, 1)
            
            # Add remaining plots with available data
            # ... (similar structure for remaining subplots)
            
        except Exception as e:
            print(f"Warning: Could not load all data for summary plot: {e}")
            # Create simplified summary with available data
            ax_main = fig.add_subplot(gs[:, :])
            ax_main.text(0.5, 0.5, f"Data loading error: {e}\nRun systematic studies first.",
                        ha='center', va='center', fontsize=16, 
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
            ax_main.set_xlim(0, 1)
            ax_main.set_ylim(0, 1)
            ax_main.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Publication summary saved to {save_path}")
        
        return fig

def generate_all_publication_figures(data_dir: str = "./data/", output_dir: str = "./figures/") -> None:
    """
    Generate all publication figures from saved data.
    
    Args:
        data_dir: Directory containing saved simulation data
        output_dir: Directory to save figure files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plotter = PurificationPlotter(data_dir=data_dir)
    
    print(f"Generating publication figures from {data_dir}")
    print(f"Saving to: {output_dir}")
    
    try:
        # Figure 1: Detailed qubit analysis
        plotter.plot_qubit_detailed_analysis(f"{output_dir}/fig1_qubit_detailed.pdf")
        
        # Figure 2: Dimension comparison
        plotter.plot_dimension_comparison(f"{output_dir}/fig2_dimension_comparison.pdf")
        
        # Figure 3: Convergence studies  
        plotter.plot_convergence_studies(f"{output_dir}/fig3_convergence_analysis.pdf")
        
        # Figure 4: Publication summary
        plotter.create_publication_summary(f"{output_dir}/fig4_publication_summary.pdf")
        
        print("All publication figures generated successfully!")
        
    except Exception as e:
        print(f"Error generating figures: {e}")
        print("Make sure to run systematic studies first to generate data.")

if __name__ == "__main__":
    # Generate all figures from saved data
    generate_all_publication_figures()