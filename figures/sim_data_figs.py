"""
Plotting Analysis for Theoretical Streaming QEC Simulation Data
Creates key plots showing system size effects and protocol performance
Focuses on final logical error vs system size M (number of qubits)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import pickle
import os
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from pathlib import Path
import glob
from dataclasses import dataclass

# Set publication-quality plotting parameters
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 30,
    'axes.labelsize': 25,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 30,
    'font.family': 'DejaVu Sans',
    'mathtext.fontset': 'dejavusans',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 3,
    'lines.markersize': 8
})

# Color schemes
PROTOCOL_COLORS = {
    'depolarizing': '#2E86AB',
    'symmetric_pauli': '#A23B72', 
    'dephasing': '#F18F01',
    'pure_dephasing': '#F18F01',
    'dephasing_z': '#F18F01',
    'dephasing_x': '#C73E1D',
    'streaming_qec': '#2E86AB',
    'no_correction': '#666666'
}

@dataclass
class ExperimentResult:
    """Store results from a single experiment - must match the original class definition."""
    M: int
    N: int
    delta: float
    noise_type: str
    trial_number: int
    metrics_by_level: Dict[int, Dict[str, float]]  # level -> {'fidelity': float, 'logical_error': float}
    final_logical_error: float
    final_fidelity: float

class TheoreticalDataPlotter:
    """Plot analysis for theoretical streaming QEC simulation data."""
    
    def __init__(self, data_dir: str = "../data/simulations"):
        # If running from figures/ directory, data is in ../data/simulations
        # If data_dir doesn't exist, try alternative paths
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            # Try other common paths
            alt_paths = ["data/simulations", "./data/simulations", "../data/simulations"]
            for alt_path in alt_paths:
                if Path(alt_path).exists():
                    self.data_dir = Path(alt_path)
                    break
        
        self.figures_dir = Path("figures/results_v3_sim")  # Create in current directory
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Looking for data in: {self.data_dir.absolute()}")
        print(f"Saving figures to: {self.figures_dir.absolute()}")
        
        # Load simulation data
        self.raw_results = self._load_raw_results()
        self.aggregated_data = self._load_aggregated_data()
        self.processed_df = self._process_to_dataframe()
    
    def _load_raw_results(self):
        """Load raw simulation results from pickle files."""
        pickle_files = list(self.data_dir.glob("*raw*.pkl"))
        
        if not pickle_files:
            print(f"No raw pickle files found in {self.data_dir}")
            # Also check for any pickle files
            all_pickles = list(self.data_dir.glob("*.pkl"))
            if all_pickles:
                print(f"Found pickle files: {[f.name for f in all_pickles]}")
            return []
        
        # Load the most recent file
        latest_file = max(pickle_files, key=os.path.getctime)
        print(f"Loading raw results from: {latest_file}")
        
        try:
            with open(latest_file, 'rb') as f:
                results = pickle.load(f)
                print(f"Successfully loaded {len(results)} experiment results")
                return results
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            print("This might be due to missing class definitions or incompatible pickle format")
            return []
    
    def _load_aggregated_data(self):
        """Load aggregated simulation results from JSON files."""
        json_files = list(self.data_dir.glob("*aggregated*.json"))
        
        if not json_files:
            print("No aggregated JSON files found")
            return {}
        
        # Load the most recent file
        latest_file = max(json_files, key=os.path.getctime)
        print(f"Loading aggregated data from: {latest_file}")
        
        with open(latest_file, 'r') as f:
            return json.load(f)
    
    def _process_to_dataframe(self) -> pd.DataFrame:
        """Convert raw results to DataFrame for easy plotting."""
        if not self.raw_results:
            return pd.DataFrame()
        
        rows = []
        for result in self.raw_results:
            # Basic experiment parameters
            row = {
                'M': result.M,
                'N': result.N,
                'delta': result.delta,
                'physical_error_rate': result.delta,  # For compatibility
                'noise_type': result.noise_type,
                'trial_number': result.trial_number,
                'final_logical_error': result.final_logical_error,
                'final_fidelity': result.final_fidelity,
                'error_reduction_ratio': result.final_logical_error / result.delta if result.delta > 0 else 1.0
            }
            
            # Add metrics by purification level
            if result.metrics_by_level:
                max_level = max(result.metrics_by_level.keys())
                row['max_purification_level'] = max_level
                
                # Store metrics for each level
                for level, metrics in result.metrics_by_level.items():
                    row[f'fidelity_level_{level}'] = metrics['fidelity']
                    row[f'logical_error_level_{level}'] = metrics['logical_error']
            else:
                row['max_purification_level'] = 0
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        print(f"Processed {len(df)} experimental results")
        print(f"M range: {df['M'].min()}-{df['M'].max()}")
        print(f"N range: {df['N'].min()}-{df['N'].max()}")
        print(f"Delta range: {df['delta'].min():.3f}-{df['delta'].max():.3f}")
        
        return df
    
    def plot_final_error_vs_system_size(self, delta_fixed: float = 0.1, N_fixed: Optional[int] = None):
        """Plot final logical error vs system size M - PRIMARY PLOT"""
        
        fig, ax = plt.subplots(figsize=(12, 9))
        
        if self.processed_df.empty:
            print("No data available for plotting")
            return
        
        # Use largest N if not specified
        if N_fixed is None:
            N_fixed = int(self.processed_df['N'].max())
        
        # Filter for fixed N and closest delta
        df_filtered = self.processed_df[
            (self.processed_df['N'] == N_fixed) &
            (abs(self.processed_df['delta'] - delta_fixed) < 0.05)
        ].copy()
        
        if df_filtered.empty:
            print(f"No data for N={N_fixed}, delta≈{delta_fixed}")
            return
        
        # Group by M and calculate mean/std
        M_values = sorted(df_filtered['M'].unique())
        mean_errors = []
        std_errors = []
        
        for M in M_values:
            M_data = df_filtered[df_filtered['M'] == M]
            if not M_data.empty:
                mean_error = M_data['final_logical_error'].mean()
                std_error = M_data['final_logical_error'].std()
                mean_errors.append(mean_error)
                std_errors.append(std_error if not np.isnan(std_error) else 0)
            else:
                mean_errors.append(np.nan)
                std_errors.append(0)
        
        # Remove NaN values
        valid_indices = [i for i, err in enumerate(mean_errors) if not np.isnan(err)]
        valid_M = [M_values[i] for i in valid_indices]
        valid_means = [mean_errors[i] for i in valid_indices]
        valid_stds = [std_errors[i] for i in valid_indices]
        
        if valid_M:
            # Plot with error bars
            ax.errorbar(valid_M, valid_means, yerr=valid_stds, 
                       fmt='o-', linewidth=3, markersize=12, capsize=8,
                       color=PROTOCOL_COLORS['depolarizing'], 
                       label=f'Streaming QEC (N={N_fixed}, δ={delta_fixed})')
            
            # Add theoretical exponential decay fit
            # if len(valid_means) >= 2:
            #     log_errors = np.log(valid_means)
            #     coeffs = np.polyfit(valid_M, log_errors, 1)
                
                # M_theory = np.linspace(min(valid_M), max(valid_M), 100)
                # theory_errors = np.exp(coeffs[1]) * np.exp(coeffs[0] * M_theory)
                
                # ax.plot(M_theory, theory_errors, '--', color='gray', 
                    #    alpha=0.7, linewidth=3, 
                    #    label=f'Exponential fit: $\\varepsilon \\propto e^{{{coeffs[0]:.2f}M}}$')
                
                # Print decay rate
                # print(f"Exponential decay rate: {coeffs[0]:.3f} per qubit")
        
        # Add no-correction line (constant at delta_fixed)
        if valid_M:
            ax.axhline(y=delta_fixed, color=PROTOCOL_COLORS['no_correction'], 
                      linestyle=':', alpha=0.7, linewidth=3, label='No correction')
        
        ax.set_yscale('log')
        ax.set_xlabel(r'System Size $M$ (qubits)', fontsize=25)
        ax.set_ylabel(r'Final Logical Error Rate, $\varepsilon_L$', fontsize=25)
        ax.set_title(f'Final Error vs System Size\n(N={N_fixed}, δ={delta_fixed})', fontsize=30)
        ax.legend(fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=20)
        # ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'final_error_vs_system_size_N{N_fixed}_delta{delta_fixed}.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return valid_M, valid_means, valid_stds
    
    def plot_threshold_curves_system_size(self, N_fixed: Optional[int] = None):
        """Plot threshold curves for different system sizes."""
        
        fig, ax = plt.subplots(figsize=(12, 9))
        
        if self.processed_df.empty:
            return
        
        if N_fixed is None:
            N_fixed = int(self.processed_df['N'].max())
        
        # Filter for fixed N
        df_N = self.processed_df[self.processed_df['N'] == N_fixed].copy()
        
        if df_N.empty:
            print(f"No data for N={N_fixed}")
            return
        
        # Plot for different M values
        M_values = sorted(df_N['M'].unique())[:6]  # Limit for clarity
        colors = plt.cm.viridis(np.linspace(0, 1, len(M_values)))
        
        for i, M in enumerate(M_values):
            M_data = df_N[df_N['M'] == M].copy()
            
            # Group by delta and calculate mean
            delta_means = []
            delta_values = []
            for delta in sorted(M_data['delta'].unique()):
                delta_data = M_data[M_data['delta'] == delta]
                mean_error = delta_data['final_logical_error'].mean()
                delta_values.append(delta)
                delta_means.append(mean_error)
            
            ax.semilogy(delta_values, delta_means, 'o-', 
                       color=colors[i], linewidth=3, markersize=8,
                       label=f'M = {M}')
        
        # Add no-correction line
        delta_range = np.linspace(0.01, 0.99, 100)
        ax.plot(delta_range, delta_range, '--', 
               color=PROTOCOL_COLORS['no_correction'], alpha=0.7, linewidth=3,
               label='No correction')
        
        ax.set_xlabel(r'Physical Error Rate, $\delta$', fontsize=25)
        ax.set_ylabel(r'Final Logical Error Rate, $\varepsilon_L$', fontsize=25)
        ax.set_title(f'Threshold Curves vs System Size\n(N={N_fixed})', fontsize=30)
        ax.legend(fontsize=16, loc='upper left')
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlim(0, 1)
        # ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'threshold_curves_system_size_N{N_fixed}.pdf', 
                   dpi=300, bbox_inches='tight')
        # plt.show()
    
    def plot_error_evolution_curves(self, M_fixed: int = 1, N_fixed: Optional[int] = None, delta_fixed: float = 0.1):
        """Plot error evolution through purification levels."""
        
        fig, ax = plt.subplots(figsize=(12, 9))
        
        if self.processed_df.empty:
            return
        
        if N_fixed is None:
            N_fixed = int(self.processed_df['N'].max())
        
        # Filter for specific M, N, delta
        df_filtered = self.processed_df[
            (self.processed_df['M'] == M_fixed) &
            (self.processed_df['N'] == N_fixed) &
            (abs(self.processed_df['delta'] - delta_fixed) < 0.05)
        ]
        
        if df_filtered.empty:
            print(f"No data for M={M_fixed}, N={N_fixed}, delta≈{delta_fixed}")
            return
        
        # Find maximum purification level
        max_level = 0
        for _, row in df_filtered.iterrows():
            for col in row.index:
                if col.startswith('logical_error_level_'):
                    level = int(col.split('_')[-1])
                    max_level = max(max_level, level)
        
        if max_level == 0:
            print("No purification level data found")
            return
        
        # Calculate mean error at each level
        levels = list(range(max_level + 1))
        mean_errors = []
        std_errors = []
        
        for level in levels:
            col_name = f'logical_error_level_{level}'
            if col_name in df_filtered.columns:
                errors = df_filtered[col_name].dropna()
                if not errors.empty:
                    mean_errors.append(errors.mean())
                    std_errors.append(errors.std() if len(errors) > 1 else 0)
                else:
                    mean_errors.append(np.nan)
                    std_errors.append(0)
            else:
                mean_errors.append(np.nan)
                std_errors.append(0)
        
        # Remove NaN values
        valid_data = [(l, m, s) for l, m, s in zip(levels, mean_errors, std_errors) 
                      if not np.isnan(m)]
        
        if valid_data:
            valid_levels, valid_means, valid_stds = zip(*valid_data)
            
            ax.errorbar(valid_levels, valid_means, yerr=valid_stds,
                       fmt='o-', linewidth=3, markersize=10, capsize=6,
                       color=PROTOCOL_COLORS['depolarizing'],
                       label=f'M={M_fixed}, N={N_fixed}, δ={delta_fixed}')
        
        ax.set_yscale('log')
        ax.set_xlabel(r'Purification Level, $n$', fontsize=25)
        ax.set_ylabel(r'Logical Error Rate, $\varepsilon_L$', fontsize=25)
        ax.set_title('Error Evolution vs Purification Level', fontsize=30)
        ax.legend(fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=20)
        # ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'error_evolution_M{M_fixed}_N{N_fixed}_delta{delta_fixed}.pdf', 
                   dpi=300, bbox_inches='tight')
        # plt.show()
    
    def plot_error_scaling_with_system_size(self, N_fixed: Optional[int] = None):
        """Plot final error vs M for different δ values."""
        
        fig, ax = plt.subplots(figsize=(12, 9))
        
        if self.processed_df.empty:
            return
        
        if N_fixed is None:
            N_fixed = int(self.processed_df['N'].max())
        
        # Select representative delta values
        available_deltas = sorted(self.processed_df['delta'].unique())
        delta_values = []
        for target in [0.01, 0.1, 0.3, 0.5, 0.7]:
            closest = min(available_deltas, key=lambda x: abs(x - target))
            if closest not in delta_values:
                delta_values.append(closest)
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(delta_values)))
        
        for idx, delta in enumerate(delta_values):
            df_delta = self.processed_df[
                (self.processed_df['N'] == N_fixed) &
                (abs(self.processed_df['delta'] - delta) < 0.05)
            ]
            
            if not df_delta.empty:
                M_values = sorted(df_delta['M'].unique())
                mean_errors = []
                
                for M in M_values:
                    M_data = df_delta[df_delta['M'] == M]
                    if not M_data.empty:
                        mean_errors.append(M_data['final_logical_error'].mean())
                
                if mean_errors:
                    ax.semilogy(M_values, mean_errors, 'o-', 
                               linewidth=3, markersize=8,
                               color=colors[idx], label=f'δ = {delta:.2f}')
        
        ax.set_xlabel(r'System Size $M$ (qubits)', fontsize=25)
        ax.set_ylabel(r'Final Logical Error Rate, $\varepsilon_L$', fontsize=25)
        ax.set_title(f'Error Scaling with System Size\n(N={N_fixed})', fontsize=30)
        ax.legend(fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=20)
        # ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'error_scaling_system_size_N{N_fixed}.pdf', 
                   dpi=300, bbox_inches='tight')
        # plt.show()
    
    def plot_error_reduction_vs_system_size(self, N_fixed: Optional[int] = None):
        """Plot error reduction ratios vs system size."""
        
        fig, ax = plt.subplots(figsize=(12, 9))
        
        if self.processed_df.empty:
            return
        
        if N_fixed is None:
            N_fixed = int(self.processed_df['N'].max())
        
        df_N = self.processed_df[self.processed_df['N'] == N_fixed]
        
        # Plot for different M values
        M_values = sorted(df_N['M'].unique())[:6]
        colors = plt.cm.viridis(np.linspace(0, 1, len(M_values)))
        
        for i, M in enumerate(M_values):
            M_data = df_N[df_N['M'] == M].copy()
            
            # Group by delta and calculate mean reduction ratio
            delta_values = []
            reduction_ratios = []
            
            for delta in sorted(M_data['delta'].unique()):
                delta_data = M_data[M_data['delta'] == delta]
                mean_ratio = delta_data['error_reduction_ratio'].mean()
                delta_values.append(delta)
                reduction_ratios.append(mean_ratio)
            
            ax.semilogy(delta_values, reduction_ratios, 'o-',
                       color=colors[i], linewidth=3, markersize=8,
                       label=f'M = {M}')
        
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, linewidth=3,
                   label='No improvement')
        ax.set_xlabel(r'Physical Error Rate, $\delta$', fontsize=25)
        ax.set_ylabel(r'Error Reduction Ratio', fontsize=25)
        ax.set_title(f'Error Reduction vs System Size\n(N={N_fixed})', fontsize=30)
        ax.legend(fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlim(0, 1)
        # ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'error_reduction_vs_system_size_N{N_fixed}.pdf', 
                   dpi=300, bbox_inches='tight')
        # plt.show()
    
    def print_data_summary(self):
        """Print summary of available data."""
        if self.processed_df.empty:
            print("No data available")
            return
        
        print("\n" + "="*60)
        print("THEORETICAL SIMULATION DATA SUMMARY")
        print("="*60)
        
        print(f"Total experiments: {len(self.processed_df)}")
        print(f"System sizes (M): {sorted(self.processed_df['M'].unique())}")
        print(f"Code sizes (N): {sorted(self.processed_df['N'].unique())}")
        print(f"Error rates (δ): {sorted(self.processed_df['delta'].unique())}")
        print(f"Noise types: {sorted(self.processed_df['noise_type'].unique())}")
        
        # Summary statistics
        print(f"\nFinal logical error range: {self.processed_df['final_logical_error'].min():.6f} - {self.processed_df['final_logical_error'].max():.6f}")
        print(f"Final fidelity range: {self.processed_df['final_fidelity'].min():.6f} - {self.processed_df['final_fidelity'].max():.6f}")
        print(f"Error reduction ratio range: {self.processed_df['error_reduction_ratio'].min():.6f} - {self.processed_df['error_reduction_ratio'].max():.6f}")
        
        # System size effect preview
        print(f"\nSystem size effect preview (δ≈0.1):")
        sample_data = self.processed_df[abs(self.processed_df['delta'] - 0.1) < 0.05]
        if not sample_data.empty:
            for M in sorted(sample_data['M'].unique())[:4]:
                M_data = sample_data[sample_data['M'] == M]
                mean_error = M_data['final_logical_error'].mean()
                print(f"  M={M}: mean εL = {mean_error:.6f}")

def main():
    """Run complete theoretical data plotting analysis."""
    
    print("Theoretical Streaming QEC Data Analysis")
    print("="*50)
    
    # Initialize plotter
    plotter = TheoreticalDataPlotter()
    
    # Print data summary
    plotter.print_data_summary()
    
    if plotter.processed_df.empty:
        print("No data found! Make sure simulation results are in the current directory.")
        return
    
    # Generate key plots
    print("\n1. Plotting final error vs system size (PRIMARY PLOT)...")
    plotter.plot_final_error_vs_system_size(delta_fixed=0.1)
    
    print("\n2. Plotting threshold curves by system size...")
    plotter.plot_threshold_curves_system_size()
    
    print("\n3. Plotting error evolution curves...")
    plotter.plot_error_evolution_curves(M_fixed=1, delta_fixed=0.1)
    
    print("\n4. Plotting error scaling with system size...")
    plotter.plot_error_scaling_with_system_size()
    
    print("\n5. Plotting error reduction vs system size...")
    plotter.plot_error_reduction_vs_system_size()
    
    print(f"\nAnalysis complete! Figures saved to: {plotter.figures_dir}")

if __name__ == "__main__":
    main()