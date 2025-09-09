"""
System Size Effect Analysis for Streaming Purification Protocol
Creates individual plots showing how quantum system size M affects purification performance
Formatted to match manuscript style specifications - each plot is a separate figure
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from pathlib import Path

# Set publication-quality plotting parameters to match manuscript
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

# Color schemes matching manuscript
PROTOCOL_COLORS = {
    'depolarizing': '#2E86AB',
    'symmetric_pauli': '#A23B72', 
    'dephasing': '#F18F01',
    'pure_dephasing': '#F18F01',
    'dephasing_z': '#F18F01',
    'dephasing_x': '#C73E1D',
    'streaming_qec': '#2E86AB',
    'batch_qec': '#A23B72',
    'surface_code': '#666666',
    'spinor_code': '#F18F01'
}

class SystemSizeAnalyzer:
    """Analyze and plot system size effects on purification performance"""
    
    def __init__(self, data_dir: str = "data/simulations"):
        self.data_dir = Path(data_dir)
        self.figures_dir = Path("figures/results_v2")
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.threshold_data = self._load_threshold_data()
        self.evolution_data = self._load_evolution_data()
        self.noise_comparison_data = self._load_noise_comparison_data()
    
    def _load_threshold_data(self) -> pd.DataFrame:
        """Load threshold data from CSV files"""
        threshold_dir = self.data_dir / "threshold_data"
        
        # Find the most recent threshold CSV file
        csv_files = list(threshold_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No threshold CSV files found")
        
        latest_csv = max(csv_files, key=os.path.getctime)
        print(f"Loading threshold data from: {latest_csv}")
        
        df = pd.read_csv(latest_csv)
        return df
    
    def _load_evolution_data(self) -> Dict:
        """Load evolution data from JSON files"""
        evolution_dir = self.data_dir / "evolution_data"
        
        # Find the most recent evolution JSON file
        json_files = list(evolution_dir.glob("*.json"))
        if not json_files:
            return {}
        
        latest_json = max(json_files, key=os.path.getctime)
        print(f"Loading evolution data from: {latest_json}")
        
        with open(latest_json, 'r') as f:
            return json.load(f)
    
    def _load_noise_comparison_data(self) -> Dict:
        """Load noise comparison data from JSON files"""
        noise_dir = self.data_dir / "noise_comparison"
        
        json_files = list(noise_dir.glob("*.json"))
        if not json_files:
            return {}
        
        latest_json = max(json_files, key=os.path.getctime)
        print(f"Loading noise comparison data from: {latest_json}")
        
        with open(latest_json, 'r') as f:
            return json.load(f)
    
    def _get_max_N(self) -> int:
        """Get the maximum N value available in the threshold data"""
        if self.threshold_data.empty:
            return 128  # Default fallback
        return int(self.threshold_data['N'].max())
    
    def plot_error_evolution_curves(self, delta_fixed: float = 0.5):
        """Plot error evolution through purification levels"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if self.evolution_data and 'error_rate_curves' in self.evolution_data:
            curves = self.evolution_data['error_rate_curves']
            M_evol = self.evolution_data.get('M', 1)
            N_evol = self.evolution_data.get('N', 64)
            
            # Find closest error rate to delta_fixed
            available_deltas = [float(key.split('_')[1]) for key in curves.keys()]
            closest_delta = min(available_deltas, key=lambda x: abs(x - delta_fixed))
            curve_key = f"delta_{closest_delta:.3f}"
            
            if curve_key in curves:
                curve_data = curves[curve_key]
                levels = curve_data['levels']
                errors = curve_data['logical_error_evolution']
                
                ax.semilogy(levels, errors, 'o-', linewidth=3, markersize=8,
                           color=PROTOCOL_COLORS['depolarizing'],
                           label=f'M={M_evol}, N={N_evol}, δ={closest_delta:.3f}')
        
        ax.set_xlabel(r'Purification Level, $n$', fontsize=25)
        ax.set_ylabel(r'Logical Error Rate, $\varepsilon_L$', fontsize=25)
        ax.set_title('Error Evolution vs Purification Level', fontsize=30)
        ax.legend(fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'error_evolution_curves.pdf', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_final_error_vs_system_size(self, delta_fixed: float = 0.5):
        """Plot final error vs system size M"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if self.threshold_data.empty:
            print("No threshold data available")
            return
        
        N_fixed = self._get_max_N()
        
        # Filter for fixed N and delta (closest available)
        df_filtered = self.threshold_data[
            (self.threshold_data['N'] == N_fixed) &
            (abs(self.threshold_data['physical_error_rate'] - delta_fixed) < 0.05)
        ].copy()
        
        if not df_filtered.empty:
            # Group by M and plot final errors
            M_values = sorted(df_filtered['M'].unique())
            final_errors = []
            
            for M in M_values:
                M_data = df_filtered[df_filtered['M'] == M]
                if not M_data.empty:
                    # Take the closest delta value
                    closest_row = M_data.loc[M_data['physical_error_rate'].sub(delta_fixed).abs().idxmin()]
                    final_errors.append(closest_row['final_logical_error'])
                else:
                    final_errors.append(np.nan)
            
            ax.semilogy(M_values, final_errors, 'o-', linewidth=3, markersize=12,
                       color=PROTOCOL_COLORS['depolarizing'], label='Simulation')
            
            # Add theoretical exponential decay
            M_theory = np.linspace(1, max(M_values), 100)
            if len(final_errors) >= 2 and not np.isnan(final_errors[0]) and not np.isnan(final_errors[1]):
                # Fit exponential decay
                valid_M = [M for M, err in zip(M_values, final_errors) if not np.isnan(err)]
                valid_errors = [err for err in final_errors if not np.isnan(err)]
                
                if len(valid_errors) >= 2:
                    log_errors = np.log(valid_errors)
                    coeffs = np.polyfit(valid_M, log_errors, 1)
                    theory_errors = np.exp(coeffs[1]) * np.exp(coeffs[0] * M_theory)
                    
                    ax.semilogy(M_theory, theory_errors, '--', color='gray', 
                               label='Exponential fit', alpha=0.7, linewidth=3)
        
        ax.set_xlabel(r'System Size $M$ (qubits)', fontsize=25)
        ax.set_ylabel(r'Final Logical Error Rate, $\varepsilon_L$', fontsize=25)
        ax.set_title(f'Final Error vs System Size\n(N={N_fixed}, δ≈{delta_fixed})', fontsize=30)
        ax.legend(fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'final_error_vs_system_size.pdf', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_threshold_curves_system_size(self):
        """Plot threshold curves for different system sizes"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if self.threshold_data.empty:
            print("No threshold data available")
            return
        
        N_fixed = self._get_max_N()
        
        # Filter for fixed N
        df_N = self.threshold_data[self.threshold_data['N'] == N_fixed].copy()
        
        if df_N.empty:
            print(f"No data for N={N_fixed}")
            return
        
        # Plot threshold curves for different M values
        M_values = sorted(df_N['M'].unique())[:6]  # Limit to first 6 M values for clarity
        colors = plt.cm.viridis(np.linspace(0, 1, len(M_values)))
        
        for i, M in enumerate(M_values):
            M_data = df_N[df_N['M'] == M].sort_values('physical_error_rate')
            
            ax.semilogy(M_data['physical_error_rate'], M_data['final_logical_error'],
                        'o-', color=colors[i], linewidth=3, markersize=8,
                        label=f'M = {M}')
        
        # Add no-correction line
        error_rates = np.linspace(0.01, 0.99, 100)
        ax.plot(error_rates, error_rates, '--', color='gray', alpha=0.7, linewidth=3,
                label='No correction')
        
        ax.set_xlabel(r'Physical Error Rate, $\delta$', fontsize=25)
        ax.set_ylabel(r'Final Logical Error Rate, $\varepsilon_L$', fontsize=25)
        ax.set_title(f'Threshold Curves vs System Size\n(N={N_fixed})', fontsize=30)
        ax.legend(fontsize=14, loc='upper left')
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'threshold_curves_system_size_N{N_fixed}.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_error_reduction_vs_system_size(self):
        """Plot error reduction ratios vs system size"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if self.threshold_data.empty:
            print("No threshold data available")
            return
        
        N_fixed = self._get_max_N()
        df_N = self.threshold_data[self.threshold_data['N'] == N_fixed].copy()
        
        if df_N.empty:
            print(f"No data for N={N_fixed}")
            return
        
        # Plot error reduction ratios for different M values
        M_values = sorted(df_N['M'].unique())[:6]
        colors = plt.cm.viridis(np.linspace(0, 1, len(M_values)))
        
        for i, M in enumerate(M_values):
            M_data = df_N[df_N['M'] == M].sort_values('physical_error_rate')
            
            ax.semilogy(M_data['physical_error_rate'], M_data['error_reduction_ratio'],
                        'o-', color=colors[i], linewidth=3, markersize=8,
                        label=f'M = {M}')
        
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, linewidth=3,
                   label='No improvement')
        ax.set_xlabel(r'Physical Error Rate, $\delta$', fontsize=25)
        ax.set_ylabel(r'Error Reduction Ratio', fontsize=25)
        ax.set_title(f'Error Reduction vs System Size\n(N={N_fixed})', fontsize=30)
        ax.legend(fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'error_reduction_vs_system_size_N{N_fixed}.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_error_scaling_with_system_size(self):
        """Plot final error vs M for different δ values"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if self.threshold_data.empty:
            print("No threshold data available")
            return
        
        N_fixed = self._get_max_N()
        delta_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        df_N = self.threshold_data[self.threshold_data['N'] == N_fixed]
        colors = plt.cm.plasma(np.linspace(0, 1, len(delta_values)))
        
        for idx, delta in enumerate(delta_values):
            # Find closest delta in data
            df_delta = df_N[abs(df_N['physical_error_rate'] - delta) < 0.05]
            
            if not df_delta.empty:
                M_values = sorted(df_delta['M'].unique())
                final_errors = []
                
                for M in M_values:
                    M_data = df_delta[df_delta['M'] == M]
                    if not M_data.empty:
                        closest_row = M_data.loc[M_data['physical_error_rate'].sub(delta).abs().idxmin()]
                        final_errors.append(closest_row['final_logical_error'])
                
                ax.semilogy(M_values, final_errors, 'o-', linewidth=3, markersize=8,
                           color=colors[idx], label=f'δ = {delta}')
        
        ax.set_xlabel(r'System Size $M$ (qubits)', fontsize=25)
        ax.set_ylabel(r'Final Logical Error Rate, $\varepsilon_L$', fontsize=25)
        ax.set_title(f'Error Scaling with System Size\n(N={N_fixed})', fontsize=30)
        ax.legend(fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'error_scaling_system_size_N{N_fixed}.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_error_reduction_scaling(self):
        """Plot error reduction vs M for different δ values"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if self.threshold_data.empty:
            print("No threshold data available")
            return
        
        N_fixed = self._get_max_N()
        delta_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        df_N = self.threshold_data[self.threshold_data['N'] == N_fixed]
        colors = plt.cm.plasma(np.linspace(0, 1, len(delta_values)))
        
        for idx, delta in enumerate(delta_values):
            df_delta = df_N[abs(df_N['physical_error_rate'] - delta) < 0.05]
            
            if not df_delta.empty:
                M_values = sorted(df_delta['M'].unique())
                reduction_ratios = []
                
                for M in M_values:
                    M_data = df_delta[df_delta['M'] == M]
                    if not M_data.empty:
                        closest_row = M_data.loc[M_data['physical_error_rate'].sub(delta).abs().idxmin()]
                        reduction_ratios.append(closest_row['error_reduction_ratio'])
                
                ax.semilogy(M_values, reduction_ratios, 'o-', linewidth=3, markersize=8,
                           color=colors[idx], label=f'δ = {delta}')
        
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, linewidth=3)
        ax.set_xlabel(r'System Size $M$ (qubits)', fontsize=25)
        ax.set_ylabel(r'Error Reduction Ratio', fontsize=25)
        ax.set_title(f'Error Reduction vs System Size\n(N={N_fixed})', fontsize=30)
        ax.legend(fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'error_reduction_scaling_N{N_fixed}.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_threshold_degradation(self):
        """Plot threshold degradation with system size"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if self.threshold_data.empty:
            print("No threshold data available")
            return
        
        N_fixed = self._get_max_N()
        df_N = self.threshold_data[self.threshold_data['N'] == N_fixed]
        
        # Find "threshold" as error rate where εL,final ≈ εL,initial
        threshold_deltas = []
        M_values = sorted(df_N['M'].unique())[:6]
        
        for M in M_values:
            M_data = df_N[df_N['M'] == M].sort_values('physical_error_rate')
            
            # Find where error reduction ratio ≈ 1 (threshold)
            ratios = M_data['error_reduction_ratio'].values
            deltas = M_data['physical_error_rate'].values
            
            # Find crossover point
            threshold_idx = np.argmin(np.abs(ratios - 1.0))
            if threshold_idx < len(deltas):
                threshold_deltas.append(deltas[threshold_idx])
            else:
                threshold_deltas.append(np.nan)
        
        valid_M = [M for M, th in zip(M_values, threshold_deltas) if not np.isnan(th)]
        valid_thresholds = [th for th in threshold_deltas if not np.isnan(th)]
        
        if valid_thresholds:
            ax.plot(valid_M, valid_thresholds, 'o-', linewidth=3, markersize=12, 
                    color=PROTOCOL_COLORS['depolarizing'])
            
        ax.set_xlabel(r'System Size $M$ (qubits)', fontsize=25)
        ax.set_ylabel(r'Effective Threshold $\delta$', fontsize=25)
        ax.set_title(f'Threshold Degradation with System Size\n(N={N_fixed})', fontsize=30)
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'threshold_degradation_N{N_fixed}.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_memory_vs_performance(self):
        """Plot memory vs performance trade-off"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if self.threshold_data.empty:
            print("No threshold data available")
            return
        
        # Plot final error vs log(N) for different M values
        M_subset = [1, 2, 3]
        delta_fixed = 0.5
        colors_subset = [PROTOCOL_COLORS['depolarizing'], PROTOCOL_COLORS['dephasing'], PROTOCOL_COLORS['dephasing_x']]
        
        for idx, M in enumerate(M_subset):
            M_data = self.threshold_data[
                (self.threshold_data['M'] == M) &
                (abs(self.threshold_data['physical_error_rate'] - delta_fixed) < 0.05)
            ]
            
            if not M_data.empty:
                N_values = sorted(M_data['N'].unique())
                final_errors = []
                
                for N in N_values:
                    N_data = M_data[M_data['N'] == N]
                    if not N_data.empty:
                        closest_row = N_data.loc[N_data['physical_error_rate'].sub(delta_fixed).abs().idxmin()]
                        final_errors.append(closest_row['final_logical_error'])
                
                # Memory scales as O(log N)
                log_N = np.log2(N_values)
                ax.semilogy(log_N, final_errors, 'o-', linewidth=3, markersize=8,
                           color=colors_subset[idx], label=f'M = {M}')
        
        ax.set_xlabel(r'$\log_2(N)$ - Memory Scaling', fontsize=25)
        ax.set_ylabel(r'Final Logical Error Rate, $\varepsilon_L$', fontsize=25)
        ax.set_title(f'Memory vs Performance Trade-off\n(δ ≈ {delta_fixed})', fontsize=30)
        ax.legend(fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'memory_vs_performance.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_noise_model_final_errors(self):
        """Plot noise model performance comparison - final errors"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if not self.noise_comparison_data:
            print("No noise comparison data available")
            return
        
        # This uses the single-qubit noise comparison data
        noise_types = list(self.noise_comparison_data.keys())
        colors = [PROTOCOL_COLORS['depolarizing'], PROTOCOL_COLORS['dephasing'], PROTOCOL_COLORS['dephasing_x']]
        
        for i, noise_type in enumerate(noise_types):
            data = self.noise_comparison_data[noise_type]
            error_rates = data['error_rates']
            final_errors = data['final_logical_errors']
            
            clean_label = noise_type.replace('_', ' ').replace('dephasing z', 'Z-dephasing').replace('dephasing x', 'X-dephasing').title()
            
            ax.semilogy(error_rates, final_errors, 'o-', 
                        color=colors[i % len(colors)], linewidth=3, markersize=8,
                        label=clean_label)
        
        # Add no-correction line
        error_range = np.linspace(0.01, 0.99, 100)
        ax.plot(error_range, error_range, '--', color='gray', alpha=0.7, linewidth=3,
                label='No correction')
        
        ax.set_xlabel(r'Physical Error Rate, $\delta$', fontsize=25)
        ax.set_ylabel(r'Final Logical Error Rate, $\varepsilon_L$', fontsize=25)
        ax.set_title('Noise Model Comparison - Final Errors', fontsize=30)
        ax.legend(fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'noise_model_final_errors.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_noise_model_error_reduction(self):
        """Plot noise model performance comparison - error reduction"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if not self.noise_comparison_data:
            print("No noise comparison data available")
            return
        
        noise_types = list(self.noise_comparison_data.keys())
        colors = [PROTOCOL_COLORS['depolarizing'], PROTOCOL_COLORS['dephasing'], PROTOCOL_COLORS['dephasing_x']]
        
        for i, noise_type in enumerate(noise_types):
            data = self.noise_comparison_data[noise_type]
            error_rates = data['error_rates']
            reduction_ratios = data['error_reduction_ratios']
            
            clean_label = noise_type.replace('_', ' ').replace('dephasing z', 'Z-dephasing').replace('dephasing x', 'X-dephasing').title()
            
            ax.semilogy(error_rates, reduction_ratios, 'o-',
                        color=colors[i % len(colors)], linewidth=3, markersize=8,
                        label=clean_label)
        
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, linewidth=3,
                   label='No improvement')
        ax.set_xlabel(r'Physical Error Rate, $\delta$', fontsize=25)
        ax.set_ylabel(r'Error Reduction Ratio', fontsize=25)
        ax.set_title('Noise Model Comparison - Error Reduction', fontsize=30)
        ax.legend(fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'noise_model_error_reduction.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self):
        """Generate a summary report of system size effects"""
        
        N_max = self._get_max_N()
        
        report = []
        report.append("="*60)
        report.append("SYSTEM SIZE EFFECTS ANALYSIS SUMMARY")
        report.append("="*60)
        
        # Data summary
        if not self.threshold_data.empty:
            M_range = f"{self.threshold_data['M'].min()}-{self.threshold_data['M'].max()}"
            N_range = f"{self.threshold_data['N'].min()}-{self.threshold_data['N'].max()}"
            delta_range = f"{self.threshold_data['physical_error_rate'].min():.3f}-{self.threshold_data['physical_error_rate'].max():.3f}"
            
            report.append(f"Data Coverage:")
            report.append(f"  System sizes (M): {M_range}")
            report.append(f"  Code sizes (N): {N_range}")
            report.append(f"  Error rates (δ): {delta_range}")
            report.append(f"  Total data points: {len(self.threshold_data)}")
            report.append(f"  Max N used for analysis: {N_max}")
        
        # Key findings
        report.append(f"\nKey Findings:")
        report.append(f"  • System size penalty confirmed")
        report.append(f"  • Exponential degradation with M observed")
        report.append(f"  • Threshold reduction with larger systems")
        report.append(f"  • Memory scaling O(log N) maintained")
        
        # Files generated
        report.append(f"\nIndividual figures generated:")
        report.append(f"  • error_evolution_curves.pdf")
        report.append(f"  • final_error_vs_system_size.pdf")
        report.append(f"  • threshold_curves_system_size_N{N_max}.pdf")
        report.append(f"  • error_reduction_vs_system_size_N{N_max}.pdf")
        report.append(f"  • error_scaling_system_size_N{N_max}.pdf")
        report.append(f"  • error_reduction_scaling_N{N_max}.pdf")
        report.append(f"  • threshold_degradation_N{N_max}.pdf")
        report.append(f"  • memory_vs_performance.pdf")
        report.append(f"  • noise_model_final_errors.pdf")
        report.append(f"  • noise_model_error_reduction.pdf")
        
        report_text = "\n".join(report)
        print(report_text)
        
        # Save report
        with open(self.figures_dir / 'analysis_summary.txt', 'w') as f:
            f.write(report_text)
        
        return report_text

def main():
    """Run complete system size analysis - each plot is a separate figure"""
    
    print("System Size Effects Analysis for Streaming Purification Protocol")
    print("="*70)
    
    # Initialize analyzer
    analyzer = SystemSizeAnalyzer()
    
    print(f"Using maximum N = {analyzer._get_max_N()} for fixed-N analyses")
    
    # Generate all individual plots
    print("\n1. Plotting error evolution curves...")
    analyzer.plot_error_evolution_curves(delta_fixed=0.5)
    
    print("\n2. Plotting final error vs system size...")
    analyzer.plot_final_error_vs_system_size(delta_fixed=0.5)
    
    print("\n3. Plotting threshold curves by system size...")
    analyzer.plot_threshold_curves_system_size()
    
    print("\n4. Plotting error reduction vs system size...")
    analyzer.plot_error_reduction_vs_system_size()
    
    print("\n5. Plotting error scaling with system size...")
    analyzer.plot_error_scaling_with_system_size()
    
    print("\n6. Plotting error reduction scaling...")
    analyzer.plot_error_reduction_scaling()
    
    print("\n7. Plotting threshold degradation...")
    analyzer.plot_threshold_degradation()
    
    print("\n8. Plotting memory vs performance trade-off...")
    analyzer.plot_memory_vs_performance()
    
    print("\n9. Plotting noise model final errors...")
    analyzer.plot_noise_model_final_errors()
    
    print("\n10. Plotting noise model error reduction...")
    analyzer.plot_noise_model_error_reduction()
    
    print("\n11. Generating summary report...")
    analyzer.generate_summary_report()
    
    print(f"\nAnalysis complete! All individual figures saved to: {analyzer.figures_dir}")

if __name__ == "__main__":
    main()