"""
Streaming Purification Quantum Error Correction - Data Visualization (Refactored)

This module loads JSON data from comprehensive_protocol_v2.py and creates 
publication-quality figures. Each plot is separated into its own function, produces a
standalone figure, and saves a high-quality PDF into the figures/ directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from typing import Dict, Optional, Tuple, Any

# Set up plotting style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

class StreamingPurificationAnalyzer:
    def __init__(self, data_file: str = "./data/comprehensive_analysis/streaming_purification_analysis.json", 
                 figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        self.data_file = data_file
        self.figsize = figsize
        self.dpi = dpi
        self.colors = sns.color_palette("husl", 10)
        self.data = self._load_data()
        os.makedirs("figures", exist_ok=True)

    def _load_data(self) -> Dict[str, Any]:
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        with open(self.data_file, 'r') as f:
            return json.load(f)

    # =============================
    # Individual Plot Functions
    # =============================

    def plot_memory_scaling_validation(self, save_path: Optional[str] = None):
        memory_data = self.data.get('memory_scaling', {})
        if not memory_data:
            return
        fig, ax = plt.subplots(figsize=self.figsize)
        N_values = sorted([int(n) for n in memory_data.keys()])
        actual_memory = [memory_data[str(n)]['actual_max_memory'] for n in N_values]
        theoretical_memory = [memory_data[str(n)]['theoretical_memory'] for n in N_values]
        ax.loglog(N_values, actual_memory, 'o-', label='Actual Memory')
        ax.loglog(N_values, theoretical_memory, 's--', label='O(log N) Theory')
        ax.set_xlabel('Input States (N)')
        ax.set_ylabel('Memory Requirement')
        ax.set_title('Memory Scaling Validation')
        ax.grid(True, alpha=0.3)
        ax.legend()
        save_path = save_path or "figures/memory_scaling.pdf"
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

    def plot_threshold_analysis(self, save_path: Optional[str] = None):
        threshold_data = self.data.get('depolarizing_thresholds', {})
        if not threshold_data:
            return
        fig, ax = plt.subplots(figsize=self.figsize)
        dimensions = sorted([int(d) for d in threshold_data.keys()])
        thresholds = [threshold_data[str(d)]['threshold_estimate'] for d in dimensions]
        ax.plot(dimensions, thresholds, 'o-', label='Threshold')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random State (50%)')
        ax.axhline(y=0.01, color='orange', linestyle='--', alpha=0.7, label='Surface Code (~1%)')
        ax.set_xlabel('System Dimension (d)')
        ax.set_ylabel('Error Threshold')
        ax.set_title('Threshold vs Dimension')
        ax.grid(True, alpha=0.3)
        ax.legend()
        save_path = save_path or "figures/threshold_analysis.pdf"
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

    def plot_pauli_approximation_comparison(self, save_path: Optional[str] = None):
        pauli_data = self.data.get('pauli_analyses', {})
        if not pauli_data:
            return
        fig, ax = plt.subplots(figsize=self.figsize)
        methods = list(pauli_data.keys())
        dephasing = [pauli_data[m]['pure_dephasing_error_reduction'] for m in methods]
        symmetric = [pauli_data[m]['symmetric_error_reduction'] for m in methods]
        x_pos = np.arange(len(methods))
        width = 0.35
        ax.bar(x_pos - width/2, dephasing, width, label='Dephasing')
        ax.bar(x_pos + width/2, symmetric, width, label='Symmetric')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45, ha='right')
        ax.set_ylabel('Error Reduction')
        ax.set_title('Pauli Approximation Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        save_path = save_path or "figures/pauli_comparison.pdf"
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

    def generate_all_figures(self, output_dir: str = "./figures/") -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.plot_memory_scaling_validation(f"{output_dir}/fig1_memory_scaling.pdf")
        self.plot_threshold_analysis(f"{output_dir}/fig2_threshold_analysis.pdf")
        self.plot_pauli_approximation_comparison(f"{output_dir}/fig3_pauli_comparison.pdf")


def main():
    data_file = "./data/comprehensive_analysis/streaming_purification_analysis.json"
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return
    analyzer = StreamingPurificationAnalyzer(data_file)
    analyzer.generate_all_figures()

if __name__ == "__main__":
    main()
