"""
Figure generation for iterative purification simulation data.
Loads CSV data from iterative purification simulations and creates publication-quality figures
showing final fidelity vs rounds of iteration for different purification levels (ℓ).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Literal
from pathlib import Path

from dataclasses import dataclass

try:
    from scipy.optimize import curve_fit
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# Publication-quality plotting
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 40,
    'axes.labelsize': 40,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'legend.fontsize': 20,
    'figure.titlesize': 40,
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'dejavusans',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 3,
    'lines.markersize': 12
})

COLORS = {
    'depolarizing': '#2E86AB',
    'dephasing': '#F18F01',
}

# Chosen for clear distinctness in print; extend if you often have many series.
MARKERS = ['o', 's', '^', 'D', 'P', 'X', 'v', '>', '<', 'h', '*']

def _mk(i: int) -> str:
    """Return a distinct marker for series index i (wraps automatically)."""
    return MARKERS[i % len(MARKERS)]


def _exp_model(t: np.ndarray, F0: float, gamma: float) -> np.ndarray:
    # F(t)=F0+(1-F0)*exp(-gamma t)
    return F0 + (1.0 - F0) * np.exp(-gamma * t)


@dataclass(frozen=True)
class GammaEstimate:
    gamma: float
    F0: float
    method: str


def _compress_to_final_per_iteration(df_steps: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the final state per (run_id, iteration) by selecting max depth,
    breaking ties by max merge_num.
    """
    needed = {'run_id', 'iteration', 'fidelity'}
    if not needed.issubset(df_steps.columns):
        raise ValueError(f"steps df missing columns: {needed - set(df_steps.columns)}")

    sort_cols = ['run_id', 'iteration']
    by_cols = ['run_id', 'iteration']

    if 'depth' in df_steps.columns:
        sort_cols += ['depth']
    if 'merge_num' in df_steps.columns:
        sort_cols += ['merge_num']

    # Sort ascending then take last row in each group
    df_sorted = df_steps.sort_values(sort_cols)
    df_final = df_sorted.groupby(by_cols, as_index=False).tail(1)

    return df_final


def estimate_gamma_from_trajectory(
    t: np.ndarray,
    F: np.ndarray,
    method: Literal["fit", "slope"] = "fit",
    tail_k_for_F0: int = 3,
    slope_k: int = 4
) -> GammaEstimate:
    """
    Estimate gamma for a single fidelity trajectory F(t).

    - method="fit": nonlinear least squares fit for (F0, gamma)
    - method="slope": gamma ~ -dF/dt|_{t=0} (TB Eq. (29))
    """
    t = np.asarray(t, dtype=float)
    F = np.asarray(F, dtype=float)

    # basic cleaning
    mask = np.isfinite(t) & np.isfinite(F)
    t, F = t[mask], F[mask]
    if len(t) < 3:
        return GammaEstimate(gamma=np.nan, F0=np.nan, method=f"{method}:too_few_points")

    # sort
    order = np.argsort(t)
    t, F = t[order], F[order]

    # Estimate F0 from tail
    k = min(tail_k_for_F0, len(F))
    F0_init = float(np.mean(F[-k:]))

    if method == "fit":
        if not _HAS_SCIPY:
            return GammaEstimate(gamma=np.nan, F0=np.nan, method="fit:no_scipy")

        # Initial gamma guess from first difference (clipped)
        dF = F[1] - F[0]
        dt = t[1] - t[0] if t[1] != t[0] else 1.0
        gamma_init = max(1e-6, float(-dF / max(dt, 1e-9)))

        # Bounds: F0 in [0,1], gamma >= 0
        p0 = [np.clip(F0_init, 0.0, 1.0), gamma_init]
        bounds = ([0.0, 0.0], [1.0, np.inf])

        try:
            popt, _ = curve_fit(_exp_model, t, F, p0=p0, bounds=bounds, maxfev=20000)
            F0_hat, gamma_hat = float(popt[0]), float(popt[1])
            return GammaEstimate(gamma=gamma_hat, F0=F0_hat, method="fit")
        except Exception as e:
            return GammaEstimate(gamma=np.nan, F0=np.nan, method=f"fit:fail:{type(e).__name__}")

    elif method == "slope":
        # Fit a line to first slope_k points: F ≈ a + b t  => gamma ≈ -b
        kk = min(slope_k, len(t))
        tt, FF = t[:kk], F[:kk]
        # robust to t not starting at 1
        A = np.vstack([np.ones_like(tt), tt]).T
        coeff, *_ = np.linalg.lstsq(A, FF, rcond=None)
        slope = float(coeff[1])
        gamma_hat = max(0.0, -slope)
        return GammaEstimate(gamma=gamma_hat, F0=F0_init, method="slope")
    else:
        raise ValueError("method must be 'fit' or 'slope'")


def build_gamma_table_longest(
    df_steps: pd.DataFrame,
    group_cols: list[str],
    method: str = "fit",
) -> pd.DataFrame:
    """
    One gamma per (group_cols) by selecting the run_id with the *longest* trajectory
    (max iteration), then estimating gamma from that trajectory.
    """
    df_final = _compress_to_final_per_iteration(df_steps)

    required = set(group_cols + ["run_id", "iteration", "fidelity"])
    missing = required - set(df_final.columns)
    if missing:
        raise ValueError(f"Missing columns needed for gamma table: {missing}")

    rows = []
    # group by requested grouping (no run_id yet)
    for key, g_all in df_final.groupby(group_cols):
        # pick the run with the most iterations
        run_lengths = g_all.groupby("run_id")["iteration"].max()
        best_run = run_lengths.idxmax()

        g = g_all[g_all["run_id"] == best_run].sort_values("iteration")
        t = g["iteration"].to_numpy()
        F = g["fidelity"].to_numpy()

        est = estimate_gamma_from_trajectory(t, F, method=method)

        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row.update({
            "run_id_used": best_run,
            "n_iters": int(np.max(t)),
            "gamma": est.gamma,
            "F0_fit": est.F0,
            "gamma_method": est.method,
        })
        rows.append(row)

    return pd.DataFrame(rows)



class IterativePurificationPlotter:
    """Generate figures from iterative purification CSV data."""
    
    def __init__(self, data_dir: str = "data/simulations_moreNoise", 
                 figures_dir: str = "figures/iterative_results"):
        self.data_dir = Path(data_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.depol_finals = self._load_csv('finals_circuit_depolarizing.csv')
        self.depol_steps = self._load_csv('steps_circuit_depolarizing.csv')
        self.dephase_finals = self._load_csv('finals_circuit_dephase_z.csv')
        self.dephase_steps = self._load_csv('steps_circuit_dephase_z.csv')
        self.dephase_untwirled_finals = self._load_csv('finals_circuit_dephase_z_untwirled.csv')
        self.dephase_untwirled_steps = self._load_csv('steps_circuit_dephase_z_untwirled.csv')
        
        self.missing_M5_l3_data_depol_finals = self._load_csv('finals_circuit_depolarizing_vM5.csv')
        self.missing_M5_l3_data_depol_steps = self._load_csv('steps_circuit_depolarizing_vM5.csv')
        self.missing_M5_l3_data_dephase_finals = self._load_csv('finals_circuit_dephase_z_vM5.csv')
        self.missing_M5_l3_data_dephase_steps = self._load_csv('steps_circuit_dephase_z_vM5.csv')
        
        self.dephase_theta_phi_finals = self._load_csv('finals_circuit_dephase_z_theta_phi_no_twirl.csv')
        self.dephase_theta_phi_steps = self._load_csv('steps_circuit_dephase_z_theta_phi_no_twirl.csv')
        
        print(f"Loaded iterative purification data:")
        print(f"  Depolarizing finals: {len(self.depol_finals)} runs")
        print(f"  Depolarizing steps: {len(self.depol_steps)} iteration steps")
        print(f"  Dephasing finals: {len(self.dephase_finals)} runs")
        print(f"  Dephasing steps: {len(self.dephase_steps)} iteration steps")
        print(f"  Dephasing untwirled finals: {len(self.dephase_untwirled_finals)} runs")
        print(f"  Dephasing untwirled steps: {len(self.dephase_untwirled_steps)} iteration steps")
    
    def _load_csv(self, filename: str) -> pd.DataFrame:
        """Load CSV file if it exists."""
        filepath = self.data_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
        
            # Normalize twirling column name and convert to boolean
            if 'twirling_applied' in df.columns:
                # Convert string "TRUE"/"FALSE" to boolean and rename column
                df['twirling_enabled'] = df['twirling_applied'].map(
                    lambda x: str(x).upper() == 'TRUE' if pd.notna(x) else False
                )
            elif 'twirling_enabled' in df.columns and df['twirling_enabled'].dtype == 'object':
                # Convert string to boolean if needed
                df['twirling_enabled'] = df['twirling_enabled'].map(
                    lambda x: str(x).upper() == 'TRUE' if pd.notna(x) else x
                )
        
            return df
        else:
            print(f"Warning: {filename} not found")
            return pd.DataFrame()
    
    def plot_fidelity_vs_iterations(self, noise_type: str, save_format: str = 'pdf') -> Optional[str]:
        """
        Plot final fidelity vs rounds of iteration for different ℓ values (M=1).
        """
        # Select data and set twirling filter
        if noise_type == 'depolarizing':
            df = self.depol_steps
            color = COLORS['depolarizing']
            twirling_filter = False
        else:  # dephasing
            df = self.dephase_steps
            color = COLORS['dephasing']
            twirling_filter = True
    
        if df.empty:
            print(f"No steps data for {noise_type}")
            return None
    
        # Check for iterative columns
        if 'iteration' not in df.columns or 'purification_level' not in df.columns:
            print(f"Missing iterative columns for {noise_type}")
            print(f"Available columns: {list(df.columns)}")
            return None
    
        # Filter M=1 AND twirling condition
        df_filtered = df[(df['M'] == 1) & (df['twirling_enabled'] == twirling_filter)].copy()
    
        if df_filtered.empty:
            print(f"No M=1 data for {noise_type} with twirling={twirling_filter}")
            return None
    
        fig, ax = plt.subplots(figsize=(10, 8))
    
        # Get unique purification levels (ℓ values)
        l_values = sorted(df_filtered['purification_level'].unique())
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'saddlebrown', 'deeppink', 'darkslategrey', 'fuchsia', 'gold']
    
        # Choose a representative p value (middle of available range)
        p_values = sorted(df_filtered['p'].unique()) if 'p' in df_filtered.columns else []
        if p_values:
            mid_p = p_values[len(p_values) // 2]
            df_p = df_filtered[df_filtered['p'] == mid_p].copy()
        else:
            df_p = df_filtered.copy()
    
        for i, l_val in enumerate(l_values):
            df_l = df_p[df_p['purification_level'] == l_val].copy()
        
            if len(df_l) > 0:
                # Sort by iteration number
                df_l = df_l.sort_values('iteration')
                
                ax.plot(df_l['iteration'], df_l['fidelity'],
                        linestyle='-', marker=_mk(i),
                        color=colors[i % len(colors)], linewidth=3, markersize=8,
                        label=rf'$\ell$ = {l_val}', alpha=0.8)
    
        ax.set_xlabel(r'PQEC Cycles, $t$', fontsize=40)
        ax.set_ylabel(r'Fidelity, $F$', fontsize=40)
    
        title_str = 'Depolarizing' if noise_type == 'depolarizing' else 'Dephasing'
        if p_values:
            ax.set_title(f'Iterative Purification\n({title_str} Noise, M=1, p={mid_p:.2f})', fontsize=40)
        else:
            ax.set_title(f'Iterative Purification\n({title_str} Noise, M=1)', fontsize=40)
    
        ax.legend(fontsize=18, loc='best', frameon=False)
        ax.set_ylim(0.0, 1.05)
        
        if not df_filtered.empty:
            max_iter = df_filtered['iteration'].max()
            ax.set_xlim(0.5, max_iter + 0.5)
    
        plt.tight_layout()
    
        filename = f"fidelity_vs_iterations_{noise_type}_M1.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
        print(f"Saved {filename}")
        return str(filepath)


    # def plot_fidelity_vs_iterations_multiple_p(self, noise_type: str, save_format: str = 'pdf') -> Optional[str]:
    #     """
    #     Plot fidelity vs iterations for different ℓ values, with 2x2 subplots for different p values.
    #     """
    #     # Select data and set twirling filter
    #     if noise_type == 'depolarizing':
    #         df = self.depol_steps
    #         color = COLORS['depolarizing']
    #         twirling_filter = False
    #     else:  # dephasing
    #         df = self.dephase_steps
    #         color = COLORS['dephasing']
    #         twirling_filter = True
    
    #     if df.empty:
    #         print(f"No steps data for {noise_type}")
    #         return None
    
    #     # Check for iterative columns
    #     if 'iteration' not in df.columns or 'purification_level' not in df.columns:
    #         print(f"Missing iterative columns for {noise_type}")
    #         return None
    
    #     # Filter M=1 AND twirling condition
    #     df_filtered = df[(df['M'] == 5) & (df['twirling_enabled'] == twirling_filter)].copy()
    
    #     if df_filtered.empty:
    #         print(f"No M=1 data for {noise_type} with twirling={twirling_filter}")
    #         return None
    
    #     # Get unique values
    #     l_values = sorted(df_filtered['purification_level'].unique())
    #     p_values = sorted(df_filtered['p'].unique()) if 'p' in df_filtered.columns else []
        
    #     if len(p_values) == 0:
    #         print(f"No p values found for {noise_type}")
    #         return None
        
    #     # Select up to 4 p values for 2x2 grid
    #     # if len(p_values) > 4:
    #     #     # Take evenly spaced p values
    #     #     indices = np.linspace(0, len(p_values)-1, 4).astype(int)
    #     #     p_subset = [p_values[i] for i in indices]
    #     # else:
    #     #     p_subset = p_values
    #     p_subset = [0.1, 0.5, 0.7, 0.8]
    
    #     # Create 2x2 subplot grid
    #     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    #     axes = axes.flatten()
    
    #     colors = ['red', 'green', 'blue', 'orange', 'purple', 'saddlebrown', 'deeppink', 'darkslategrey', 'fuchsia', 'gold']
    
    #     # Plot each p value (up to 4)
    #     for i, p_val in enumerate(p_subset[:4]):
    #         ax = axes[i]
    #         df_p = df_filtered[df_filtered['p'] == p_val].copy()
            
    #         if df_p.empty:
    #             ax.text(0.5, 0.5, f'No data\nfor p = {p_val:.2f}',
    #                    horizontalalignment='center', verticalalignment='center',
    #                    transform=ax.transAxes, fontsize=14, alpha=0.7)
    #             continue
            
    #         # Plot curves for different ℓ values
    #         for j, l_val in enumerate(l_values):
    #             df_l = df_p[df_p['purification_level'] == l_val].copy()
                
    #             if df_l.empty:
    #                 continue
                
    #             # Sort by iteration
    #             df_l = df_l.sort_values('iteration')
                
    #             # --- prepend perfect initial point (0, 1) ---
    #             x = df_l['iteration'].to_numpy()
    #             y = df_l['fidelity'].to_numpy()

    #             if len(x) == 0:
    #                 continue

    #             # Only prepend if we don't already have iteration 0
    #             if x[0] != 0:
    #                 x = np.insert(x, 0, 0)
    #                 y = np.insert(y, 0, 1.0)
    #             # -------------------------------------------
                
    #             # Plot trajectory
    #             if l_val==0:
    #                 ax.plot(x, y,
    #                     linestyle='dotted', marker=_mk(j), color=colors[j % len(colors)],
    #                     linewidth=2, markersize=12, alpha=0.8,
    #                     label=rf'No Correction')
    #             else:
    #                 ax.plot(x, y,
    #                     linestyle='-', marker=_mk(j), color=colors[j % len(colors)],
    #                     linewidth=2, markersize=12, alpha=0.8,
    #                     label=rf'$\ell$ = {l_val}')
            
    #         # Formatting for this subplot
    #         ax.set_title(f'p = {p_val:.2f}', fontsize=30)
    #         ax.set_ylim(0.005, 1.05)
            
    #         if not df_p.empty:
    #             max_iter = df_p['iteration'].max()
    #             ax.set_xlim(0.0, max_iter + 0.5)
    #             ax.set_xticks(range(0, int(max_iter) + 1, 2))
            
    #         # Labels
    #         if i >= 2:  # Bottom row
    #             ax.set_xlabel(r'PQEC Cycles, $t$', fontsize=40)
    #         if i % 2 == 0:  # Left column
    #             ax.set_ylabel(r'Fidelity, $F$', fontsize=40)
            
    #         ax.set_yscale('log')
            
    #         # Add legend to first subplot
    #         if i == 1 and len(l_values) > 0:
    #             ax.legend(fontsize=18, loc='best', frameon=False)
                
    #         # Add subplot label 
    #         subplot_labels = ['a', 'b', 'c', 'd']
    #         ax.text(0.98, 0.14, subplot_labels[i], transform=ax.transAxes, fontsize=28, 
    #                 fontweight='bold', fontfamily='sans-serif', va='top', ha='right')
        
    #     # Hide unused subplots
    #     for i in range(len(p_subset), 4):
    #         axes[i].axis('off')
        
    #     # Main title
    #     title_str = 'Depolarizing' if noise_type == 'depolarizing' else 'Dephasing'
    #     # fig.suptitle(f'Iterative Purification - {title_str} Noise (M=1)', fontsize=32, y=0.96)
        
    #     plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        
    #     filename = f"fidelity_vs_iterations_{noise_type}_multi_p_M5.{save_format}"
    #     filepath = self.figures_dir / filename
    #     plt.savefig(filepath, dpi=300, bbox_inches='tight')
    #     plt.close()
        
    #     print(f"Saved {filename}")
    #     return str(filepath)

    def plot_fidelity_vs_iterations_multiple_p(self, noise_type: str, save_format: str = 'pdf') -> Optional[str]:
        """
        Plot fidelity vs iterations for different ℓ values, with 1x2 subplots for p = 0.1 and p = 0.3.
        """
        # Select data and set twirling filter
        if noise_type == 'depolarizing':
            df = self.depol_steps
            color = COLORS['depolarizing']
            twirling_filter = False
        else:  # dephasing
            df = self.dephase_steps
            color = COLORS['dephasing']
            twirling_filter = True
    
        if df.empty:
            print(f"No steps data for {noise_type}")
            return None
    
        # Check for iterative columns
        if 'iteration' not in df.columns or 'purification_level' not in df.columns:
            print(f"Missing iterative columns for {noise_type}")
            return None
    
        # Filter M=5 AND twirling condition
        df_filtered = df[(df['M'] == 5) & (df['twirling_enabled'] == twirling_filter)].copy()
    
        if df_filtered.empty:
            print(f"No M=5 data for {noise_type} with twirling={twirling_filter}")
            return None
    
        # Get unique values
        l_values = sorted(df_filtered['purification_level'].unique())
        p_values = sorted(df_filtered['p'].unique()) if 'p' in df_filtered.columns else []
        
        if len(p_values) == 0:
            print(f"No p values found for {noise_type}")
            return None
        
        # Use only p = 0.1 and p = 0.3 for 1x2 grid
        p_subset = [0.1, 0.3]
    
        # Create 1x2 subplot grid
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'saddlebrown', 'deeppink', 'darkslategrey', 'fuchsia', 'gold']
    
        # Plot each p value
        for i, p_val in enumerate(p_subset):
            ax = axes[i]
            df_p = df_filtered[df_filtered['p'] == p_val].copy()
            
            if df_p.empty:
                ax.text(0.5, 0.5, f'No data\nfor p = {p_val:.1f}',
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14, alpha=0.7)
                continue
            
            # Plot curves for different ℓ values
            for j, l_val in enumerate(l_values):
                df_l = df_p[df_p['purification_level'] == l_val].copy()
                
                # # === HOT FIX: ADD MISSING ℓ=3 DATA ===
                # if l_val == 3 and len(df_l) > 0 and df_l['iteration'].max() <= 5:
                #     print(f"Hot fix: Adding missing ℓ=3 data for p={p_val:.2f}")
                    
                #     missing_points = {}
                #     if abs(p_val - 0.1) <= 0.01:  # p ≈ 0.1
                #         missing_points = {6: 0.876659901, 7: 0.876659901, 
                #                         8: 0.876659901, 9: 0.876659901, 
                #                         10: 0.876659901}
                #     elif abs(p_val - 0.3) <= 0.01:  # p ≈ 0.3
                #         missing_points = {6: 0.05054126978, 7: 0.05054126978,
                #                         8: 0.05054126978, 9: 0.05054126978,
                #                         10: 0.05054126978}
                    
                #     if missing_points:
                #         new_rows = []
                #         for iteration, fidelity in missing_points.items():
                #             new_row = df_l.iloc[0].copy()
                #             new_row['iteration'] = iteration
                #             new_row['fidelity'] = fidelity
                #             new_rows.append(new_row)
                #         df_missing = pd.DataFrame(new_rows)
                #         df_l = pd.concat([df_l, df_missing], ignore_index=True)
                #         print(f"  Added iterations: {list(missing_points.keys())}")
                # # === END HOT FIX ===
                
                
                # === CSV HOT FIX: ADD MISSING ℓ=3 DATA ===
                if l_val == 3 and (len(df_l) == 0 or df_l['iteration'].max() <= 5):
                    # Select appropriate missing data CSV
                    if noise_type == 'depolarizing':
                        missing_data = self.missing_M5_l3_data_depol_steps
                        twirling_target = False
                    else:
                        missing_data = self.missing_M5_l3_data_dephase_steps
                        twirling_target = True
                    
                    if not missing_data.empty:
                        # Filter for our conditions
                        missing_filtered = missing_data[
                            (missing_data['M'] == 5) &
                            (missing_data['purification_level'] == 3) &
                            (abs(missing_data['p'] - p_val) <= 0.001) &
                            (missing_data['twirling_enabled'] == twirling_target)
                        ]
                        
                        if not missing_filtered.empty:
                            # Add missing data
                            df_l = pd.concat([df_l, missing_filtered], ignore_index=True)
                            print(f"CSV Hot fix: Added {len(missing_filtered)} ℓ=3 points for p={p_val:.2f}")
                # === END CSV HOT FIX ===
                
                if df_l.empty:
                    continue
                
                # Sort by iteration
                df_l = df_l.sort_values('iteration')
                
                # --- prepend perfect initial point (0, 1) ---
                x = df_l['iteration'].to_numpy()
                y = df_l['fidelity'].to_numpy()

                if len(x) == 0:
                    continue

                # Only prepend if we don't already have iteration 0
                if x[0] != 0:
                    x = np.insert(x, 0, 0)
                    y = np.insert(y, 0, 1.0)
                # -------------------------------------------
                
                # Plot trajectory
                if l_val==0:
                    ax.plot(x, y,
                        linestyle='dotted', marker=_mk(j), color=colors[j % len(colors)],
                        linewidth=2, markersize=12, alpha=0.8,
                        label=rf'No Correction')
                else:
                    ax.plot(x, y,
                        linestyle='-', marker=_mk(j), color=colors[j % len(colors)],
                        linewidth=2, markersize=12, alpha=0.8,
                        label=rf'$\ell$ = {l_val}')
            
            # Formatting for this subplot
            ax.set_title(f'p = {p_val:.1f}', fontsize=30)
            ax.set_ylim(0.005, 1.05)
            
            if not df_p.empty:
                max_iter = df_p['iteration'].max()
                ax.set_xlim(0.0, max_iter + 0.5)
                ax.set_xticks(range(0, int(max_iter) + 1, 2))
            
            # Labels - both plots get x-label, only left gets y-label
            ax.set_xlabel(r'PQEC Cycles, $t$', fontsize=40)
            if i == 0:  # Left subplot only
                ax.set_ylabel(r'Fidelity, $F$', fontsize=40)
            
            ax.set_yscale('log')
            
            # Add legend to right subplot
            if i == 1 and len(l_values) > 0:
                ax.legend(fontsize=18, loc='best', frameon=False)
                
            # Add subplot label 
            subplot_labels = ['a', 'b']
            ax.text(0.98, 0.14, subplot_labels[i], transform=ax.transAxes, fontsize=28, 
                    fontweight='bold', fontfamily='sans-serif', va='top', ha='right')
        
        # Main title (commented out)
        title_str = 'Depolarizing' if noise_type == 'depolarizing' else 'Dephasing'
        # fig.suptitle(f'Iterative Purification - {title_str} Noise (M=5)', fontsize=32, y=0.96)
        
        plt.tight_layout()
        
        filename = f"fidelity_vs_iterations_{noise_type}_multi_p_M5.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {filename}")
        return str(filepath)



    def add_missing_ell3_data_extended(self, df_l, l_val, p_val, noise_type):
        """
        Hot fix to add missing ℓ=3 data from your extended vM5 files.
        Now includes p=0.1 and p=0.3 data for iterations 6-10.
        """
        import pandas as pd
        import numpy as np
        
        # Only apply fix for ℓ=3
        if l_val != 3:
            return df_l
        
        # Check if we need the fix (missing high iterations)
        if len(df_l) > 0:
            max_iteration = df_l['iteration'].max()
            if max_iteration >= 6:
                # Already has high iteration data
                return df_l
        else:
            # No data at all for this ℓ, definitely need missing data
            max_iteration = 0
        
        print(f"Hot fix: ℓ=3 data stops at t={max_iteration}, adding missing data for p={p_val:.2f}")
        
        # Select the appropriate missing data CSV
        if noise_type == 'depolarizing':
            missing_data_df = self.missing_M5_l3_data_depol_steps
            twirling_filter = False
            missing_finals = self.missing_M5_l3_data_depol_finals
        else:  # dephasing
            missing_data_df = self.missing_M5_l3_data_dephase_steps  
            twirling_filter = True
            missing_finals = self.missing_M5_l3_data_dephase_finals
        
        if missing_data_df.empty:
            print(f"  ❌ No missing {noise_type} steps data loaded")
            return df_l
        
        print(f"  📊 Missing data file has {len(missing_data_df)} rows")
        
        # Handle column name variations
        twirling_col = None
        if 'twirling_applied' in missing_data_df.columns:
            twirling_col = 'twirling_applied'
        elif 'twirling_enabled' in missing_data_df.columns:
            twirling_col = 'twirling_enabled'
        
        # Build filter conditions
        conditions = [
            missing_data_df['M'] == 5,
            missing_data_df['purification_level'] == l_val,
            abs(missing_data_df['p'] - p_val) <= 0.02  # Slightly relaxed tolerance
        ]
        
        if twirling_col:
            conditions.append(missing_data_df[twirling_col] == twirling_filter)
            print(f"  🔧 Using {twirling_col}={twirling_filter} filter")
        
        # Apply all filters
        mask = conditions[0]
        for condition in conditions[1:]:
            mask = mask & condition
        
        missing_filtered = missing_data_df[mask].copy()
        
        if missing_filtered.empty:
            # Try broader p-value search
            broader_p_filter = abs(missing_data_df['p'] - p_val) <= 0.05
            broader_conditions = [missing_data_df['M'] == 5, 
                                missing_data_df['purification_level'] == l_val, 
                                broader_p_filter]
            if twirling_col:
                broader_conditions.append(missing_data_df[twirling_col] == twirling_filter)
            
            broader_mask = broader_conditions[0]
            for condition in broader_conditions[1:]:
                broader_mask = broader_mask & condition
            missing_filtered = missing_data_df[broader_mask].copy()
            
            print(f"  🔍 Expanded p-value search (±0.05): {len(missing_filtered)} rows")
        
        if missing_filtered.empty:
            print(f"  ❌ No matching missing data found for:")
            print(f"      M=5, ℓ={l_val}, p={p_val:.2f}, {noise_type}")
            if 'p' in missing_data_df.columns:
                available_p = sorted(missing_data_df['p'].unique())
                print(f"      Available p values in missing data: {available_p}")
            return df_l
        
        print(f"  ✅ Found {len(missing_filtered)} matching missing data rows")
        
        # Check iterations in missing data
        if 'iteration' in missing_filtered.columns:
            missing_iterations = sorted(missing_filtered['iteration'].unique())
            print(f"  📅 Missing data iterations: {missing_iterations}")
        
        # Get only the high iterations we don't already have
        existing_iterations = set(df_l['iteration'].values) if len(df_l) > 0 else set()
        missing_high_iters = missing_filtered[
            (missing_filtered['iteration'] > max_iteration) &
            (~missing_filtered['iteration'].isin(existing_iterations))
        ].copy()
        
        if len(missing_high_iters) == 0:
            print(f"  ⚠️  No new high iterations to add")
            return df_l
        
        # Handle column name alignment before concatenation
        if len(df_l) > 0:
            # Make sure both dataframes have compatible columns
            common_columns = set(df_l.columns) & set(missing_high_iters.columns)
            if len(common_columns) < 10:  # Sanity check
                print(f"  ⚠️  Column mismatch detected")
                print(f"      Original columns: {list(df_l.columns)}")
                print(f"      Missing columns: {list(missing_high_iters.columns)}")
            
            # Align missing data columns to match original data
            missing_high_iters = missing_high_iters[list(df_l.columns)]
        
        # Combine existing and missing data
        df_combined = pd.concat([df_l, missing_high_iters], ignore_index=True)
        
        added_iterations = sorted(missing_high_iters['iteration'].unique())
        print(f"  ✅ Added {len(missing_high_iters)} missing ℓ=3 points")
        print(f"      New iterations: {added_iterations}")
        print(f"      Total iterations now: {sorted(df_combined['iteration'].unique())}")
        
        return df_combined.sort_values('iteration')


    def plot_fidelity_vs_iterations_with_extended_hotfix(self, noise_type: str, save_format: str = 'pdf'):
        """
        Plotting function with extended hot fix for p=0.1, 0.3 missing data.
        """
        # Select data and set twirling filter
        if noise_type == 'depolarizing':
            df = self.depol_steps
            twirling_filter = False
        else:  # dephasing
            df = self.dephase_steps
            twirling_filter = True

        if df.empty:
            print(f"No steps data for {noise_type}")
            return None

        # Check for iterative columns
        if 'iteration' not in df.columns or 'purification_level' not in df.columns:
            print(f"Missing iterative columns for {noise_type}")
            return None

        # Use correct twirling column name (from our earlier analysis)
        twirling_col = 'twirling_applied' if 'twirling_applied' in df.columns else 'twirling_enabled'
        
        if twirling_col not in df.columns:
            print(f"No twirling column found in {noise_type} data")
            return None
        
        # Filter M=5 AND twirling condition
        df_filtered = df[(df['M'] == 5) & (df[twirling_col] == twirling_filter)].copy()
        
        print(f"After M=5 and {twirling_col}={twirling_filter}: {len(df_filtered)} rows")

        if df_filtered.empty:
            print(f"No M=5 data for {noise_type} with {twirling_col}={twirling_filter}")
            return None

        # Get unique values
        l_values = sorted(df_filtered['purification_level'].unique())
        p_values = sorted(df_filtered['p'].unique()) if 'p' in df_filtered.columns else []
        
        print(f"Available ℓ values: {l_values}")
        print(f"Available p values sample: {p_values[:10]}...")
        
        if len(p_values) == 0:
            print(f"No p values found for {noise_type}")
            return None
        
        # Use p = 0.1 and p = 0.3 for 1x2 grid (now supported by extended missing data)
        p_subset = [0.1, 0.3]

        # Create 1x2 subplot grid
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'saddlebrown', 'deeppink', 'darkslategrey', 'fuchsia', 'gold']

        # Plot each p value
        for i, p_val in enumerate(p_subset):
            ax = axes[i]
            
            # Find closest p-value in data (with tolerance)
            available_p = df_filtered['p'].unique()
            closest_p = min(available_p, key=lambda x: abs(x - p_val))
            
            if abs(closest_p - p_val) > 0.02:
                print(f"Warning: No close p-value to {p_val}, closest is {closest_p}")
                
            df_p = df_filtered[abs(df_filtered['p'] - closest_p) <= 0.02].copy()
            print(f"p={closest_p}: {len(df_p)} rows after p-filter")
            
            if df_p.empty:
                ax.text(0.5, 0.5, f'No data\\nfor p = {p_val:.1f}',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14, alpha=0.7)
                continue
            
            # Plot curves for different ℓ values
            for j, l_val in enumerate(l_values):
                df_l = df_p[df_p['purification_level'] == l_val].copy()
                
                # **EXTENDED HOT FIX: Add missing ℓ=3 data from extended vM5 files**
                df_l = self.add_missing_ell3_data_extended(df_l, l_val, closest_p, noise_type)
                
                if df_l.empty:
                    continue
                
                # Sort by iteration
                df_l = df_l.sort_values('iteration')
                
                # Get arrays
                x = df_l['iteration'].to_numpy()
                y = df_l['fidelity'].to_numpy()

                if len(x) == 0:
                    continue

                # Only prepend if we don't already have iteration 0
                if x[0] != 0:
                    x = np.insert(x, 0, 0)
                    y = np.insert(y, 0, 1.0)
                
                print(f"  Plotting ℓ={l_val}: {len(x)} points, t=0 to {x.max()}")
                
                # Plot trajectory
                if l_val==0:
                    ax.plot(x, y,
                        linestyle='dotted', marker=_mk(j), color=colors[j % len(colors)],
                        linewidth=2, markersize=12, alpha=0.8,
                        label=rf'No Correction')
                else:
                    ax.plot(x, y,
                        linestyle='-', marker=_mk(j), color=colors[j % len(colors)],
                        linewidth=2, markersize=12, alpha=0.8,
                        label=rf'$\ell$ = {l_val}')
            
            # Formatting for this subplot
            ax.set_title(f'p = {closest_p:.1f}', fontsize=30)
            ax.set_ylim(0.005, 1.05)
            
            # Set x-axis to go to at least 10
            max_iter_plot = max(10, df_p['iteration'].max()) if 'iteration' in df_p.columns else 10
            ax.set_xlim(0.0, max_iter_plot + 0.5)
            ax.set_xticks(range(0, int(max_iter_plot) + 1, 2))
            
            # Labels
            ax.set_xlabel(r'PQEC Cycles, $t$', fontsize=40)
            if i == 0:  # Left subplot only
                ax.set_ylabel(r'Fidelity, $F$', fontsize=40)
            
            ax.set_yscale('log')
            
            # Add legend to right subplot
            if i == 1 and len(l_values) > 0:
                ax.legend(fontsize=18, loc='best', frameon=False)
                
            # Add subplot label 
            subplot_labels = ['a', 'b']
            ax.text(0.98, 0.14, subplot_labels[i], transform=ax.transAxes, fontsize=28, 
                    fontweight='bold', fontfamily='sans-serif', va='top', ha='right')
        
        plt.tight_layout()
        
        filename = f"fidelity_vs_iterations_{noise_type}_multi_p_M5_EXTENDED_HOTFIX.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {filename}")
        return str(filepath)







































    def plot_fidelity_vs_iterations_selected_p(self, noise_type: str, save_format: str = 'pdf') -> Optional[str]:
        """
        Plot fidelity vs iterations with multiple p value curves on the same plot.
        Uses p = 0.1, 0.5, 0.7 and shows different ℓ values as separate plots.
        """
        # Select data and set twirling filter
        if noise_type == 'depolarizing':
            df = self.depol_steps
            color = COLORS['depolarizing']
            twirling_filter = False
        else:  # dephasing
            df = self.dephase_steps
            color = COLORS['dephasing']
            twirling_filter = True
    
        if df.empty:
            print(f"No steps data for {noise_type}")
            return None
    
        # Check for iterative columns
        if 'iteration' not in df.columns or 'purification_level' not in df.columns:
            print(f"Missing iterative columns for {noise_type}")
            return None
    
        # Filter M=1 AND twirling condition
        df_filtered = df[(df['M'] == 1) & (df['twirling_enabled'] == twirling_filter)].copy()
    
        if df_filtered.empty:
            print(f"No M=1 data for {noise_type} with twirling={twirling_filter}")
            return None
    
        # Selected p values to plot
        target_p_values = [0.1, 0.5, 0.7]
        available_p_values = sorted(df_filtered['p'].unique()) if 'p' in df_filtered.columns else []
        
        # Find closest available p values to targets
        selected_p_values = []
        for target_p in target_p_values:
            if available_p_values:
                closest_p = min(available_p_values, key=lambda x: abs(x - target_p))
                if abs(closest_p - target_p) < 0.05:  # Within 5% tolerance
                    selected_p_values.append(closest_p)
        
        if len(selected_p_values) == 0:
            print(f"No p values close to targets {target_p_values} for {noise_type}")
            return None
        
        # Get unique ℓ values
        l_values = sorted(df_filtered['purification_level'].unique())
        
        if len(l_values) == 0:
            print(f"No purification levels found for {noise_type}")
            return None
        
        # Create subplots for each ℓ value
        if len(l_values) <= 2:
            fig, axes = plt.subplots(1, len(l_values), figsize=(6*len(l_values), 8))
            if len(l_values) == 1:
                axes = [axes]
        else:
            rows = int(np.ceil(len(l_values) / 2))
            fig, axes = plt.subplots(rows, 2, figsize=(12, 6*rows))
            axes = axes.flatten()
        
        # Colors for different p values
        p_colors = ['#2E86AB', '#F18F01', '#A23B72']  # Blue, orange, purple
        
        # Plot each ℓ value
        for i, l_val in enumerate(l_values):
            ax = axes[i]
            df_l = df_filtered[df_filtered['purification_level'] == l_val].copy()
            
            if df_l.empty:
                ax.text(0.5, 0.5, f'No data\nfor ℓ = {l_val}',
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14, alpha=0.7)
                continue
            
            # Plot trajectories for different p values
            for j, p_val in enumerate(selected_p_values):
                df_p = df_l[df_l['p'] == p_val].copy()
                
                if df_p.empty:
                    continue
                
                # Sort by iteration
                df_p = df_p.sort_values('iteration')
                
                # Plot trajectory
                ax.plot(df_p['iteration'], df_p['fidelity'],
                       linestyle='-', marker=_mk(j), color=p_colors[j % len(p_colors)],
                       linewidth=3, markersize=12, alpha=0.8,
                       label=f'p = {p_val:.1f}')
            
            # Formatting for this subplot
            ax.set_title(rf'$\ell$ = {l_val}', fontsize=30)
            ax.set_ylim(0.0, 1.05)
            
            if not df_l.empty:
                max_iter = df_l['iteration'].max()
                ax.set_xlim(0.5, max_iter + 0.5)
            
            # Labels
            if i >= len(l_values) - 2 or (len(l_values) > 2 and i >= len(l_values) - 2):  # Bottom row
                ax.set_xlabel(r'PQEC Cycles, $t$', fontsize=40)
            if i % 2 == 0 or len(l_values) == 1:  # Left column or single plot
                ax.set_ylabel(r'Fidelity, $F$', fontsize=40)
            
            # Add legend to first subplot
            if i == 0 and len(selected_p_values) > 0:
                ax.legend(fontsize=18, loc='best', frameon=False)
        
        # Hide unused subplots
        for i in range(len(l_values), len(axes)):
            axes[i].axis('off')
        
        # Main title
        title_str = 'Depolarizing' if noise_type == 'depolarizing' else 'Dephasing'
        fig.suptitle(f'Iterative Purification - {title_str} Noise (M=1)\nMultiple Error Rates', fontsize=32, y=0.96)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        
        filename = f"fidelity_vs_iterations_{noise_type}_selected_p_M1.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {filename}")
        return str(filepath)


    def plot_final_fidelity_vs_p(self, noise_type: str, save_format: str = 'pdf') -> Optional[str]:
        """
        Plot fidelity vs physical error rate p for different ℓ values.
        Shows two curves per ℓ: 1st iteration and last iteration.
        """
        # Select data and set twirling filter
        if noise_type == 'depolarizing':
            df = self.depol_steps  # Use steps data to get iteration-level info
            color = COLORS['depolarizing']
            twirling_filter = False
            param_label = r'$p$'
        else:  # dephasing
            df = self.dephase_steps
            color = COLORS['dephasing']
            twirling_filter = True
            param_label = r'$p$'
    
        if df.empty:
            print(f"No steps data for {noise_type}")
            return None
    
        # Check for iterative columns
        if 'iteration' not in df.columns or 'purification_level' not in df.columns:
            print(f"Missing iteration/purification_level columns for {noise_type}")
            return None
    
        # Filter M=1 AND twirling condition
        df_filtered = df[(df['M'] == 1) & (df['twirling_enabled'] == twirling_filter)].copy()
    
        if df_filtered.empty:
            print(f"No M=1 data for {noise_type} with twirling={twirling_filter}")
            return None
    
        fig, ax = plt.subplots(figsize=(10, 8))
    
        # Get unique purification levels
        l_values = sorted(df_filtered['purification_level'].unique())
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'saddlebrown', 'deeppink', 'darkslategrey', 'fuchsia', 'gold']
    
        for i, l_val in enumerate(l_values):
            df_l = df_filtered[df_filtered['purification_level'] == l_val].copy()
        
            if len(df_l) > 0:
                # For each p value, get the 1st and last iteration
                p_values = sorted(df_l['p'].unique())
                
                first_iter_data = []
                last_iter_data = []
                
                for p_val in p_values:
                    df_p = df_l[df_l['p'] == p_val].copy()
                    
                    if df_p.empty:
                        continue
                    
                    # Sort by iteration to get first and last
                    df_p = df_p.sort_values('iteration')
                    
                    # Get first iteration (iteration = 1)
                    first_iter_row = df_p[df_p['iteration'] == 1]
                    if not first_iter_row.empty:
                        first_iter_data.append({
                            'p': p_val,
                            'fidelity': first_iter_row['fidelity'].iloc[0]
                        })
                    
                    # Get last iteration
                    last_iter_row = df_p.iloc[-1]  # Last row after sorting by iteration
                    last_iter_data.append({
                        'p': p_val,
                        'fidelity': last_iter_row['fidelity']
                    })
                
                # Convert to DataFrames for easier plotting
                if first_iter_data:
                    df_first = pd.DataFrame(first_iter_data).sort_values('p')
                    ax.plot(df_first['p'], df_first['fidelity'],
                            linestyle='--', marker=_mk(i), 
                            color=colors[i % len(colors)], linewidth=2, markersize=6,
                            label=rf'$\ell$ = {l_val} (1st iter)', alpha=0.7)
                
                if last_iter_data:
                    df_last = pd.DataFrame(last_iter_data).sort_values('p')
                    ax.plot(df_last['p'], df_last['fidelity'],
                            linestyle='-', marker=_mk(i),
                            color=colors[i % len(colors)], linewidth=3, markersize=8,
                            label=rf'$\ell$ = {l_val} (final iter)', alpha=0.8)
    
        ax.set_xlabel(f'Physical Error Rate, {param_label}', fontsize=40)
        ax.set_ylabel(r'Fidelity, $F$', fontsize=40)
    
        title_str = 'Depolarizing' if noise_type == 'depolarizing' else 'Dephasing'
        ax.set_title(f'Iterative PEC Performance\n({title_str} Noise, M=1)', fontsize=40)
    
        ax.legend(fontsize=18, loc='best', frameon=False)
        ax.set_xlim(0.05, 1.0)
        ax.set_ylim(0.0, 1.05)
    
        plt.tight_layout()
    
        filename = f"final_fidelity_vs_p_{noise_type}_M1.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
        print(f"Saved {filename}")
        return str(filepath)
    
    def plot_fidelity_combined_M(self, save_format: str = 'pdf') -> Optional[str]:
        """
        Combined fidelity plots: 2x2 grid with depolarizing (top) and dephasing (bottom).
        Left column: M=1, Right column: M=5.
        Plots final iteration fidelity vs physical error rate with separate curves for each ℓ value.
        """
        # Check if we have steps data (needed for iteration information)
        if self.depol_steps.empty and self.dephase_steps.empty and self.dephase_untwirled_steps.empty:
            print("No steps data for combined fidelity plots")
            return None

        # Create 2x2 subplot grid
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))

        # Noise type configurations
        # noise_configs = [
        #     {
        #         'type': 'depolarizing',
        #         'df': self.depol_steps,
        #         'twirling_filter': False,
        #         'row_idx': 0,
        #         'row_label': 'Depolarizing Noise'
        #     },
        #     {
        #         'type': 'dephasing', 
        #         'df': self.dephase_steps,
        #         'twirling_filter': True,
        #         'row_idx': 1,
        #         'row_label': 'Dephasing Noise'
        #     }
        # ]
        
        noise_configs = [
            {
                'type': 'depolarizing',
                'df': self.depol_steps,
                'twirling_filter': False,
                'row_idx': 0,
                'row_label': 'Depolarizing Noise',
                'subplot_label': ['a','b']
            },
            {
                'type': 'dephasing_untwirled', 
                'df': self.dephase_untwirled_steps,
                'twirling_filter': False,
                'row_idx': 1,
                'row_label': 'Untwirled Dephasing',
                'subplot_label': ['c','d']
            },
            {
                'type': 'dephasing_twirled', 
                'df': self.dephase_steps,
                'twirling_filter': True,
                'row_idx': 2,
                'row_label': 'Twirled Dephasing',
                'subplot_label': ['e','f']
            }
        ]

        # M values to plot - only M=1 and M=5
        M_values = [1, 5]
        
        # Colors for different ℓ values
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'saddlebrown', 'deeppink', 'darkslategrey', 'fuchsia', 'gold']
        
        for noise_config in noise_configs:
            df = noise_config['df']
            twirling_filter = noise_config['twirling_filter']
            row_idx = noise_config['row_idx']
            noise_type = noise_config['type']
            
            if df.empty:
                # If no data for this noise type, show empty row with message
                for col_idx in range(2):
                    ax = axes[row_idx, col_idx]
                    ax.text(0.5, 0.5, f'No {noise_type}\ndata', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=14, alpha=0.7)
                    if col_idx == 0:
                        ax.set_ylabel(r'Fidelity, $F$', fontsize=35)
                    if row_idx == 1:  # Bottom row
                        # ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=35)
                        ax.set_xlabel('Physical Error Rate, p', fontsize=35)
                continue

            # Check for iterative columns
            if 'iteration' not in df.columns or 'purification_level' not in df.columns:
                print(f"Missing iterative columns for {noise_type}")
                for col_idx in range(2):
                    ax = axes[row_idx, col_idx]
                    ax.text(0.5, 0.5, f'No iterative\ndata for {noise_type}', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=14, alpha=0.7)
                    if col_idx == 0:
                        ax.set_ylabel(r'Fidelity, $F$', fontsize=35)
                    if row_idx == 1:  # Bottom row
                        # ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=35)
                        ax.set_xlabel('Physical Error Rate, p', fontsize=35)
                continue

            # Filter by twirling condition
            if 'twirling_enabled' in df.columns:
                df_filtered = df[df['twirling_enabled'] == twirling_filter].copy()
            elif 'twirling_applied' in df.columns:
                df_filtered = df[df['twirling_applied'] == twirling_filter].copy()
            else:
                df_filtered = df.copy()
            
            if df_filtered.empty:
                # If no data after filtering, show empty row
                for col_idx in range(2):
                    ax = axes[row_idx, col_idx]
                    ax.text(0.5, 0.5, f'No data\n(twirling={twirling_filter})', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=14, alpha=0.7)
                    if col_idx == 0:
                        ax.set_ylabel(r'Fidelity, $F$', fontsize=35)
                    if row_idx == 2:  # Bottom row
                        # ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=35)
                        ax.set_xlabel('Physical Error Rate, p', fontsize=40)
                continue

            # Plot each M value
            for col_idx, M in enumerate(M_values):
                ax = axes[row_idx, col_idx]
                
                # Filter for this M value
                df_M = df_filtered[df_filtered['M'] == M].copy()
                
                # Add subplot label 
                ax.text(0.98, 0.98, noise_config['subplot_label'][col_idx], transform=ax.transAxes, fontsize=28, 
                    fontweight='bold', fontfamily='sans-serif', va='top', ha='right')
                
                if df_M.empty:
                    # If no data for this M, show message
                    ax.text(0.5, 0.5, f'No data\nfor M={M}', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=12, alpha=0.7)
                else:
                    # Get unique ℓ values
                    l_values = sorted(df_M['purification_level'].unique())

                    # Plot separate curve for each ℓ value
                    for i, l_val in enumerate(l_values):
                        df_l = df_M[df_M['purification_level'] == l_val].copy()
                        
                        if df_l.empty:
                            continue
                        
                        # For each (ℓ, p) combination, get the final iteration data
                        p_values = sorted(df_l['p'].unique())
                        final_iter_data = []
                        
                        for p_val in p_values:
                            df_p = df_l[df_l['p'] == p_val].copy()
                            
                            if df_p.empty:
                                continue
                            
                            # Get the final iteration (highest iteration number)
                            max_iter = df_p['iteration'].max()
                            final_iter_row = df_p[df_p['iteration'] == max_iter].iloc[0]
                            
                            final_iter_data.append({
                                'p': p_val,
                                'fidelity': final_iter_row['fidelity']
                            })
                        
                        # Plot this ℓ curve if we have data
                        if final_iter_data:
                            df_plot = pd.DataFrame(final_iter_data).sort_values('p')
                            
                            ax.plot(df_plot['p'], df_plot['fidelity'],
                                linestyle='-', marker=_mk(i),
                                color=colors[i % len(colors)], linewidth=3, markersize=12,
                                label=rf'$\ell$ = {l_val}', alpha=0.8)

                    # Set axis limits
                    ax.set_xlim(0.09, 1.0)
                    ax.set_ylim(0.0, 1.05)
                    ax.set_xscale('linear')
                    ax.set_yscale('linear')
                    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
                    
                    # Add legend to bottom-right subplot
                    if row_idx == 2 and col_idx == 1 and len(l_values) > 0:
                        ax.legend(fontsize=18, loc='lower right', frameon=False)

                # Subplot titles (M values) only on top row
                if row_idx == 0:
                    ax.set_title(f'M = {M}', fontsize=40)
                
                # Y-axis label only on first column
                if col_idx == 0:
                    ax.set_ylabel(r'Fidelity, $F$', fontsize=35)
                
                # X-axis label only on bottom row
                if row_idx == 2:
                    # ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=35)
                    ax.set_xlabel('Physical Error Rate, p', fontsize=35)

        # Add row labels
        # fig.text(0.02, 0.75, 'Depolarizing Noise', rotation=90, fontsize=35, 
        #         verticalalignment='center', weight='bold')
        # fig.text(0.02, 0.30, 'Dephasing Noise', rotation=90, fontsize=35, 
        #         verticalalignment='center', weight='bold')
        
        fig.text(0.02, 0.83, 'Depolarizing Noise', rotation=90, fontsize=25, 
                verticalalignment='center', weight='bold')
        fig.text(0.02, 0.50, ' Untwirled\nDephasing', rotation=90, fontsize=25, 
                verticalalignment='center', weight='bold')
        fig.text(0.02, 0.20, ' Twirled\nDephasing', rotation=90, fontsize=25, 
                verticalalignment='center', weight='bold')

        plt.tight_layout()
        plt.subplots_adjust(left=0.18)  # Make room for row labels

        filename = f"fidelity_combined_M_final_iter.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved {filename}")
        return str(filepath)
    
    
    def plot_gamma_vs_p_combined(
        self,
        method: str = "fit",
        save_format: str = "pdf",
    ) -> Optional[str]:
        """
        Single plot: gamma vs p.
        - Depolarizing: solid
        - Dephase_z (twirled): dashed
        Curves: one per M, using max available ell for that M (per noise family).
        Includes optional 'No QEC' reference (ell=0) if present.
        """
        # --- sanity checks
        for df_name, df in [("depol_steps", self.depol_steps), ("dephase_steps", self.dephase_steps)]:
            if df.empty:
                print(f"No data in {df_name}")
                return None
            for col in ["M", "p", "purification_level", "iteration", "fidelity", "run_id"]:
                if col not in df.columns:
                    print(f"Missing '{col}' in {df_name}")
                    return None

        # --- filter twirling meaningfully
        depol = self.depol_steps.copy()
        deph  = self.dephase_steps.copy()

        if "twirling_enabled" in depol.columns:
            depol = depol[depol["twirling_enabled"] == False].copy()

        if "twirling_enabled" in deph.columns:
            deph = deph[deph["twirling_enabled"] == True].copy()

        # --- choose Ms to plot: all Ms up to 5 as requested
        Ms = sorted(set(depol["M"].unique()).union(set(deph["M"].unique())))
        Ms = [int(m) for m in Ms if int(m) <= 5]
        if not Ms:
            print("No M values <= 5 found.")
            return None

        # --- select max ell per M for each noise family
        def max_ell_per_M(df: pd.DataFrame) -> dict[int, int]:
            out = {}
            for M in sorted(df["M"].unique()):
                dfM = df[df["M"] == M]
                if dfM.empty:
                    continue
                out[int(M)] = int(dfM["purification_level"].max())
            return out

        max_ell_depol = max_ell_per_M(depol)
        max_ell_deph  = max_ell_per_M(deph)

        # --- compute gamma tables (longest trajectory per (M,p,ell))
        # For each noise type, keep only rows at that max ell for each M.
        depol_sel = []
        for M in Ms:
            if M in max_ell_depol:
                ell = max_ell_depol[M]
                depol_sel.append(depol[(depol["M"] == M) & (depol["purification_level"] == ell)])
        depol_sel = pd.concat(depol_sel, ignore_index=True) if depol_sel else pd.DataFrame()

        deph_sel = []
        for M in Ms:
            if M in max_ell_deph:
                ell = max_ell_deph[M]
                deph_sel.append(deph[(deph["M"] == M) & (deph["purification_level"] == ell)])
        deph_sel = pd.concat(deph_sel, ignore_index=True) if deph_sel else pd.DataFrame()

        if depol_sel.empty and deph_sel.empty:
            print("No data after selecting max ell per M.")
            return None

        group_cols = ["M", "p", "purification_level"]
        depol_gamma = build_gamma_table_longest(depol_sel, group_cols, method=method) if not depol_sel.empty else pd.DataFrame()
        deph_gamma  = build_gamma_table_longest(deph_sel,  group_cols, method=method) if not deph_sel.empty else pd.DataFrame()

        # --- optional No-QEC reference: ell=0 curves if present
        depol_noqec = depol[depol["purification_level"] == 0].copy()
        deph_noqec  = deph[deph["purification_level"] == 0].copy()

        depol_noqec_gamma = build_gamma_table_longest(depol_noqec, ["M","p","purification_level"], method=method) if not depol_noqec.empty else pd.DataFrame()
        deph_noqec_gamma  = build_gamma_table_longest(deph_noqec,  ["M","p","purification_level"], method=method) if not deph_noqec.empty else pd.DataFrame()

        if depol_noqec_gamma.empty and deph_noqec_gamma.empty:
            print("No-QEC (ell=0) data not found. Run with --purification-level 0 to include reference lines.")

        # --- plotting
        fig, ax = plt.subplots(figsize=(10, 8))

        # consistent colors per M (reuse your marker helper)
        colors = ['red','green','blue','orange','purple','saddlebrown','deeppink','darkslategrey','fuchsia','gold']

        # depol solid
        for i, M in enumerate(Ms):
            gM = depol_gamma[depol_gamma["M"] == M].sort_values("p")
            if gM.empty:
                continue
            ell = int(gM["purification_level"].iloc[0])
            ax.plot(
                gM["p"], gM["gamma"],
                linestyle='-',
                marker=_mk(i),
                linewidth=2.5,
                markersize=8,
                color=colors[i % len(colors)],
                label=rf'Depol: $M={M}, \ell={ell}$'
            )

        # deph dashed
        for i, M in enumerate(Ms):
            gM = deph_gamma[deph_gamma["M"] == M].sort_values("p")
            if gM.empty:
                continue
            ell = int(gM["purification_level"].iloc[0])
            ax.plot(
                gM["p"], gM["gamma"],
                linestyle='--',
                marker=_mk(i),
                linewidth=2.5,
                markersize=8,
                color=colors[i % len(colors)],
                label=rf'Dephase (twirled): $M={M}, \ell={ell}$'
            )

        # optional No-QEC: thinner lines
        for i, M in enumerate(Ms):
            gM = depol_noqec_gamma[depol_noqec_gamma["M"] == M].sort_values("p")
            if not gM.empty:
                ax.plot(
                    gM["p"], gM["gamma"],
                    linestyle='-',
                    linewidth=1.2,
                    color=colors[i % len(colors)],
                    alpha=0.35,
                    label=rf'No QEC depol: $M={M}$'
                )

        for i, M in enumerate(Ms):
            gM = deph_noqec_gamma[deph_noqec_gamma["M"] == M].sort_values("p")
            if not gM.empty:
                ax.plot(
                    gM["p"], gM["gamma"],
                    linestyle='--',
                    linewidth=1.2,
                    color=colors[i % len(colors)],
                    alpha=0.35,
                    label=rf'No QEC dephase: $M={M}$'
                )

        # ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=40)
        ax.set_xlabel('Physical Error Rate, p', fontsize=40)
        ax.set_ylabel(r'Logical Error, $\gamma_L$', fontsize=40)
        ax.set_title(r'$\gamma$ vs $p$ (solid=depol, dashed=twirled dephasing)', fontsize=22)
        ax.set_xlim(0.00, 1.0)
        # ax.set_ylim(bottom=0.0)
        ax.set_yscale('log')
        ax.legend(fontsize=18, loc='best', ncol=2, frameon=False)
        plt.tight_layout()

        filename = f"gamma_vs_p_combined_maxell_Mle5_{method}.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}")
        return str(filepath)
    
    
    def plot_gamma_vs_p_depolarizing_all_ell(self, method: str = "difference", save_format: str = 'pdf') -> Optional[str]:
        """
        Plot gamma vs p for depolarizing noise with separate curves for all available ℓ values.
        """
        depol = self.depol_steps.copy()
        
        if depol.empty:
            print("No depolarizing steps data")
            return None

        # Check for required columns
        required_cols = {'M', 'p', 'purification_level', 'iteration', 'fidelity', 'run_id'}
        if not required_cols.issubset(depol.columns):
            missing = required_cols - set(depol.columns)
            print(f"Missing columns in depolarizing steps data: {missing}")
            return None

        # Apply twirling filter for depolarizing
        if 'twirling_enabled' in depol.columns:
            depol = depol[depol["twirling_enabled"] == False].copy()
        elif 'twirling_applied' in depol.columns:
            depol = depol[depol["twirling_applied"] == False].copy()

        if depol.empty:
            print("No depolarizing data after twirling filter")
            return None

        # Choose Ms to plot: all Ms up to 5
        Ms = sorted([int(m) for m in depol["M"].unique() if int(m) <= 5])
        if not Ms:
            print("No M values <= 5 found for depolarizing.")
            return None

        # Get all available ℓ values for each M
        fig, axes = plt.subplots(1, len(Ms), figsize=(5*len(Ms), 6), sharey=True)
        if len(Ms) == 1:
            axes = [axes]

        # Colors for different ℓ values
        colors = ['red','green','blue','orange','purple','saddlebrown','deeppink','darkslategrey','fuchsia','gold']

        for ax_idx, M in enumerate(Ms):
            ax = axes[ax_idx]
            
            # Get data for this M
            depol_M = depol[depol["M"] == M].copy()
            if depol_M.empty:
                continue
                
            # Get all ℓ values for this M
            ell_values = sorted(depol_M["purification_level"].unique())
            
            # Plot each ℓ value
            for ell_idx, ell in enumerate(ell_values):
                depol_M_ell = depol_M[depol_M["purification_level"] == ell].copy()
                
                if depol_M_ell.empty:
                    continue
                
                # Build gamma table for this (M, ℓ) combination
                group_cols = ["M", "p", "purification_level"]
                gamma_df = build_gamma_table_longest(depol_M_ell, group_cols, method=method)
                
                if gamma_df.empty:
                    continue
                
                gamma_df = gamma_df.sort_values("p")
                
                ax.plot(
                    gamma_df["p"], gamma_df["gamma"],
                    linestyle='-',
                    marker=_mk(ell_idx),
                    linewidth=2.5,
                    markersize=8,
                    color=colors[ell_idx % len(colors)],
                    label=rf'$\ell = {ell}$'
                )
            
            # Subplot formatting
            ax.set_title(f'M = {M}', fontsize=30)
            # ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=24)
            ax.set_xlabel('Physical Error Rate, p', fontsize=24)
            if ax_idx == 0:  # Only first subplot gets y-label
                if method == "difference":
                    ax.set_ylabel(r'First Iteration Drop, $\gamma$', fontsize=24)
                else:
                    ax.set_ylabel(r'Logical Error, $\gamma$', fontsize=24)
            
            ax.set_xlim(0.0, 1.0)
            if method == "difference":
                ax.set_ylim(bottom=0.0)
            else:
                ax.set_yscale('log')
            
            ax.legend(fontsize=18, loc='best', frameon=False)

        plt.suptitle(f'Depolarizing Noise: γ vs p ({method} method)', fontsize=32)
        plt.tight_layout()

        filename = f"gamma_vs_p_depolarizing_all_ell_{method}.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}")
        return str(filepath)


    def plot_gamma_vs_p_dephasing_all_ell(self, method: str = "difference", save_format: str = 'pdf') -> Optional[str]:
        """
        Plot gamma vs p for dephasing noise with separate curves for all available ℓ values.
        """
        deph = self.dephase_steps.copy()
        
        if deph.empty:
            print("No dephasing steps data")
            return None

        # Check for required columns
        required_cols = {'M', 'p', 'purification_level', 'iteration', 'fidelity', 'run_id'}
        if not required_cols.issubset(deph.columns):
            missing = required_cols - set(deph.columns)
            print(f"Missing columns in dephasing steps data: {missing}")
            return None

        # Apply twirling filter for dephasing
        if 'twirling_enabled' in deph.columns:
            deph = deph[deph["twirling_enabled"] == True].copy()
        elif 'twirling_applied' in deph.columns:
            deph = deph[deph["twirling_applied"] == True].copy()

        if deph.empty:
            print("No dephasing data after twirling filter")
            return None

        # Choose Ms to plot: all Ms up to 5
        Ms = sorted([int(m) for m in deph["M"].unique() if int(m) <= 5])
        if not Ms:
            print("No M values <= 5 found for dephasing.")
            return None

        # Get all available ℓ values for each M
        fig, axes = plt.subplots(1, len(Ms), figsize=(5*len(Ms), 6), sharey=True)
        if len(Ms) == 1:
            axes = [axes]

        # Colors for different ℓ values
        colors = ['red','green','blue','orange','purple','saddlebrown','deeppink','darkslategrey','fuchsia','gold']

        for ax_idx, M in enumerate(Ms):
            ax = axes[ax_idx]
            
            # Get data for this M
            deph_M = deph[deph["M"] == M].copy()
            if deph_M.empty:
                continue
                
            # Get all ℓ values for this M
            ell_values = sorted(deph_M["purification_level"].unique())
            
            # Plot each ℓ value
            for ell_idx, ell in enumerate(ell_values):
                deph_M_ell = deph_M[deph_M["purification_level"] == ell].copy()
                
                if deph_M_ell.empty:
                    continue
                
                # Build gamma table for this (M, ℓ) combination
                group_cols = ["M", "p", "purification_level"]
                gamma_df = build_gamma_table_longest(deph_M_ell, group_cols, method=method)
                
                if gamma_df.empty:
                    continue
                
                gamma_df = gamma_df.sort_values("p")
                
                ax.plot(
                    gamma_df["p"], gamma_df["gamma"],
                    linestyle='-',
                    marker=_mk(ell_idx),
                    linewidth=2.5,
                    markersize=8,
                    color=colors[ell_idx % len(colors)],
                    label=rf'$\ell = {ell}$'
                )
            
            # Subplot formatting
            ax.set_title(f'M = {M}', fontsize=30)
            # ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=24)
            ax.set_xlabel('Physical Error Rate, p', fontsize=24)
            if ax_idx == 0:  # Only first subplot gets y-label
                if method == "difference":
                    ax.set_ylabel(r'First Iteration Drop, $\gamma$', fontsize=24)
                else:
                    ax.set_ylabel(r'Logical Error, $\gamma$', fontsize=24)
            
            ax.set_xlim(0.0, 1.0)
            if method == "difference":
                ax.set_ylim(bottom=0.0)
            else:
                ax.set_yscale('log')
            
            ax.legend(fontsize=18, loc='best', frameon=False)

        plt.suptitle(f'Dephasing Noise (Twirled): γ vs p ({method} method)', fontsize=32)
        plt.tight_layout()

        filename = f"gamma_vs_p_dephasing_all_ell_{method}.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}")
        return str(filepath)
    
    def plot_gamma_from_first_step_vs_p(self, noise_type: str, save_format: str = "pdf") -> Optional[str]:
        """
        Compute gamma(p) via gamma = F(t=0) - F(t=1) with F(t=0)=1 baseline,
        and plot gamma vs p for all available purification levels ℓ.

        ℓ=0 curves are dotted.
        Legend ordering is grouped by M (all M=1 entries first, then M=5, etc.).
        """

        # Select data and twirling condition
        if noise_type == "depolarizing":
            df = self.depol_steps
            twirling_filter = False
            title_str = "Depolarizing"
        else:  # 'dephasing'
            df = self.dephase_steps
            twirling_filter = True
            title_str = "Dephasing (twirled Z)"

        if df.empty:
            print(f"No steps data for {noise_type}")
            return None

        required = {"M", "p", "fidelity", "iteration", "purification_level"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"Missing columns for gamma plot: {missing}")
            print(f"Available columns: {list(df.columns)}")
            return None

        # Twirling filter
        if "twirling_enabled" in df.columns:
            df_tw = df[df["twirling_enabled"] == twirling_filter].copy()
        elif "twirling_applied" in df.columns:
            df_tw = df[df["twirling_applied"] == twirling_filter].copy()
        else:
            df_tw = df.copy()

        if df_tw.empty:
            print(f"No data after twirling filter for {noise_type} (twirling={twirling_filter})")
            return None

        # Use iteration==1 rows for F(t=1)
        df_t1 = df_tw[df_tw["iteration"] == 1].copy()
        if df_t1.empty:
            print(f"No iteration==1 rows found for {noise_type}; cannot compute gamma")
            return None

        # Average duplicates (if any) over identical (M, ℓ, p)
        grp_cols = ["M", "purification_level", "p"]
        df_gamma = (
            df_t1.groupby(grp_cols, as_index=False)["fidelity"]
            .mean()
            .rename(columns={"fidelity": "F_t1"})
        )
        df_gamma["gamma"] = 1.0 - df_gamma["F_t1"]

        # -------------------------
        # Plot: group by M in order
        # -------------------------
        fig, ax = plt.subplots(figsize=(10, 8))

        Ms = sorted(df_gamma["M"].unique())                    # ensures M=1 then M=5
        l_values = sorted(df_gamma["purification_level"].unique())

        colors = [
            "red", "green", "blue", "orange", "purple", "saddlebrown",
            "deeppink", "darkslategrey", "fuchsia", "gold"
        ]

        handles = []
        labels = []
        series_idx = 0

        # IMPORTANT: outer loop is M, so legend groups by M
        for M in Ms:
            for l_val in l_values:
                df_ml = df_gamma[(df_gamma["M"] == M) & (df_gamma["purification_level"] == l_val)].copy()
                if df_ml.empty:
                    continue
                df_ml = df_ml.sort_values("p")

                linestyle = ":" if int(l_val) == 0 else "-"   # ℓ=0 dotted

                (line,) = ax.plot(
                    df_ml["p"], df_ml["gamma"],
                    linestyle=linestyle,
                    marker=_mk(series_idx),
                    linewidth=2.5,
                    markersize=8,
                    alpha=0.85,
                    color=colors[series_idx % len(colors)],
                    label=rf"$M={M},\,\ell={l_val}$",
                )

                handles.append(line)
                labels.append(rf"$M={M},\,\ell={l_val}$")
                series_idx += 1

        # ax.set_xlabel(r"Physical Error Rate, $p$", fontsize=40)
        ax.set_xlabel('Physical Error Rate, p', fontsize=40)
        ax.set_ylabel(r"Logical Error, $\gamma_L$", fontsize=40)
        # ax.set_title(f"Gamma from First Iteration\n({title_str} Noise)", fontsize=36)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        # ax.set_ylim(1e-4, 1.05)
        # ax.set_yscale("log")

        # Legend uses our handle order (grouped by M)
        ax.legend(handles, labels, fontsize=18, loc="best", ncol=1, frameon=False)

        plt.tight_layout()

        filename = f"gamma_firststep_vs_p_{noise_type}.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved {filename}")
        return str(filepath)


    def plot_final_fidelity_vs_p_all_l(self, noise_type: str, save_format: str = "pdf") -> Optional[str]:
        """
        Plot FINAL fidelity vs p with curves for all purification levels ℓ.
        Uses steps.csv and picks the last-iteration fidelity for each (M, p, ℓ).

        - One figure per noise type.
        - Curves are grouped in the legend by M (all M=1 entries together, then M=5, ...).
        - ℓ=0 (if present) is dotted.
        """
        # Select data and twirling condition
        if noise_type == "depolarizing":
            df = self.depol_steps
            twirling_filter = False
            title_str = "Depolarizing"
        else:  # dephasing
            df = self.dephase_steps
            twirling_filter = True
            title_str = "Dephasing (twirled Z)"

        if df.empty:
            print(f"No steps data for {noise_type}")
            return None

        required = {"M", "p", "fidelity", "iteration", "purification_level"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"Missing columns for final fidelity plot: {missing}")
            print(f"Available columns: {list(df.columns)}")
            return None

        # Apply twirling filter if present
        if "twirling_enabled" in df.columns:
            df_tw = df[df["twirling_enabled"] == twirling_filter].copy()
        elif "twirling_applied" in df.columns:
            df_tw = df[df["twirling_applied"] == twirling_filter].copy()
        else:
            df_tw = df.copy()

        if df_tw.empty:
            print(f"No data after twirling filter for {noise_type} (twirling={twirling_filter})")
            return None

        # ----------------------------------------------------
        # Reduce to "final iteration" per (M, p, ℓ)
        # ----------------------------------------------------
        # If there are multiple rows per (M,p,ℓ), we:
        #   1) find max iteration
        #   2) take the fidelity at that max iteration
        #   3) if duplicates remain, average them
        grp_cols = ["M", "purification_level", "p"]
        df_max_iter = df_tw.groupby(grp_cols, as_index=False)["iteration"].max().rename(columns={"iteration": "iteration_max"})
        df_final = df_tw.merge(df_max_iter, on=grp_cols, how="inner")
        df_final = df_final[df_final["iteration"] == df_final["iteration_max"]].copy()

        # Average if duplicates (should be rare)
        df_final = (
            df_final.groupby(grp_cols, as_index=False)["fidelity"]
            .mean()
            .rename(columns={"fidelity": "fidelity_final"})
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 8))

        Ms = sorted(df_final["M"].unique())  # groups legend by M
        l_values = sorted(df_final["purification_level"].unique())

        colors = [
            "red", "green", "blue", "orange", "purple", "saddlebrown",
            "deeppink", "darkslategrey", "fuchsia", "gold"
        ]

        handles, labels = [], []
        series_idx = 0

        # Legend/grouping order: M outer loop
        for M in Ms:
            for l_val in l_values:
                d = df_final[(df_final["M"] == M) & (df_final["purification_level"] == l_val)].copy()
                if d.empty:
                    continue
                d = d.sort_values("p")

                linestyle = ":" if int(l_val) == 0 else "-"  # ℓ=0 dotted

                (line,) = ax.plot(
                    d["p"], d["fidelity_final"],
                    linestyle=linestyle,
                    marker=_mk(series_idx),
                    linewidth=2.5,
                    markersize=8,
                    alpha=0.85,
                    color=colors[series_idx % len(colors)],
                    label=rf"$M={M},\,\ell={l_val}$",
                )

                handles.append(line)
                labels.append(rf"$M={M},\,\ell={l_val}$")
                series_idx += 1

        # ax.set_xlabel(r"Physical Error Rate, $p$", fontsize=40)
        ax.set_xlabel('Physical Error Rate, p', fontsize=40)
        ax.set_ylabel(r"Fidelity, $F$", fontsize=40)
        # ax.set_title(f"Final Fidelity vs $p$\n({title_str} Noise)", fontsize=36)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)

        ax.legend(handles, labels, fontsize=18, loc="best", ncol=1, frameon=False)
        plt.tight_layout()

        filename = f"final_fidelity_vs_p_all_l_{noise_type}.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved {filename}")
        return str(filepath)
    
    
    def plot_gamma_from_first_step_vs_p_M1_M5_grid_with_insets(
        self,
        noise_type: str,
        save_format: str = "pdf",
        *,
        p_center: float = 0.75,
        p_halfwidth: float = 0.05,
    ) -> Optional[str]:
        """
        Vertical 2x1 plot of gamma(p)=1-F(t=1):
        - top:    M=1 (NO inset)
        - bottom: M=5 (WITH inset)

        Inset behavior:
        - Plot the full M=5 curves in the inset (no p-window filtering),
            then apply xlim to zoom. This avoids "not enough points" when
            tightening the zoom window.

        Notes
        -----
        - Uses iteration==1 rows for F(t=1).
        - Filters by twirling condition:
            * depolarizing -> twirling=False
            * dephasing    -> twirling=True (twirled Z)
        - ℓ=0 dotted, ℓ>0 solid.
        """
        colors = ["red", "green", "blue", "orange", "purple", "saddlebrown",
                "deeppink", "darkslategrey", "fuchsia", "gold"]

        # -------------------------
        # Select data and twirling condition
        # -------------------------
        if noise_type == "depolarizing":
            df = self.depol_steps
            twirling_filter = False
            title_str = "Depolarizing"
        else:  # 'dephasing'
            df = self.dephase_steps
            twirling_filter = True
            title_str = "Dephasing (twirled Z)"

        if df.empty:
            print(f"No steps data for {noise_type}")
            return None

        required = {"M", "p", "fidelity", "iteration", "purification_level"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"Missing columns for gamma plot: {missing}")
            print(f"Available columns: {list(df.columns)}")
            return None

        # -------------------------
        # Twirling filter
        # -------------------------
        if "twirling_enabled" in df.columns:
            df_tw = df[df["twirling_enabled"] == twirling_filter].copy()
        elif "twirling_applied" in df.columns:
            df_tw = df[df["twirling_applied"] == twirling_filter].copy()
        else:
            df_tw = df.copy()

        if df_tw.empty:
            print(f"No data after twirling filter for {noise_type} (twirling={twirling_filter})")
            return None

        # -------------------------
        # Use iteration==1 rows for F(t=1)
        # -------------------------
        df_t1 = df_tw[df_tw["iteration"] == 1].copy()
        if df_t1.empty:
            print(f"No iteration==1 rows found for {noise_type}; cannot compute gamma")
            return None

        # Average duplicates over identical (M, ℓ, p)
        grp_cols = ["M", "purification_level", "p"]
        df_gamma = (
            df_t1.groupby(grp_cols, as_index=False)["fidelity"]
            .mean()
            .rename(columns={"fidelity": "F_t1"})
        )
        df_gamma["gamma"] = 1.0 - df_gamma["F_t1"]

        # -------------------------
        # Plot config
        # -------------------------
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        import matplotlib.ticker as mticker

        target_Ms = [1, 5]
        l_values = sorted(df_gamma["purification_level"].unique())
        l_to_color_idx = {l_val: idx for idx, l_val in enumerate(l_values)}

        # Zoom window for inset (x only; Option B plots full curve then zooms)
        p_min_zoom = max(0.0, p_center - p_halfwidth)
        p_max_zoom = min(1.0, p_center + p_halfwidth)

        # Keep your per-M y-lims to expose the crossover clearly
        inset_ylim_by_M = {
            5: (0.95, 0.98),
        }

        # Vertical layout: 2 rows x 1 col
        fig, axes = plt.subplots(2, 1, figsize=(11, 14), sharex=False)
        # fig.suptitle(f"Gamma from First Iteration ({title_str} Noise)", fontsize=26)

        subplot_labels = {1: "a", 5: "b"}

        for ax, M in zip(axes, target_Ms):
            df_M = df_gamma[df_gamma["M"] == M].copy()

            ax.set_title(f"$M={M}$", fontsize=40)
            ax.set_xlabel("Physical Error Rate, p", fontsize=40)
            ax.set_ylabel(r"Logical Error, $\gamma_L$", fontsize=40)

            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.05)
            ax.tick_params(axis="both", which="major", labelsize=30)
            exclude = {0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79}
            p_round = df_M["p"].round(2)
            mask_keep = ~p_round.isin(exclude)
            df_main = df_M[mask_keep].copy()

            # subplot label in the corner
            ax.text(
                0.08, 0.98, subplot_labels[M],
                transform=ax.transAxes,
                fontsize=36, fontweight="bold", fontfamily="sans-serif",
                va="top", ha="right",
            )

            if df_M.empty:
                ax.text(
                    0.5, 0.5, f"No data for M={M}",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=18, alpha=0.8,
                )
                continue

            # ---------- main panel lines ----------
            handles, labels = [], []
            for l_val in l_values:
                df_ml = df_M[df_M["purification_level"] == l_val].sort_values("p")
                if df_ml.empty:
                    continue
                
                p_round = df_ml["p"].round(2)
                df_main = df_ml[~p_round.isin(exclude)].copy()

                linestyle = ":" if int(l_val) == 0 else "-"
                cidx = l_to_color_idx[l_val]
                

                (line,) = ax.plot(
                    df_main["p"], df_main["gamma"],
                    linestyle=linestyle,
                    marker=_mk(cidx),
                    linewidth=2.5,
                    markersize=12,
                    alpha=0.85,
                    color=colors[cidx % len(colors)],
                    label=rf"$\ell={l_val}$",
                )
                handles.append(line)
                labels.append(rf"$\ell={l_val}$")

            # keep legend only on M=1 (top) to reduce clutter (same as your previous style)
            if M == 1:
                ax.legend(handles, labels, fontsize=18, loc="best", ncol=1, frameon=False)

            # -------------------------
            # Inset ONLY for M=5
            # -------------------------
            if M != 5:
                continue

            axins = inset_axes(ax, width="48%", height="42%", loc="lower right", borderpad=1.1)

            # Option B: plot full curves (no zoom filtering), then just zoom via xlim
            for l_val in l_values:
                df_ml = df_M[df_M["purification_level"] == l_val].sort_values("p")
                if df_ml.empty:
                    continue

                linestyle = ":" if int(l_val) == 0 else "-"
                cidx = l_to_color_idx[l_val]

                x = df_ml["p"].to_numpy(dtype=float)
                y = df_ml["gamma"].to_numpy(dtype=float)

                # log scale can't show nonpositive values
                mask = y > 0.0
                if mask.sum() < 2:
                    continue

                axins.plot(
                    x[mask], y[mask],
                    linestyle=linestyle,
                    marker=_mk(cidx),
                    linewidth=2.0,
                    markersize=6,
                    alpha=0.9,
                    color=colors[cidx % len(colors)],
                )

            axins.set_xlim(p_min_zoom, p_max_zoom)
            axins.set_yscale("log")

            # Force your desired y-range for crossover visibility
            if 5 in inset_ylim_by_M:
                ymin, ymax = inset_ylim_by_M[5]
                ymin = max(float(ymin), 1e-12)
                ymax = max(float(ymax), ymin * 1.01)
                axins.set_ylim(ymin, ymax)
    

            # vertical reference line at p_center
            # axins.axvline(p_center, color="black", linewidth=1.6, alpha=0.9)

            # Tick label sizes (override global rcParams)
            axins.tick_params(axis="x", which="major", labelsize=10, length=4, width=1.0)
            axins.tick_params(axis="y", which="major", labelsize=10, length=4, width=1.0)

            # If you want only a few y ticks (optional, but can help readability)
            # yt = [0.960, 0.965, 0.970, 0.975]
            # axins.yaxis.set_major_locator(mticker.FixedLocator(yt))
            # axins.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{v:.3f}"))
            # axins.yaxis.set_minor_formatter(mticker.NullFormatter())
            
            axins.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
            axins.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10.0))
            axins.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
            axins.yaxis.set_minor_formatter(mticker.NullFormatter())

        plt.tight_layout()

        filename = f"gamma_firststep_vs_p_{noise_type}_M1_M5_vertical_insetM5.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved {filename}")
        return str(filepath)
    
    
    def plot_combined_fidelity_and_gamma_2x2_grid(self, noise_type: str, save_format: str = 'pdf', 
                                              *, p_center: float = 0.75, p_halfwidth: float = 0.05):
        """
        Combined 2x2 grid plot:
        - Top row (a,b): Fidelity vs iterations for p=0.1, 0.3 (M=5)
        - Bottom row (c,d): Gamma vs p for M=1, M=5 (with inset on M=5)
        
        Args:
            noise_type: 'depolarizing' or 'dephasing'
            save_format: 'pdf', 'png', etc.
            p_center: Center p-value for inset zoom
            p_halfwidth: Half-width for inset zoom window
        """
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        import matplotlib.ticker as mticker
        import numpy as np
        
        # Select data and set twirling filter
        if noise_type == 'depolarizing':
            df_steps = self.depol_steps
            twirling_filter = False
            title_str = "Depolarizing"
        else:  # dephasing
            df_steps = self.dephase_steps
            twirling_filter = True
            title_str = "Dephasing (twirled Z)"

        if df_steps.empty:
            print(f"No steps data for {noise_type}")
            return None

        # Check for required columns
        required_cols = ['iteration', 'purification_level', 'M', 'p', 'fidelity']
        missing_cols = [col for col in required_cols if col not in df_steps.columns]
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            return None

        # Handle twirling column name
        twirling_col = 'twirling_applied' if 'twirling_applied' in df_steps.columns else 'twirling_enabled'
        if twirling_col not in df_steps.columns:
            print(f"No twirling column found in {noise_type} data")
            return None

        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'saddlebrown', 
                'deeppink', 'darkslategrey', 'fuchsia', 'gold']
        
        # Subplot labels
        subplot_labels = [['a', 'b'], ['c', 'd']]
        
        # ================================================================
        # TOP ROW: FIDELITY VS ITERATIONS (M=5, p=0.1 and p=0.3)
        # ================================================================
        
        print("Plotting top row: Fidelity vs Iterations")
        
        # Filter for M=5 and correct twirling
        df_fid_filtered = df_steps[(df_steps['M'] == 5) & (df_steps[twirling_col] == twirling_filter)].copy()
        print(f"Fidelity plots: {len(df_fid_filtered)} rows after M=5, twirling filter")
        
        if not df_fid_filtered.empty:
            l_values = sorted(df_fid_filtered['purification_level'].unique())
            p_subset_fid = [0.1, 0.3]  # For top row
            
            for i, p_val in enumerate(p_subset_fid):
                ax = axes[0, i]  # Top row
                
                # Find closest p-value
                available_p = df_fid_filtered['p'].unique()
                closest_p = min(available_p, key=lambda x: abs(x - p_val))
                
                df_p = df_fid_filtered[abs(df_fid_filtered['p'] - closest_p) <= 0.02].copy()
                
                if df_p.empty:
                    ax.text(0.5, 0.5, f'No data\\nfor p = {p_val:.1f}',
                        ha='center', va='center', transform=ax.transAxes, fontsize=14)
                    continue
                
                # Plot curves for different ℓ values
                for j, l_val in enumerate(l_values):
                    df_l = df_p[df_p['purification_level'] == l_val].copy()
                    
                    # Apply extended hotfix for ℓ=3
                    df_l = self.add_missing_ell3_data_extended(df_l, l_val, closest_p, noise_type)
                    
                    if df_l.empty:
                        continue
                    
                    # Sort and get arrays
                    df_l = df_l.sort_values('iteration')
                    x = df_l['iteration'].to_numpy()
                    y = df_l['fidelity'].to_numpy()
                    
                    if len(x) == 0:
                        continue
                    
                    # Prepend (0,1) if needed
                    if x[0] != 0:
                        x = np.insert(x, 0, 0)
                        y = np.insert(y, 0, 1.0)
                    
                    # Plot
                    if l_val == 0:
                        ax.plot(x, y, linestyle='dotted', marker=_mk(j), 
                            color=colors[j % len(colors)], linewidth=2, markersize=12, 
                            alpha=0.8, label=rf'No QEC')
                    else:
                        ax.plot(x, y, linestyle='-', marker=_mk(j), 
                            color=colors[j % len(colors)], linewidth=2, markersize=12, 
                            alpha=0.8, label=rf'$\ell$ = {l_val}')
                
                # Formatting for fidelity plots
                ax.set_title(f'p = {closest_p:.1f}', fontsize=55)
                ax.set_ylim(0.005, 1.05)
                ax.set_yscale('log')
                
                # X-axis
                max_iter = max(10, df_p['iteration'].max()) if len(df_p) > 0 else 10
                ax.set_xlim(0.0, max_iter + 0.5)
                ax.set_xticks(range(0, int(max_iter) + 1, 2))
                ax.set_xlabel(r'PQEC Cycles, $t$', fontsize=50)
                
                # Y-axis label only on left
                if i == 0:
                    ax.set_ylabel(r'Fidelity, $F$', fontsize=50)
                
                # Legend on right plot
                if i == 1:
                    ax.legend(fontsize=28, loc='best', frameon=False)
                
                # Subplot label
                ax.text(0.08, 0.14, subplot_labels[0][i], transform=ax.transAxes, 
                    fontsize=50, fontweight='bold', va='top', ha='right')
        
        # ================================================================
        # BOTTOM ROW: GAMMA VS P (M=1 and M=5)
        # ================================================================
        
        print("Plotting bottom row: Gamma vs p")
        
        # Prepare gamma data (same logic as original function)
        df_tw = df_steps[df_steps[twirling_col] == twirling_filter].copy()
        df_t1 = df_tw[df_tw['iteration'] == 1].copy()
        
        if not df_t1.empty:
            # Calculate gamma = 1 - F(t=1)
            grp_cols = ['M', 'purification_level', 'p']
            df_gamma = (df_t1.groupby(grp_cols, as_index=False)['fidelity']
                        .mean().rename(columns={'fidelity': 'F_t1'}))
            df_gamma['gamma'] = 1.0 - df_gamma['F_t1']
            
            l_values_gamma = sorted(df_gamma['purification_level'].unique())
            l_to_color_idx = {l_val: idx for idx, l_val in enumerate(l_values_gamma)}
            
            # Exclusion set for main plots
            exclude = {0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79}
            
            # Inset zoom parameters
            p_min_zoom = max(0.0, p_center - p_halfwidth)
            p_max_zoom = min(1.0, p_center + p_halfwidth)
            inset_ylim = (0.95, 0.98)  # For M=5 inset
            
            target_Ms = [1, 5]
            
            for j, M in enumerate(target_Ms):
                ax = axes[1, j]  # Bottom row
                
                df_M = df_gamma[df_gamma['M'] == M].copy()
                
                # Formatting
                ax.set_title(f'M={M}', fontsize=55)
                ax.set_xlabel('Physical Error Rate, p', fontsize=50)
                if j == 0:  # Left plot only
                    ax.set_ylabel(r'Logical Error, $\gamma_L$', fontsize=50)
                
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.05)
                
                if df_M.empty:
                    ax.text(0.5, 0.5, f'No data for M={M}', ha='center', va='center',
                        transform=ax.transAxes, fontsize=14)
                    continue
                
                # Plot main gamma curves
                handles, labels = [], []
                for l_val in l_values_gamma:
                    df_ml = df_M[df_M['purification_level'] == l_val].copy()
                    if df_ml.empty:
                        continue
                    
                    # Apply exclusion filter
                    p_round = df_ml['p'].round(2)
                    df_main = df_ml[~p_round.isin(exclude)].copy()
                    
                    if df_main.empty:
                        continue
                    
                    # Sort and plot
                    df_main = df_main.sort_values('p')
                    linestyle = ':' if int(l_val) == 0 else '-'
                    cidx = l_to_color_idx[l_val]
                    
                    
                    if l_val == 0:
                        label = rf'No QEC'
                    else:
                        label = rf'$\ell={l_val}$'
                    line, = ax.plot(df_main['p'], df_main['gamma'],
                                linestyle=linestyle, marker=_mk(cidx),
                                linewidth=2.5, markersize=12, alpha=0.85,
                                color=colors[cidx % len(colors)],
                                label=label)
                    handles.append(line)
                    labels.append(label)
                
                # Legend only on M=1 (left plot)
                if M == 1:
                    ax.legend(handles, labels, fontsize=28, loc='best', frameon=False)
                
                # Subplot label
                ax.text(0.08, 0.98, subplot_labels[1][j], transform=ax.transAxes,
                    fontsize=50, fontweight='bold', va='top', ha='right')
                
                # ========================
                # INSET ONLY FOR M=5
                # ========================
                if M == 5:
                    axins = inset_axes(ax, width='48%', height='42%', 
                                    loc='lower right', borderpad=1.1)
                    
                    # Plot full curves in inset
                    for l_val in l_values_gamma:
                        df_ml = df_M[df_M['purification_level'] == l_val].copy()
                        if df_ml.empty:
                            continue
                        
                        df_ml = df_ml.sort_values('p')
                        linestyle = ':' if int(l_val) == 0 else '-'
                        cidx = l_to_color_idx[l_val]
                        
                        x = df_ml['p'].to_numpy(dtype=float)
                        y = df_ml['gamma'].to_numpy(dtype=float)
                        
                        # Filter positive values for log scale
                        mask = y > 0.0
                        if mask.sum() < 2:
                            continue
                        
                        axins.plot(x[mask], y[mask], linestyle=linestyle, marker=_mk(cidx),
                                linewidth=2.0, markersize=6, alpha=0.9,
                                color=colors[cidx % len(colors)])
                    
                    # Inset formatting
                    axins.set_xlim(p_min_zoom, p_max_zoom)
                    axins.set_yscale('log')
                    
                    # Set y-limits for crossover visibility
                    ymin, ymax = inset_ylim
                    ymin = max(float(ymin), 1e-12)
                    ymax = max(float(ymax), ymin * 1.01)
                    axins.set_ylim(ymin, ymax)
                    
                    # Inset tick formatting
                    axins.tick_params(axis='both', which='major', labelsize=10)
                    axins.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
                    axins.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10.0))
                    axins.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
                    axins.yaxis.set_minor_formatter(mticker.NullFormatter())
        
        # Overall formatting
        plt.tight_layout()
        
        # Save
        filename = f"combined_fidelity_gamma_2x2_{noise_type}.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {filename}")
        return str(filepath)
    
    
    
    def plot_fidelity_vs_purification_rounds_iteration1(self, noise_type: str, save_format: str = 'pdf') -> Optional[str]:
        """
        2x2 grid showing fidelity vs purification rounds (ℓ) for iteration==1.
        Each subplot shows curves for different M values at fixed p.
        Uses the same formatting as plot_fidelity_grid_vs_depth_mini.
        """
        # Select data based on noise type
        if noise_type == 'depolarizing':
            df = self.depol_steps
            target_p_values = [0.1, 0.3, 0.7, 0.8]
            title_suffix = "Depolarizing"
            filename_suffix = "depolarizing"
            twirling_filter = False
        elif noise_type == 'dephasing':
            df = self.dephase_theta_phi_steps
            target_p_values = [0.1, 0.3, 0.5, 0.7]
            title_suffix = "Dephasing (Twirled)"
            filename_suffix = "dephasing"
            twirling_filter = False
        else:
            print(f"Unknown noise type: {noise_type}")
            return None

        if df.empty:
            print(f"No steps data for {noise_type}")
            return None

        # Check for required columns
        required_cols = ['iteration', 'purification_level', 'M', 'p', 'fidelity']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            return None

        # Handle twirling column name and apply filter
        twirling_col = 'twirling_applied' if 'twirling_applied' in df.columns else 'twirling_enabled'
        if twirling_col in df.columns:
            df_filtered = df[df[twirling_col] == twirling_filter].copy()
        else:
            df_filtered = df.copy()

        # Filter for iteration==1
        df_iter1 = df_filtered[df_filtered['iteration'] == 1].copy()
        
        if df_iter1.empty:
            print(f"No iteration==1 data for {noise_type}")
            return None

        # Find closest p values in data for each target p
        available_ps = df_iter1['p'].unique()
        ps = []
        for target_p in target_p_values:
            closest_p = min(available_ps, key=lambda x: abs(x - target_p))
            # Only use if within reasonable tolerance (0.05)
            if abs(closest_p - target_p) <= 0.05:
                ps.append(closest_p)
            else:
                print(f"Warning: No data found close to p={target_p:.1f} for {noise_type}")
        
        if len(ps) == 0:
            print(f"No valid p values found for {noise_type}")
            return None

        # Get unique M values
        M_values = sorted(df_iter1['M'].unique())
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'cyan']
        subplot_labels = ['a', 'b', 'c', 'd']

        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Flatten axes for easier iteration
        axes_flat = axes.flatten()

        for plot_idx, p in enumerate(ps):
            if plot_idx >= 4:  # Safety check for 2x2 grid
                break
                
            ax = axes_flat[plot_idx]
            df_p = df_iter1[abs(df_iter1['p'] - p) <= 0.01].copy()  # Use small tolerance for p matching

            if df_p.empty:
                ax.text(0.5, 0.5, f'No data\\nfor p = {p:.2f}',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14, alpha=0.7)
                continue

            # Plot curves for each M value
            for i, M in enumerate(M_values):
                df_M = df_p[df_p['M'] == M].copy()
                
                if len(df_M) > 0:
                    # Group by purification_level and take mean fidelity (in case of duplicates)
                    evolution = df_M.groupby('purification_level')['fidelity'].mean().reset_index()
                    evolution = evolution.sort_values('purification_level')
                    
                    if len(evolution) > 0:
                        ax.plot(evolution['purification_level'], evolution['fidelity'],
                            linestyle='-', marker=_mk(i),
                            color=colors[i], linewidth=2, markersize=12,
                            label=f'M = {M}', alpha=0.8)

            # Subplot formatting
            ax.set_title(f'$p = {p:.2f}$', fontsize=30)
            ax.set_ylim(0, 1.05)
            
            # Set x-axis based on available purification levels
            if not df_p.empty:
                max_ell = df_p['purification_level'].max()
                min_ell = df_p['purification_level'].min()
                ax.set_xlim(min_ell - 0.1, max_ell + 0.1)
                # Set reasonable tick marks
                if max_ell <= 5:
                    ax.set_xticks(range(int(min_ell), int(max_ell) + 1))
                else:
                    ax.set_xticks(range(int(min_ell), int(max_ell) + 1, 2))
            
            # Add legend to top-left subplot
            if plot_idx == 0:
                ax.legend(fontsize=20, loc='lower right')
                
            if plot_idx == 1:
                ax.axhline(1.0, color='gray', linestyle='dotted', linewidth=5.0, alpha=1.0)
                
            # Y-axis label only on first column
            if plot_idx == 0 or plot_idx == 2:
                ax.set_ylabel(r'Fidelity, $F$', fontsize=40)
                
            # X-axis label only on bottom row
            if plot_idx == 2 or plot_idx == 3:
                ax.set_xlabel(r'Purification Rounds, $\ell$', fontsize=40)
                
            # Add subplot label with same positioning as original
            if plot_idx == 0 or plot_idx == 1:
                ax.text(0.08, 0.02, subplot_labels[plot_idx], transform=ax.transAxes, fontsize=28, 
                    fontweight='bold', fontfamily='sans-serif', va='bottom', ha='right')
            else:
                ax.text(0.08, 0.88, subplot_labels[plot_idx], transform=ax.transAxes, fontsize=28, 
                    fontweight='bold', fontfamily='sans-serif', va='bottom', ha='right')

        # Hide unused subplots (if less than 4 p values found)
        for plot_idx in range(len(ps), 4):
            axes_flat[plot_idx].set_visible(False)

        plt.tight_layout()

        filename = f"fidelity_vs_purification_rounds_iter1_{filename_suffix}.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved {filename}")
        return str(filepath)
    
    
    
    
    
    
    
    
    def plot_gamma_vs_purification_level_fixed_p(self, noise_type: str, target_p: float = 0.3, save_format: str = "pdf") -> Optional[str]:
        """
        Plot gamma vs purification level (ℓ) for a fixed p-value.
        Cross-section of the gamma vs p plots.
        
        Args:
            noise_type: 'depolarizing' or 'dephasing'
            target_p: Fixed p-value to plot cross-section at (default 0.3)
            save_format: File format for saving
        """
        
        # Select data and twirling condition
        if noise_type == "depolarizing":
            df = self.depol_steps
            twirling_filter = False
            title_str = "Depolarizing"
        else:  # 'dephasing'
            df = self.dephase_steps
            twirling_filter = True
            title_str = "Dephasing (twirled Z)"

        if df.empty:
            print(f"No steps data for {noise_type}")
            return None

        required = {"M", "p", "fidelity", "iteration", "purification_level"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"Missing columns for gamma plot: {missing}")
            print(f"Available columns: {list(df.columns)}")
            return None

        # Twirling filter
        if "twirling_enabled" in df.columns:
            df_tw = df[df["twirling_enabled"] == twirling_filter].copy()
        elif "twirling_applied" in df.columns:
            df_tw = df[df["twirling_applied"] == twirling_filter].copy()
        else:
            df_tw = df.copy()

        if df_tw.empty:
            print(f"No data after twirling filter for {noise_type} (twirling={twirling_filter})")
            return None

        # Use iteration==1 rows for F(t=1)
        df_t1 = df_tw[df_tw["iteration"] == 1].copy()
        if df_t1.empty:
            print(f"No iteration==1 rows found for {noise_type}; cannot compute gamma")
            return None

        # Filter for target p-value with tolerance
        p_tolerance = 0.02
        df_p_filtered = df_t1[abs(df_t1["p"] - target_p) <= p_tolerance].copy()
        
        if df_p_filtered.empty:
            # Try to find the closest available p-value
            available_p = df_t1["p"].unique()
            closest_p = min(available_p, key=lambda x: abs(x - target_p))
            print(f"No data for p≈{target_p:.2f}, using closest p={closest_p:.2f}")
            df_p_filtered = df_t1[abs(df_t1["p"] - closest_p) <= p_tolerance].copy()
            actual_p = closest_p
        else:
            actual_p = target_p

        if df_p_filtered.empty:
            print(f"No data found for any p-value near {target_p:.2f}")
            return None

        # Average duplicates over identical (M, ℓ)
        grp_cols = ["M", "purification_level"]
        df_gamma = (
            df_p_filtered.groupby(grp_cols, as_index=False)["fidelity"]
            .mean()
            .rename(columns={"fidelity": "F_t1"})
        )
        df_gamma["gamma"] = 1.0 - df_gamma["F_t1"]

        # -------------------------
        # Plot: separate curves for each M
        # -------------------------
        fig, ax = plt.subplots(figsize=(10, 8))

        Ms = sorted(df_gamma["M"].unique())
        colors = [
            "red", "green", "blue", "orange", "purple", "saddlebrown",
            "deeppink", "darkslategrey", "fuchsia", "gold"
        ]

        handles = []
        labels = []

        for i, M in enumerate(Ms):
            df_M = df_gamma[df_gamma["M"] == M].copy()
            if df_M.empty:
                continue
                
            # Sort by purification level
            df_M = df_M.sort_values("purification_level")
            
            # Plot this M curve
            (line,) = ax.plot(
                df_M["purification_level"], df_M["gamma"],
                linestyle="-",
                marker=_mk(i),
                linewidth=2.5,
                markersize=12,
                alpha=0.85,
                color=colors[i % len(colors)],
                label=f"$M={M}$",
            )
            
            handles.append(line)
            labels.append(f"$M={M}$")

        # Formatting
        ax.set_xlabel(r"Purification Rounds, $\ell$", fontsize=40)
        ax.set_ylabel(r"Logical Error, $\gamma_L$", fontsize=40)
        # ax.set_title(f"Gamma vs Purification Level\n({title_str}, p = {actual_p:.2f})", fontsize=32)
        ax.set_yscale("log")

        # Set x-axis limits and ticks based on available data
        if not df_gamma.empty:
            min_ell = df_gamma["purification_level"].min()
            max_ell = df_gamma["purification_level"].max()
            ax.set_xlim(min_ell - 0.1, max_ell + 0.1)
            
            # Set integer ticks for purification levels
            ell_values = sorted(df_gamma["purification_level"].unique())
            ax.set_xticks(ell_values)

        # Y-axis formatting
        ax.set_ylim(1e-2, 1.0)
        
        # Add legend
        if handles:
            ax.legend(handles, labels, fontsize=18, loc="best", frameon=False)

        plt.tight_layout()

        filename = f"gamma_vs_purification_level_{noise_type}_p{actual_p:.2f}.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved {filename}")
        return str(filepath)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    







    
    def generate_all_plots(self, save_format: str = 'pdf') -> Dict[str, Optional[str]]:
        """Generate all iterative purification plots."""
        print("\n" + "="*70)
        print("GENERATING ITERATIVE PURIFICATION FIGURES")
        print("="*70)
        
        plots = {}
        
        # Fidelity vs iterations (single p value)
        print("\n1. Fidelity vs iterations (representative p)...")
        plots['fidelity_vs_iter_depol'] = self.plot_fidelity_vs_iterations('depolarizing', save_format)
        plots['fidelity_vs_iter_dephase'] = self.plot_fidelity_vs_iterations('dephasing', save_format)
        
        # Fidelity vs iterations (2x2 grid of p values)
        print("\n2. Fidelity vs iterations (2x2 grid of p values)...")
        plots['fidelity_vs_iter_multi_depol'] = self.plot_fidelity_vs_iterations_multiple_p('depolarizing', save_format)
        plots['fidelity_vs_iter_multi_dephase'] = self.plot_fidelity_vs_iterations_multiple_p('dephasing', save_format)
        plots['fidelity_vs_iter_multi_depol_hotfix'] = self.plot_fidelity_vs_iterations_with_extended_hotfix('depolarizing', save_format)
        plots['fidelity_vs_iter_multi_dephase_hotfix'] = self.plot_fidelity_vs_iterations_with_extended_hotfix('dephasing', save_format)
        
        # Fidelity vs iterations (selected p values: 0.1, 0.5, 0.7)
        print("\n3. Fidelity vs iterations (selected p values: 0.1, 0.5, 0.7)...")
        plots['fidelity_vs_iter_selected_depol'] = self.plot_fidelity_vs_iterations_selected_p('depolarizing', save_format)
        plots['fidelity_vs_iter_selected_dephase'] = self.plot_fidelity_vs_iterations_selected_p('dephasing', save_format)
        
        # Final fidelity vs p (threshold-like plots)
        print("\n4. Final fidelity vs physical error rate...")
        plots['final_fidelity_vs_p_depol'] = self.plot_final_fidelity_vs_p('depolarizing', save_format)
        plots['final_fidelity_vs_p_dephase'] = self.plot_final_fidelity_vs_p('dephasing', save_format)
        
        # Combined fidelity plot (2x2 grid: M=1,5 vs depol,dephasing)
        print("\n5. Combined fidelity plot (2x2 grid: M=1,5 vs depol,dephasing)...")
        plots['fidelity_combined_M'] = self.plot_fidelity_combined_M(save_format)
        
        # Gamma vs p plots
        print("\n6. Gamma vs p (combined depol + dephase)...")
        plots['gamma_vs_p_combined'] = self.plot_gamma_vs_p_combined(method="fit", save_format=save_format)
        
        # Gamma vs p plots (difference method)
        print("\n7. Gamma vs p using difference method F(t=0) - F(t=1)...")
        plots["gamma_firststep_vs_p_depol"] = self.plot_gamma_from_first_step_vs_p("depolarizing", save_format)
        plots["gamma_firststep_vs_p_dephase"] = self.plot_gamma_from_first_step_vs_p("dephasing", save_format)
        plots["gamma_firstep_vs_p_depol_M1_M5_insets"] = self.plot_gamma_from_first_step_vs_p_M1_M5_grid_with_insets("depolarizing", save_format)
        plots["gamma_firstep_vs_p_dephase_M1_M5_insets"] = self.plot_gamma_from_first_step_vs_p_M1_M5_grid_with_insets("dephasing", save_format)
        
        # Final fidelity vs p for all ℓ
        print("\n8. Final fidelity vs p for all purification levels ℓ...")
        plots["final_fidelity_vs_p_all_l_depol"] = self.plot_final_fidelity_vs_p_all_l("depolarizing", save_format)
        plots["final_fidelity_vs_p_all_l_dephase"] = self.plot_final_fidelity_vs_p_all_l("dephasing", save_format)
        
        # Combined fidelity and gamma plot (2x2 grid)
        print("\n9. Combined fidelity and gamma plot (2x2 grid)...")
        plots["combined_fidelity_gamma_2x2_depol"] = self.plot_combined_fidelity_and_gamma_2x2_grid("depolarizing", save_format)
        plots["combined_fidelity_gamma_2x2_dephase"] = self.plot_combined_fidelity_and_gamma_2x2_grid("dephasing", save_format)
        
        # Fidelity vs purification rounds for iteration==1 (2x2 grid of p values)
        print("\n10. Fidelity vs purification rounds for iteration==1 (2x2 grid of p values)...")
        # plots["fidelity_vs_purification_rounds_iter1_depol"] = self.plot_fidelity_vs_purification_rounds_iteration1("depolarizing", save_format)
        plots["fidelity_vs_purification_rounds_iter1_dephase"] = self.plot_fidelity_vs_purification_rounds_iteration1("dephasing", save_format)
        
        # Gamma vs purification level for fixed p (cross-section of gamma vs p)
        print("\n11. Gamma vs purification level for fixed p (cross-section of gamma vs p)...")
        plots["gamma_vs_purification_level_fixed_p_depol"] = self.plot_gamma_vs_purification_level_fixed_p("depolarizing", target_p=0.3, save_format=save_format)
        plots["gamma_vs_purification_level_fixed_p_dephase"] = self.plot_gamma_vs_purification_level_fixed_p("dephasing", target_p=0.3, save_format=save_format)

        
        
        
        # Summary
        successful = [name for name, path in plots.items() if path is not None]
        print(f"\n{len(successful)}/{len(plots)} plots generated successfully")
        print(f"Figures saved to: {self.figures_dir}")
        
        return plots


def main():
    """Main function."""
    import sys
    
    data_dir = "data/simulations_moreNoise"
    figures_dir = "figures/iterative_results"
    save_format = "pdf"
    
    if '--data-dir' in sys.argv:
        idx = sys.argv.index('--data-dir')
        if idx + 1 < len(sys.argv):
            data_dir = sys.argv[idx + 1]
    
    if '--figures-dir' in sys.argv:
        idx = sys.argv.index('--figures-dir')
        if idx + 1 < len(sys.argv):
            figures_dir = sys.argv[idx + 1]
    
    if '--format' in sys.argv:
        idx = sys.argv.index('--format')
        if idx + 1 < len(sys.argv):
            save_format = sys.argv[idx + 1]
    
    plotter = IterativePurificationPlotter(data_dir, figures_dir)
    plots = plotter.generate_all_plots(save_format)
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    
    return plots


if __name__ == "__main__":
    main()