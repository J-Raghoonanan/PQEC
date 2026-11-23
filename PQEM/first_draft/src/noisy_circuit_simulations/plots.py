"""Publication figures for the circuit-level noisy PQES datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import NoiseFamily, PurificationSchedule


COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
]
MARKERS = ["o", "s", "^", "D", "P", "X"]
LINESTYLES = {
    PurificationSchedule.NONE.value: ":",
    PurificationSchedule.INTERLEAVED.value: "-",
    PurificationSchedule.END_ONLY.value: "--",
}
SCHEDULE_LABELS = {
    PurificationSchedule.NONE.value: "No PQES",
    PurificationSchedule.INTERLEAVED.value: "Interleaved",
    PurificationSchedule.END_ONLY.value: "End-only",
}


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 8,
            "figure.titlesize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
        }
    )


def _select_evenly(values: Iterable[float], maximum: int) -> List[float]:
    unique = sorted(set(float(value) for value in values))
    if len(unique) <= maximum:
        return unique
    indices = np.linspace(0, len(unique) - 1, maximum).round().astype(int)
    return [unique[index] for index in indices]


def _closest_rows(frame: pd.DataFrame, column: str, value: float) -> pd.DataFrame:
    return frame[np.isclose(frame[column].astype(float), float(value))]


class NoisyCircuitPlotter:
    """Generate analogues of the active rho2 plot families."""

    def __init__(
        self,
        data_dir: Union[Path, str] = Path("data/noisy_circuit_simulations"),
        figures_dir: Union[Path, str] = Path(
            "figures/noisy_circuit_simulations"
        ),
    ) -> None:
        _configure_matplotlib()
        self.data_dir = Path(data_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.steps = pd.read_csv(self.data_dir / "steps.csv")
        self.finals = pd.read_csv(self.data_dir / "finals.csv")

    def _save(self, figure: plt.Figure, stem: str) -> Dict[str, Path]:
        pdf_path = self.figures_dir / f"{stem}.pdf"
        png_path = self.figures_dir / f"{stem}.png"
        figure.savefig(pdf_path, bbox_inches="tight")
        figure.savefig(png_path, dpi=220, bbox_inches="tight")
        plt.close(figure)
        return {"pdf": pdf_path, "png": png_path}

    @staticmethod
    def _schedule_depth_curve(
        frame: pd.DataFrame,
        *,
        M: int,
        p: float,
        schedule: str,
        fidelity_column: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        subset = frame[(frame["M"] == M)]
        subset = _closest_rows(subset, "p", p)
        baseline = subset[subset["schedule"] == PurificationSchedule.NONE.value]
        scheduled = subset[subset["schedule"] == schedule]
        scheduled = scheduled.sort_values("purification_level")

        x: List[int] = []
        y: List[float] = []
        if not baseline.empty:
            x.append(0)
            y.append(float(baseline.iloc[0][fidelity_column]))
        for _, row in scheduled.iterrows():
            x.append(int(row["purification_level"]))
            y.append(float(row[fidelity_column]))
        return np.asarray(x), np.asarray(y)

    def plot_fidelity_vs_purification_rounds_iteration1(
        self,
        noise: str,
    ) -> Dict[str, Path]:
        """Legacy-style fidelity versus ell panels at iteration one."""
        frame = self.steps[
            (self.steps["noise"] == noise) & (self.steps["iteration"] == 1)
        ].copy()
        p_values = _select_evenly(frame["p"], 4)
        M_values = sorted(frame["M"].unique())

        figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.5), squeeze=False)
        for panel, axis in enumerate(axes.flat):
            if panel >= len(p_values):
                axis.set_visible(False)
                continue
            p = p_values[panel]
            for M_index, M in enumerate(M_values):
                for schedule in (
                    PurificationSchedule.INTERLEAVED.value,
                    PurificationSchedule.END_ONLY.value,
                ):
                    x, y = self._schedule_depth_curve(
                        frame,
                        M=int(M),
                        p=p,
                        schedule=schedule,
                        fidelity_column="fidelity",
                    )
                    if len(x) == 0:
                        continue
                    axis.plot(
                        x,
                        y,
                        color=COLORS[M_index % len(COLORS)],
                        marker=MARKERS[M_index % len(MARKERS)],
                        linestyle=LINESTYLES[schedule],
                        label=f"M={M}, {SCHEDULE_LABELS[schedule]}",
                    )

            axis.axhline(0.5, color="#999999", linestyle="-.", linewidth=0.8)
            axis.set_title(f"p = {p:g}")
            axis.set_xlabel(r"Purification levels per cycle, $\ell$")
            axis.set_ylabel("Signed-estimator fidelity")
            axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)

        figure.suptitle(
            f"Noisy-circuit PQES: iteration-one purification scan ({noise})"
        )
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncols=3,
            frameon=False,
        )
        figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.96))
        return self._save(
            figure,
            f"fidelity_vs_purification_rounds_iter1_{noise}",
        )

    def plot_final_fidelity_grid(self) -> Dict[str, Path]:
        """Combined final fidelity versus p for both modeled noise families."""
        noise_values = [
            noise.value
            for noise in NoiseFamily
            if noise.value in set(self.finals["noise"])
        ]
        all_M = sorted(self.finals["M"].unique())
        M_values = all_M if len(all_M) <= 2 else [all_M[0], all_M[-1]]
        figure, axes = plt.subplots(
            len(noise_values),
            len(M_values),
            figsize=(5.2 * len(M_values), 3.8 * len(noise_values)),
            squeeze=False,
        )

        for row, noise in enumerate(noise_values):
            for column, M in enumerate(M_values):
                axis = axes[row, column]
                subset = self.finals[
                    (self.finals["noise"] == noise)
                    & (self.finals["M"] == M)
                ].copy()
                baseline = subset[
                    subset["schedule"] == PurificationSchedule.NONE.value
                ].sort_values("p")
                if not baseline.empty:
                    axis.plot(
                        baseline["p"],
                        baseline["fidelity_final"],
                        color="#666666",
                        linestyle=":",
                        marker="o",
                        label="No PQES",
                    )

                ell_values = sorted(
                    value
                    for value in subset["purification_level"].unique()
                    if value > 0
                )
                for ell_index, ell in enumerate(ell_values):
                    for schedule in (
                        PurificationSchedule.INTERLEAVED.value,
                        PurificationSchedule.END_ONLY.value,
                    ):
                        curve = subset[
                            (subset["purification_level"] == ell)
                            & (subset["schedule"] == schedule)
                        ].sort_values("p")
                        if curve.empty:
                            continue
                        axis.plot(
                            curve["p"],
                            curve["fidelity_final"],
                            color=COLORS[ell_index % len(COLORS)],
                            marker=MARKERS[ell_index % len(MARKERS)],
                            linestyle=LINESTYLES[schedule],
                            label=(
                                rf"$\ell={int(ell)}$, "
                                f"{SCHEDULE_LABELS[schedule]}"
                            ),
                        )

                axis.axhline(
                    0.5,
                    color="#999999",
                    linestyle="-.",
                    linewidth=0.8,
                )
                axis.set_title(f"{noise.replace('_', ' ').title()}, M={M}")
                axis.set_xlabel("One-qubit location error, p")
                axis.set_ylabel("Final signed-estimator fidelity")
                axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        figure.suptitle("Final fidelity of noisy-gate PQES circuits")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncols=4,
            frameon=False,
            fontsize=8,
        )
        figure.tight_layout(rect=(0.0, 0.09, 1.0, 0.96))
        return self._save(figure, "final_fidelity_vs_p")

    def plot_fidelity_and_gamma(self, noise: str) -> Dict[str, Path]:
        """Legacy-style 2x2 fidelity trajectory and gamma figure."""
        frame = self.steps[self.steps["noise"] == noise].copy()
        M_values = sorted(frame["M"].unique())
        target_M = int(M_values[-1])
        nonzero_p = sorted(value for value in frame["p"].unique() if value > 0)
        p_values = _select_evenly(nonzero_p or frame["p"].unique(), 2)
        while len(p_values) < 2:
            p_values.append(p_values[-1])

        figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
        ell_values = sorted(frame["purification_level"].unique())
        for column, p in enumerate(p_values[:2]):
            axis = axes[0, column]
            subset = frame[frame["M"] == target_M]
            subset = _closest_rows(subset, "p", p)
            for ell in ell_values:
                schedules: Sequence[str]
                if ell == 0:
                    schedules = [PurificationSchedule.NONE.value]
                else:
                    schedules = [
                        PurificationSchedule.INTERLEAVED.value,
                        PurificationSchedule.END_ONLY.value,
                    ]
                for schedule in schedules:
                    curve = subset[
                        (subset["purification_level"] == ell)
                        & (subset["schedule"] == schedule)
                    ].sort_values("iteration")
                    if curve.empty:
                        continue
                    x = np.concatenate(([0], curve["iteration"].to_numpy()))
                    y = np.concatenate(([1.0], curve["fidelity"].to_numpy()))
                    label = (
                        "No PQES"
                        if ell == 0
                        else rf"$\ell={int(ell)}$, {SCHEDULE_LABELS[schedule]}"
                    )
                    axis.plot(
                        x,
                        y,
                        color=(
                            "#666666"
                            if ell == 0
                            else COLORS[(int(ell) - 1) % len(COLORS)]
                        ),
                        marker=(
                            "o"
                            if ell == 0
                            else MARKERS[(int(ell) - 1) % len(MARKERS)]
                        ),
                        linestyle=LINESTYLES[schedule],
                        label=label,
                    )
            axis.axhline(0.5, color="#999999", linestyle="-.", linewidth=0.8)
            axis.set_title(f"M={target_M}, p={p:g}")
            axis.set_xlabel("Circuit/PQES cycle, t")
            axis.set_ylabel("Signed-estimator fidelity")
            axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)

        iteration_one = frame[frame["iteration"] == 1].copy()
        iteration_one["gamma"] = 1.0 - iteration_one["fidelity"]
        gamma_M_values = (
            M_values if len(M_values) <= 2 else [M_values[0], M_values[-1]]
        )
        for column, M in enumerate(gamma_M_values[:2]):
            axis = axes[1, column]
            subset = iteration_one[iteration_one["M"] == M]
            baseline = subset[
                subset["schedule"] == PurificationSchedule.NONE.value
            ].sort_values("p")
            if not baseline.empty:
                axis.plot(
                    baseline["p"],
                    baseline["gamma"],
                    color="#666666",
                    marker="o",
                    linestyle=":",
                    label="No PQES",
                )
            for ell_index, ell in enumerate(
                value for value in ell_values if value > 0
            ):
                for schedule in (
                    PurificationSchedule.INTERLEAVED.value,
                    PurificationSchedule.END_ONLY.value,
                ):
                    curve = subset[
                        (subset["purification_level"] == ell)
                        & (subset["schedule"] == schedule)
                    ].sort_values("p")
                    if curve.empty:
                        continue
                    axis.plot(
                        curve["p"],
                        curve["gamma"],
                        color=COLORS[ell_index % len(COLORS)],
                        marker=MARKERS[ell_index % len(MARKERS)],
                        linestyle=LINESTYLES[schedule],
                        label=(
                            rf"$\ell={int(ell)}$, "
                            f"{SCHEDULE_LABELS[schedule]}"
                        ),
                    )
            axis.set_title(f"Iteration-one logical error, M={M}")
            axis.set_xlabel("One-qubit location error, p")
            axis.set_ylabel(r"$\gamma_L = 1-F(t=1)$")
            axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        figure.suptitle(f"Fidelity and logical error under {noise} gate noise")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncols=4,
            frameon=False,
            fontsize=8,
        )
        figure.tight_layout(rect=(0.0, 0.09, 1.0, 0.96))
        return self._save(figure, f"combined_fidelity_gamma_{noise}")

    def plot_sampling_overhead(self) -> Dict[str, Path]:
        """Signed-estimator sampling overhead, absent from the old figures."""
        noise_values = sorted(self.finals["noise"].unique())
        target_M = int(self.finals["M"].max())
        source = self.finals[
            (self.finals["M"] == target_M)
            & (self.finals["purification_level"] > 0)
        ]
        ell_values = sorted(source["purification_level"].unique())
        figure, axes = plt.subplots(
            len(ell_values),
            len(noise_values),
            figsize=(5.2 * len(noise_values), 3.1 * len(ell_values)),
            squeeze=False,
        )

        for row, ell in enumerate(ell_values):
            for column, noise in enumerate(noise_values):
                axis = axes[row, column]
                subset = source[
                    (source["noise"] == noise)
                    & (source["purification_level"] == ell)
                ]
                for schedule in (
                    PurificationSchedule.INTERLEAVED.value,
                    PurificationSchedule.END_ONLY.value,
                ):
                    curve = subset[
                        (subset["purification_level"] == ell)
                        & (subset["schedule"] == schedule)
                    ].copy()
                    if "log10_sampling_overhead_final" in curve:
                        overhead_column = "log10_sampling_overhead_final"
                        curve = curve[
                            np.isfinite(curve[overhead_column])
                        ].sort_values("p")
                    else:
                        overhead_column = "sampling_overhead_final"
                        curve = curve[
                            np.isfinite(curve[overhead_column])
                            & (curve[overhead_column] > 0)
                        ].sort_values("p")
                    if curve.empty:
                        continue
                    axis.plot(
                        curve["p"],
                        curve[overhead_column],
                        color=COLORS[(int(ell) - 1) % len(COLORS)],
                        marker=MARKERS[(int(ell) - 1) % len(MARKERS)],
                        linestyle=LINESTYLES[schedule],
                        label=SCHEDULE_LABELS[schedule],
                    )
                if "log10_sampling_overhead_final" not in subset:
                    axis.set_yscale("log")
                axis.set_title(
                    f"{noise.replace('_', ' ').title()}, "
                    rf"$M={target_M},\,\ell={int(ell)}$"
                )
                axis.set_xlabel("One-qubit location error, p")
                if column == 0:
                    if "log10_sampling_overhead_final" in subset:
                        axis.set_ylabel(
                            r"$\log_{10}[1/|\mathrm{Tr}S|^2]$"
                        )
                    else:
                        axis.set_ylabel(
                            r"Sampling overhead, $1/|\mathrm{Tr}S|^2$"
                        )
                axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)

        figure.suptitle("Noisy SWAP-record sampling overhead")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncols=2,
            frameon=False,
        )
        figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.97))
        return self._save(figure, "sampling_overhead_vs_p")

    def plot_interleaving_advantage(self) -> Dict[str, Path]:
        """Heatmaps of final-fidelity advantage at matched PQES depth."""
        purified = self.finals[
            self.finals["purification_level"] > 0
        ].pivot_table(
            index=["noise", "M", "p", "purification_level"],
            columns="schedule",
            values="fidelity_final",
        )
        purified["delta"] = (
            purified[PurificationSchedule.INTERLEAVED.value]
            - purified[PurificationSchedule.END_ONLY.value]
        )
        purified = purified.reset_index()

        noise_values = sorted(purified["noise"].unique())
        M_values = sorted(purified["M"].unique())
        p_values = sorted(purified["p"].unique())
        ell_values = sorted(purified["purification_level"].unique())
        maximum = float(np.nanmax(np.abs(purified["delta"])))

        figure, axes = plt.subplots(
            len(noise_values),
            len(M_values),
            figsize=(4.2 * len(M_values), 3.5 * len(noise_values)),
            squeeze=False,
        )
        image = None
        for row, noise in enumerate(noise_values):
            for column, M in enumerate(M_values):
                axis = axes[row, column]
                subset = purified[
                    (purified["noise"] == noise) & (purified["M"] == M)
                ]
                matrix = (
                    subset.pivot(
                        index="purification_level",
                        columns="p",
                        values="delta",
                    )
                    .reindex(index=ell_values, columns=p_values)
                    .to_numpy()
                )
                image = axis.imshow(
                    matrix,
                    aspect="auto",
                    cmap="RdBu",
                    vmin=-maximum,
                    vmax=maximum,
                )
                for y, ell in enumerate(ell_values):
                    for x, p in enumerate(p_values):
                        value = matrix[y, x]
                        displayed_value = 0.0 if abs(value) < 5e-4 else value
                        text_color = (
                            "white"
                            if abs(value) > 0.55 * maximum
                            else "#222222"
                        )
                        axis.text(
                            x,
                            y,
                            f"{displayed_value:+.3f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color=text_color,
                        )
                axis.set_xticks(range(len(p_values)))
                axis.set_xticklabels([f"{p:g}" for p in p_values])
                axis.set_yticks(range(len(ell_values)))
                axis.set_yticklabels([str(int(ell)) for ell in ell_values])
                axis.set_title(
                    f"{noise.replace('_', ' ').title()}, M={int(M)}"
                )
                axis.set_xlabel("One-qubit location error, p")
                if column == 0:
                    axis.set_ylabel(
                        r"Purification levels per cycle, $\ell$"
                    )

        if image is None:  # pragma: no cover - validated datasets are nonempty
            raise ValueError("No matched schedule data available")
        figure.suptitle("Matched-depth interleaving advantage")
        figure.subplots_adjust(
            left=0.07,
            right=0.88,
            bottom=0.09,
            top=0.90,
            wspace=0.28,
            hspace=0.38,
        )
        colorbar_axis = figure.add_axes([0.91, 0.18, 0.018, 0.64])
        colorbar = figure.colorbar(image, cax=colorbar_axis)
        colorbar.set_label(
            r"$F_{\mathrm{interleaved}}-F_{\mathrm{end\ only}}$"
        )
        return self._save(figure, "interleaving_advantage")

    def generate_all(self) -> Dict[str, Dict[str, Path]]:
        outputs: Dict[str, Dict[str, Path]] = {}
        for noise in sorted(self.steps["noise"].unique()):
            outputs[f"ell_{noise}"] = (
                self.plot_fidelity_vs_purification_rounds_iteration1(noise)
            )
            outputs[f"gamma_{noise}"] = self.plot_fidelity_and_gamma(noise)
        outputs["final_fidelity"] = self.plot_final_fidelity_grid()
        outputs["sampling_overhead"] = self.plot_sampling_overhead()
        outputs["interleaving_advantage"] = (
            self.plot_interleaving_advantage()
        )
        return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot circuit-level noisy PQES data"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/noisy_circuit_simulations"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures/noisy_circuit_simulations"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plotter = NoisyCircuitPlotter(args.data_dir, args.figures_dir)
    outputs = plotter.generate_all()
    for name, paths in outputs.items():
        print(f"{name}: {paths['pdf']}")


if __name__ == "__main__":
    main()
