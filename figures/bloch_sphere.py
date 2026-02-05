"""
bloch_depolarization_plus_pqec_svg.py

Bloch-sphere schematic:
- Red: original (pure) Bloch vector r0 on the unit sphere.
- Blue: depolarized examples: random vectors inside an inner sphere of radius alpha.
- Show one example vector growth under one PQEC (SWAP purification) round:
      r -> (4/(3+|r|^2)) r
Saves as SVG for manuscript use.

Edits vs your base version:
- The three labels are spatially separated (offset in 3D using a local perpendicular basis).
- The PQEC (green) arrow is made clearly visible by a small azimuthal rotation about a
  perpendicular axis (so it does not lie exactly on top of the red arrow).
  The dashed "growth" guide still connects r_dep -> r_pqec along the original ray.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def _set_equal_3d_axes(ax):
    """Ensure equal scaling on 3D axes."""
    ax.set_box_aspect((1, 1, 1))


def pqec_map(r: np.ndarray) -> np.ndarray:
    """Your manuscript map: r -> (4/(3+|r|^2)) r."""
    r = np.asarray(r, dtype=float)
    rn2 = float(np.dot(r, r))
    return (4.0 / (3.0 + rn2)) * r


def sample_vectors_in_ball(n: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    """
    Uniform-ish samples inside a 3D ball using Gaussian directions + radius^(1/3).
    Returns array shape (n, 3).
    """
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    u = rng.random(n) ** (1.0 / 3.0)  # radial distribution for uniform ball
    return (radius * u)[:, None] * v


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _perp_basis(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a unit vector u, return two unit perpendicular vectors (e1, e2)
    such that {u, e1, e2} is right-handed (approximately).
    """
    u = _unit(u)
    # pick a helper not parallel to u
    a = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(u, a))) > 0.90:
        a = np.array([0.0, 1.0, 0.0])
    e1 = _unit(np.cross(u, a))
    e2 = _unit(np.cross(u, e1))
    return e1, e2


def _rot_about_axis(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    Rodrigues rotation: rotate vector v about (unit) axis by angle_rad.
    """
    v = np.asarray(v, dtype=float)
    k = _unit(np.asarray(axis, dtype=float))
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return v * c + np.cross(k, v) * s + k * float(np.dot(k, v)) * (1.0 - c)


def save_bloch_depol_plus_pqec_svg(
    out_svg: str = "bloch_depol_pqec.svg",
    *,
    alpha: float = 0.80,                 # depolarization contraction radius
    r0: tuple[float, float, float] = (0.55, 0.25, 0.75),  # direction for the "original" vector
    n_samples: int = 10,                # how many blue sample vectors
    seed: int = 7,
    elev: float = 18.0,
    azim: float = 35.0,
) -> str:
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0,1].")

    # normalize r0 to unit length (pure state direction)
    r0 = np.array(r0, dtype=float)
    r0 /= np.linalg.norm(r0)

    # depolarized example for the “growth” arrow
    r_dep = alpha * r0
    r_pqec = pqec_map(r_dep)

    # Build a local perpendicular basis for label offsets + tiny rotation
    e1, e2 = _perp_basis(r0)

    # Make PQEC arrow clearly visible: rotate it slightly around e2 (perp to r0)
    # (small enough to read as "same direction", but large enough to not overlap visually)
    pqec_angle = np.deg2rad(8.0)
    r_pqec_vis = _rot_about_axis(r_pqec, axis=e2, angle_rad=pqec_angle)

    # random blue examples inside inner sphere (radius alpha)
    rng = np.random.default_rng(seed)
    samples = sample_vectors_in_ball(n_samples, radius=alpha, rng=rng)

    fig = plt.figure(figsize=(6.4, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    # --- outer Bloch sphere wireframe (unit)
    u = np.linspace(0, 2 * np.pi, 120)
    v = np.linspace(0, np.pi, 80)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, rstride=6, cstride=6, linewidth=0.6, alpha=0.25)

    # --- inner sphere (depolarized radius alpha) as faint surface
    ax.plot_surface(alpha * xs, alpha * ys, alpha * zs, linewidth=0, alpha=0.08)

    # --- axes
    L = 1.50
    ax.plot([0, L], [0, 0], [0, 0], color="black", linewidth=1.2)
    ax.plot([0, 0], [0, L], [0, 0], color="black", linewidth=1.2)
    ax.plot([0, 0], [0, 0], [0, L], color="black", linewidth=1.2)
    # ax.text(L, 0, 0, r"$x$", fontsize=14, ha="left", va="center")
    # ax.text(0, L, 0, r"$y$", fontsize=14, ha="left", va="center")
    # ax.text(0, 0, L, r"$z$", fontsize=14, ha="left", va="center")
    

    # --- blue sample depolarized vectors inside inner sphere
    ax.quiver(
        np.zeros(n_samples), np.zeros(n_samples), np.zeros(n_samples),
        samples[:, 0], samples[:, 1], samples[:, 2],
        color="blue",
        alpha=0.35,
        linewidth=0.8,
        arrow_length_ratio=0.08,
        length=1.0,
        normalize=False,
    )

    # --- original (pure) vector: RED (per your latest preference)
    # ax.quiver(
    #     0, 0, 0,
    #     r0[0], r0[1], r0[2],
    #     color="red",
    #     linewidth=2.6,
    #     arrow_length_ratio=0.10,
    #     length=1.0,
    #     normalize=False,
    # )

    # --- one depolarized instance along r0 direction: BLUE (endpoint is r_dep)
    # ax.quiver(
    #     0, 0, 0,
    #     r_dep[0], r_dep[1], r_dep[2],
    #     color="blue",
    #     linewidth=2.3,
    #     arrow_length_ratio=0.10,
    #     length=1.0,
    #     normalize=False,
    # )

    # --- PQEC-updated vector: GREEN (slightly rotated for visibility)
    # ax.quiver(
    #     0, 0, 0,
    #     r_pqec_vis[0], r_pqec_vis[1], r_pqec_vis[2],
    #     color="green",
    #     linewidth=2.6,
    #     arrow_length_ratio=0.10,
    #     length=1.0,
    #     normalize=False,
    # )

    # --- show “growth” under PQEC as a dashed guide along the ORIGINAL ray (r_dep -> r_pqec)
    # This keeps the physics mapping clear even though the visual PQEC arrow is slightly rotated.
    # ax.plot(
    #     [r_dep[0], r_pqec[0]],
    #     [r_dep[1], r_pqec[1]],
    #     [r_dep[2], r_pqec[2]],
    #     color="green",
    #     linewidth=2.0,
    #     linestyle="--",
    #     alpha=0.9,
    # )

    # -----------------------
    # Label placement (separated in 3D)
    # -----------------------
    # Put each label on a distinct nearby offset direction so they don't stack.
    # (Offsets are in world coordinates; scale tuned for typical figure size.)
    off0 = 0.10 * e1 + 0.02 * e2
    offd = -0.10 * e1 + 0.02 * e2
    offp = 0.02 * e1 + 0.12 * e2

    p0 = 1.05 * r0 + off0
    pd = 1.05 * r_dep + offd
    pp = 1.05 * r_pqec_vis + offp

    # ax.text(p0[0], p0[1], p0[2], r"$\vec r_0$", fontsize=14, color="red")
    # ax.text(pd[0], pd[1], pd[2], r"$\vec r_{\rm dep}$", fontsize=14, color="blue")
    # ax.text(pp[0], pp[1], pp[2], r"$\vec r_{\rm PQEC}$", fontsize=14, color="green")

    # --- annotation with your exact map
    # ax.text2D(
    #     0.02, 0.98,
    #     r"Depolarization: $\vec r \mapsto \alpha \vec r$"
    #     + "\n"
    #     + r"PQEC: $\vec r \mapsto \frac{4}{3+|\vec r|^2}\,\vec r$",
    #     transform=ax.transAxes,
    #     fontsize=14,
    #     va="top",
    #     ha="left",
    #     color="black",
    # )

    # view/limits
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    _set_equal_3d_axes(ax)

    # clean ticks/panes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    try:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
    except Exception:
        pass

    # --- remove 3D bounding box / panes completely ---
    ax.set_axis_off()
    
    fig.tight_layout()
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_svg


if __name__ == "__main__":
    path = save_bloch_depol_plus_pqec_svg(
        out_svg="figures/diagrams/bloch_depol_pqec.svg",
        alpha=0.80,
        r0=(0.35, -0.25, 0.30),
        n_samples=0,
        seed=7,
        elev=18,
        azim=35,
    )
    print("Saved:", path)
