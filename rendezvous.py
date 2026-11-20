"""Numerical model and plotting helpers for the paper's rendezvous experiments.

The chief starts at periapsis (mean anomaly zero). All controls are accelerations
in the radial, along-track, cross-track Hill frame. A zero-order hold freezes
the Yamanaka--Ankersen matrix at the start of each control interval.
"""

from __future__ import annotations

import itertools
import shutil
import time
from pathlib import Path

import cvxpy as cp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.linalg import expm
from scipy.spatial import ConvexHull

MU = 3.986e14
SEMI_MAJOR_AXIS = 7102.8e3
U_MAX = 5.0 / 100.0
INPUT_SET = np.array(list(itertools.product((-U_MAX, 0.0, U_MAX), repeat=3)))
POSITION_SCALE = np.array([500.0] * 3 + [5.0] * 3)
SUCCESS = {"optimal", "optimal_inaccurate"}


def sample_state(rng):
    """Draw a state uniformly from the initial-state box in the paper."""
    return np.r_[rng.uniform(-500.0, 500.0, 3), rng.uniform(-5.0, 5.0, 3)]


def solve_kepler(mean_anomaly, eccentricity, tolerance=1e-8):
    eccentric_anomaly = float(mean_anomaly)
    for _ in range(100):
        step = (eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly)
                - mean_anomaly) / (1.0 - eccentricity * np.cos(eccentric_anomaly))
        eccentric_anomaly -= step
        if abs(step) < tolerance:
            return eccentric_anomaly
    raise RuntimeError("Kepler equation failed to converge")


def discretize_dynamics(steps, horizon, eccentricity, semi_major_axis=SEMI_MAJOR_AXIS):
    """Return N exact ZOH pairs for piecewise-frozen YA continuous matrices."""
    if steps < 1 or horizon <= 0 or not 0 <= eccentricity < 1:
        raise ValueError("Require steps >= 1, horizon > 0, and eccentricity in [0,1)")
    dt = horizon / steps
    n = np.sqrt(MU / semi_major_axis**3)
    h = np.sqrt(MU * semi_major_axis * (1 - eccentricity**2))
    p = semi_major_axis * (1 - eccentricity**2)
    a_discrete = np.empty((steps, 6, 6))
    b_discrete = np.empty((steps, 6, 3))
    for k in range(steps):
        E = solve_kepler(n * k * dt, eccentricity)
        theta = 2 * np.arctan2(np.sqrt(1 + eccentricity) * np.sin(E / 2),
                               np.sqrt(1 - eccentricity) * np.cos(E / 2))
        radius = p / (1 + eccentricity * np.cos(theta))
        theta_dot = h / radius**2
        radius_dot = (MU / h) * eccentricity * np.sin(theta)
        theta_ddot = -2 * (radius_dot / radius) * theta_dot
        A = np.zeros((6, 6))
        A[:3, 3:] = np.eye(3)
        A[3, 0], A[3, 1], A[3, 4] = theta_dot**2 + 2 * MU / radius**3, theta_ddot, 2 * theta_dot
        A[4, 0], A[4, 1], A[4, 3] = -theta_ddot, theta_dot**2 - MU / radius**3, -2 * theta_dot
        A[5, 2] = -MU / radius**3
        B = np.zeros((6, 3))
        B[3:, :] = np.eye(3)
        block = np.zeros((9, 9))
        block[:6, :6], block[:6, 6:] = A, B
        transition = expm(block * dt)
        a_discrete[k], b_discrete[k] = transition[:6, :6], transition[:6, 6:]
    return a_discrete, b_discrete


def solve_convex(x0, steps, horizon, eccentricity, *, solver="ECOS", dynamics=None):
    """Solve the paper's finite-grid Mayer relaxation (exact terminal equality).

    Deliberately retain the paper-run CVXPY constraint structure, including
    the explicit cost state and hull inequalities, for close numerical parity.
    """
    x0 = np.asarray(x0, dtype=float)
    A, B = dynamics if dynamics is not None else discretize_dynamics(steps, horizon, eccentricity)
    hull = ConvexHull(INPUT_SET)
    A_u = hull.equations[:, :-1]
    b_u = -hull.equations[:, -1]
    x = cp.Variable((6, steps + 1))
    x_c = cp.Variable((1, steps + 1))
    u = cp.Variable((3, steps))
    nu = cp.Variable((1, steps))
    constraints = [x[:, 0] == x0, x_c[:, 0] == 0]
    for k in range(steps):
        constraints += [x[:, k + 1] == A[k] @ x[:, k] + B[k] @ u[:, k],
                        x_c[:, k + 1] == x_c[:, k] + horizon / steps * nu[:, k],
                        nu[:, k] >= 0,
                        nu[:, k] >= cp.norm1(u[:, k]),
                        nu[:, k] <= 3 * U_MAX,
                        A_u @ u[:, k] <= b_u]
    constraints.append(x[:, steps] == np.zeros(6))
    problem = cp.Problem(cp.Minimize(x_c[:, steps]), constraints)
    started = time.perf_counter()
    problem.solve(solver=solver, verbose=False)
    wall_time = time.perf_counter() - started
    return {"status": problem.status, "cost": problem.value,
            "x": x.value, "u": u.value, "nu": nu.value,
            "solver_time": problem.solver_stats.solve_time,
            "wall_time": wall_time, "steps": steps, "horizon": horizon}


def feasible(result):
    return (result["status"] in SUCCESS and result["cost"] is not None
            and np.isfinite(result["cost"]) and result["x"] is not None)


def bisect_horizon(x0, *, lower=100.0, upper=500.0, tolerance=1.0,
                   steps=200, eccentricity=0.7, solver="ECOS"):
    """Find the shortest feasible grid horizon within tolerance.

    The upper endpoint is checked explicitly; the lower endpoint need not be
    infeasible. The returned horizon is the feasible upper endpoint, including
    when no midpoint was feasible. All solve times, including endpoint checks,
    are in `wall_time`; setup of the sampling distribution is excluded.
    """
    if lower <= 0 or upper <= lower or tolerance <= 0:
        raise ValueError("Require 0 < lower < upper and tolerance > 0")
    started = time.perf_counter()
    history = []

    def check(t):
        try:
            result = solve_convex(x0, steps, t, eccentricity, solver=solver)
        except cp.SolverError:
            result = {"status": "solver_error", "cost": None, "x": None,
                      "solver_time": None, "wall_time": None}
        history.append({"horizon": t, "feasible": feasible(result),
                        "solver_time": result["solver_time"], "wall_time": result["wall_time"]})
        return result

    best = check(upper)
    if not feasible(best):
        raise RuntimeError("The specified upper endpoint is infeasible")
    lower_result = check(lower)
    if feasible(lower_result):
        return {"horizon": lower, "result": lower_result, "history": history,
                "wall_time": time.perf_counter() - started}
    while upper - lower > tolerance:
        mid = (lower + upper) / 2
        trial = check(mid)
        if feasible(trial):
            upper, best = mid, trial
        else:
            lower = mid
    return {"horizon": upper, "result": best, "history": history,
            "wall_time": time.perf_counter() - started}


def solve_milp(x0, steps, horizon, eccentricity, *, terminal_tolerance=1e-3,
               solver="GUROBI", time_limit=None):
    """Condensed discrete-input MILP with u_i in {-u_max,0,u_max}.

    Nonzero tolerance is normalized by [500 m, 500 m, 500 m, 5 m/s, ...].
    It makes the comparison approximate rather than an exact terminal match.
    """
    started = time.perf_counter()
    A, B = discretize_dynamics(steps, horizon, eccentricity)
    Phi = np.eye(6)
    gamma = [None] * steps
    for k in range(steps - 1, -1, -1):
        gamma[k] = Phi @ B[k]
        Phi = Phi @ A[k]
    plus = cp.Variable((3, steps), boolean=True)
    minus = cp.Variable((3, steps), boolean=True)
    terminal = Phi @ np.asarray(x0) + U_MAX * np.hstack(gamma) @ cp.vec(plus - minus, order="F")
    constraints = [plus + minus <= 1]
    if terminal_tolerance:
        constraints += [terminal <= terminal_tolerance * POSITION_SCALE,
                        terminal >= -terminal_tolerance * POSITION_SCALE]
    else:
        constraints.append(terminal == 0)
    problem = cp.Problem(cp.Minimize(horizon / steps * U_MAX * cp.sum(plus + minus)), constraints)
    options = {}
    if solver == "GUROBI":
        options["MIPGap"] = 1e-6
        if time_limit is not None:
            options["TimeLimit"] = time_limit
    elif solver == "SCIPY":
        options["scipy_options"] = {"method": "highs", "mip_rel_gap": 1e-6}
        if time_limit is not None:
            options["scipy_options"]["time_limit"] = time_limit
    problem.solve(solver=solver, verbose=False, **options)
    if plus.value is None or minus.value is None:
        raise RuntimeError(f"MILP has no incumbent: {problem.status}")
    u = U_MAX * (np.rint(plus.value) - np.rint(minus.value))
    x = np.empty((6, steps + 1))
    x[:, 0] = x0
    for k in range(steps):
        x[:, k + 1] = A[k] @ x[:, k] + B[k] @ u[:, k]
    return {"status": problem.status, "cost": problem.value, "x": x, "u": u,
            "solver_time": problem.solver_stats.solve_time,
            "wall_time": time.perf_counter() - started,
            "terminal_residual": np.max(np.abs(x[:, -1]) / POSITION_SCALE),
            "steps": steps, "horizon": horizon}


def mean_distance(u, normalized=True):
    """Paper's distance statistic: sum over N controls divided by N-1.

    The N-1 divisor follows the displayed manuscript formula. It is not the
    conventional average over the N control intervals; keep that distinction
    explicit if the manuscript definition changes.
    """
    u = np.asarray(u)
    if u.ndim != 2 or u.shape[0] != 3 or u.shape[1] < 2:
        raise ValueError("Expected a 3-by-N control array with N >= 2")
    # Cartesian product makes component-wise nearest-neighbor selection exact.
    nearest = U_MAX * np.clip(np.rint(u / U_MAX), -1, 1)
    distances = np.linalg.norm(u - nearest, axis=0)
    return float(distances.sum() / (u.shape[1] - 1) / (U_MAX if normalized else 1.0))


def _save(fig, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


PAPER_FIGURES = {
    "framework.pdf": "framework.pdf",  # supplied schematic
    "polytopes2.pdf": "polytopes2.pdf",  # Cartesian input set, Example 1
    "polytopes1.pdf": "polytopes1.pdf",  # Cross-polytope, Example 2
    "validation_states.pdf": "rendezvous_example/validation_states.pdf",
    "validation_trajectory.pdf": "rendezvous_example/validation_trajectory.pdf",
    "validation_inputs.pdf": "rendezvous_example/validation_inputs.pdf",
    "grid_size_tradeoff_mean_std.pdf": "rendezvous_example/grid_size_tradeoff_mean_std.pdf",
    "solver_time_hist.pdf": "rendezvous_example/solver_time_hist.pdf",
    "hist_avg_dist.pdf": "rendezvous_example/hist_avg_dist.pdf",
    "mc_trajectories_2d.pdf": "mc_trajectories_2d.pdf",
}


def set_paper_style():
    """Compact serif graphics sized for the paper's two-column layout."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "savefig.dpi": 300,
    })


def export_paper_figures(output_dir, figures_dir):
    """Copy complete generated figure set to the paths in main_v3.tex."""
    output_dir, figures_dir = Path(output_dir), Path(figures_dir)
    missing = [name for name in PAPER_FIGURES if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Generate all experiments before exporting: " + ", ".join(missing))
    for source, destination in PAPER_FIGURES.items():
        target = figures_dir / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_dir / source, target)
    return [figures_dir / dest for dest in PAPER_FIGURES.values()]


def _method_handles():
    return [Line2D([0], [0], color="k", lw=1.4, label="Convex Program"),
            Line2D([0], [0], color="k", lw=1.1, ls="--", label="MILP")]


def plot_validation(convex, milp, output_dir):
    """Validation states, Hill-frame trajectory, and controls on native grids."""
    set_paper_style()
    output_dir = Path(output_dir)
    tf = convex["horizon"]
    tc = np.linspace(0, tf, convex["steps"] + 1)
    tm = np.linspace(0, tf, milp["steps"] + 1)
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    ax.plot(convex["x"][1], convex["x"][0], color="C0", lw=1.4)
    ax.plot(milp["x"][1], milp["x"][0], color="C1", ls="--", lw=1.1)
    ax.scatter(convex["x"][1, 0], convex["x"][0, 0], color="forestgreen",
               s=25, label=r"$r(0)$", zorder=5)
    ax.scatter(0, 0, color="red", marker="x", s=35, label="Chief", zorder=5)
    ax.set(xlabel=r"$y$ (Along-track) [m]", ylabel=r"$x$ (Radial) [m]")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(handles=_method_handles() + ax.get_legend_handles_labels()[0],
              loc="best", fontsize=7)
    _save(fig, output_dir / "validation_trajectory.pdf")

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.7), sharex=True)
    for i in range(3):
        for ax, j, prefix in ((axes[0], i, "r"), (axes[1], i + 3, "v")):
            ax.plot(tc, convex["x"][j], color=f"C{i}", lw=1.4,
                    label=rf"${prefix}_{{{'xyz'[i]}}}$")
            ax.plot(tm, milp["x"][j], color=f"C{i}", ls="--", lw=1)
    axes[0].set_ylabel(r"$r\;[\mathrm{m}]$")
    axes[1].set_ylabel(r"$v\;[\mathrm{m/s}]$")
    axes[1].set_xlabel("Time [s]")
    for ax in axes:
        ax.axhline(0, color="k", ls=":", lw=0.7)
        ax.set_xlim(0, tf)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(ncol=3, loc="best", fontsize=7)
    fig.legend(handles=_method_handles(), loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.03), frameon=False)
    _save(fig, output_dir / "validation_states.pdf")

    fig, axes = plt.subplots(3, 1, figsize=(3.5, 2.7), sharex=True)
    for i, ax in enumerate(axes):
        ax.step(tc[:-1], convex["u"][i], where="post", color=f"C{i}", lw=1.4)
        ax.step(tm[:-1], milp["u"][i], where="post", color=f"C{i}", ls="--", lw=1)
        for level in (-U_MAX, 0, U_MAX):
            ax.axhline(level, color="k", ls=":", alpha=0.5, lw=0.7)
        ax.set_ylabel(rf"$u_{{{'xyz'[i]}}}\;[\mathrm{{m/s^2}}]$")
        ax.set_xlim(0, tf)
        ax.grid(True, linestyle=":", alpha=0.6)
    axes[-1].set_xlabel("Time [s]")
    fig.legend(handles=_method_handles(), loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.03), frameon=False)
    _save(fig, output_dir / "validation_inputs.pdf")


def plot_grid(rows, output):
    set_paper_style()
    n = np.array([int(r["N"]) for r in rows])
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.8), sharex=True)
    for ax, prefix, label, color in ((axes[0], "solver_time", "Solver time [s]", "C0"),
                                     (axes[1], "normalized_distance", r"$\bar{d}(u)$", "darkred")):
        mean = np.array([float(r[f"{prefix}_mean_s" if prefix == "solver_time" else f"{prefix}_mean"]) for r in rows])
        std = np.array([float(r[f"{prefix}_std_s" if prefix == "solver_time" else f"{prefix}_std"]) for r in rows])
        ax.plot(n, mean, "o-", color=color, markersize=3, lw=1.2, label="Mean")
        ax.fill_between(n, np.maximum(0, mean - std), mean + std,
                        color=color, alpha=0.2, label=r"$\pm 1$ std. dev.")
        ax.set_ylabel(label)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=7)
    axes[-1].set_xlabel(r"Grid size $N$")
    _save(fig, output)


def plot_monte_carlo(rows, trajectories, output_dir):
    set_paper_style()
    output_dir = Path(output_dir)
    for column, label, name in (("solver_time_s", "Solver time [s]", "solver_time_hist.pdf"),
                                ("normalized_mean_distance", r"$\bar{d}(u)$", "hist_avg_dist.pdf")):
        values = np.array([float(row[column]) for row in rows])
        fig, ax = plt.subplots(figsize=(3.5, 1.5))
        ax.hist(values, bins=20, color="C1" if column == "solver_time_s" else "C0", alpha=0.75)
        ax.axvline(values.mean(), color="k", ls="--", lw=1, label=f"Mean = {values.mean():.5g}")
        ax.axvline(np.median(values), color="k", ls=":", lw=1, label=f"Median = {np.median(values):.5g}")
        ax.set(xlabel=label, ylabel="Count")
        ax.legend(fontsize=7)
        ax.grid(True, linestyle=":", alpha=0.6)
        _save(fig, output_dir / name)
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))
    for state, color in zip(trajectories, colors):
        ax.plot(state[1], state[0], color=color, lw=0.8, alpha=0.85)
        ax.scatter(state[1, 0], state[0, 0], color="forestgreen", s=7, alpha=0.7)
    ax.scatter([], [], color="forestgreen", s=22, label=r"$r(0)$")
    ax.scatter(0, 0, c="red", marker="x", s=35, label="Chief")
    ax.set(xlabel=r"$y$ (Along-track) [m]", ylabel=r"$x$ (Radial) [m]")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    _save(fig, output_dir / "mc_trajectories_2d.pdf")


def plot_relaxation_examples(output_dir):
    """Make the two side-by-side input-set figures referenced by main_v3.tex."""
    set_paper_style()
    output_dir = Path(output_dir)
    examples = {
        "polytopes1.pdf": np.array([[0., 0.], [1., 0.], [-1., 0.], [0., 1.], [0., -1.]]),
        "polytopes2.pdf": np.array(list(itertools.product((-1., 0., 1.), repeat=2))),
    }
    for name, points in examples.items():
        upper = np.max(np.abs(points).sum(axis=1))
        hull = ConvexHull(points)
        fig = plt.figure(figsize=(7.0, 2.7))
        ax = fig.add_subplot(121)
        ax.fill(points[hull.vertices, 0], points[hull.vertices, 1],
                color="lightsteelblue", alpha=0.45, label=r"$\mathrm{conv}(\mathcal{U})$")
        ax.plot(*np.vstack([points[hull.vertices], points[hull.vertices[0]]]).T,
                color="C0", lw=0.8)
        ax.scatter(points[:, 0], points[:, 1], c="firebrick", s=17, label=r"$\mathcal{U}$")
        ax.set(xlabel=r"$u_1$", ylabel=r"$u_2$", xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
        ax.set_aspect("equal")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

        ax = fig.add_subplot(122, projection="3d")
        grid = np.linspace(-1, 1, 60)
        u1, u2 = np.meshgrid(grid, grid)
        allowed = ((np.abs(u1) + np.abs(u2) <= 1)
                   if name == "polytopes1.pdf" else np.ones_like(u1, dtype=bool))
        lower = np.where(allowed, np.abs(u1) + np.abs(u2), np.nan)
        top = np.where(allowed, upper, np.nan)
        ax.plot_surface(u1, u2, lower, cmap="Reds", alpha=0.7, edgecolor="none")
        ax.plot_surface(u1, u2, top, color="royalblue", alpha=0.3, edgecolor="none")
        ax.scatter(points[:, 0], points[:, 1], np.abs(points).sum(axis=1), c="black", s=10)
        ax.set(xlabel=r"$u_1$", ylabel=r"$u_2$", zlabel=r"$\mu$",
               zlim=(0, upper + 0.2), xlim=(-1, 1), ylim=(-1, 1))
        ax.view_init(elev=22, azim=45)
        ax.legend(handles=[Line2D([0], [0], color="firebrick", label=r"$\mu=\|u\|_1$"),
                           Line2D([0], [0], color="royalblue", label=r"$\mu=\bar u$")],
                  loc="upper right", fontsize=7)
        _save(fig, output_dir / name)


def plot_framework(source_png, output_pdf):
    """Package the supplied framework schematic as the PDF TeX expects."""
    image = plt.imread(source_png)
    width, height = image.shape[1], image.shape[0]
    fig, ax = plt.subplots(figsize=(7, 7 * height / width))
    ax.imshow(image)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
