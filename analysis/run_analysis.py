#!/usr/bin/env python3
"""Monte-Carlo analysis of the PyroScout navigation stack.

Runs the closed-loop navigator over hundreds of randomized episodes and
aggregates the results into CSV tables and figures:

* **Baseline study** — 100 seeds on the standard three-room scenario:
  success rate, time-to-rescue, path efficiency vs an A* oracle, collision
  count, victim-estimate error, and map-coverage growth.
* **Sensor sensitivity sweeps** — degrade one sensor parameter at a time
  (thermal bearing/intensity noise, LIDAR noise, LIDAR beam count) and
  measure how success rate and time-to-rescue respond.
* **Generalization study** — re-sample the victim's position uniformly over
  the reachable arena to check the stack is not tuned to one goal location.

Usage
-----
    python analysis/run_analysis.py              # full study (~10 min on 4 cores)
    python analysis/run_analysis.py --quick      # reduced episode counts (smoke test)
    python analysis/run_analysis.py --reuse      # regenerate figures from existing CSV
    python analysis/run_analysis.py --jobs 8     # parallel workers
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from pyroscout.control import PurePursuit
from pyroscout.mapping import OccupancyGrid
from pyroscout.navigator import Navigator, NavState
from pyroscout.planning import astar
from pyroscout.robot import DiffDriveRobot
from pyroscout.scenarios import search_rescue_world
from pyroscout.sensors import Lidar2D, ThermalSensor
from pyroscout.world import HeatSource, World

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIGURES_DIR = os.path.join(HERE, "figures")

DT = 0.1
MAX_STEPS = 700
ORACLE_INFLATE = 0.30  # robot radius + 5 cm: tightest physically sane path
COVERAGE_STRIDE = 10  # record map coverage every N steps (baseline runs only)

CSV_FIELDS = [
    "kind", "param", "value", "seed", "success", "state", "steps", "sim_time",
    "path_length", "collisions", "first_detect_t", "reach_t", "homing_t",
    "est_err_cm", "known_frac", "oracle_len", "path_eff", "victim_x", "victim_y",
    "wall_s",
]


@dataclass(frozen=True)
class EpisodeConfig:
    """One episode: which study it belongs to and every knob it overrides."""

    kind: str  # "baseline" | "sweep" | "budget" | "random_victim"
    param: str  # swept parameter name ("" for baseline)
    value: float  # swept parameter value (nan for baseline)
    seed: int
    lidar_noise: float = 0.02
    lidar_range: float = 8.0
    num_beams: int = 140
    bearing_noise: float = 0.03
    intensity_noise: float = 0.05
    thermal_range: float = 30.0
    random_victim: bool = False
    max_steps: int = MAX_STEPS


def ground_truth_blocked(world: World, grid: OccupancyGrid, inflate: float) -> np.ndarray:
    """Rasterize the *true* world into a blocked-cell grid (planner's-eye view)."""
    xs = (np.arange(grid.nx) + 0.5) * grid.resolution
    ys = (np.arange(grid.ny) + 0.5) * grid.resolution
    xx, yy = np.meshgrid(xs, ys)
    blocked = (
        (xx < inflate)
        | (xx > world.width - inflate)
        | (yy < inflate)
        | (yy > world.height - inflate)
    )
    for obs in world.obstacles:
        blocked |= (
            (xx >= obs.x - inflate)
            & (xx <= obs.x + obs.w + inflate)
            & (yy >= obs.y - inflate)
            & (yy <= obs.y + obs.h + inflate)
        )
    return blocked


def oracle_path_length(
    blocked: np.ndarray, grid: OccupancyGrid, start_xy, goal_xy
) -> float | None:
    """Length (m) of the shortest ground-truth A* path, or None if unreachable."""
    cells = astar(blocked, grid.world_to_grid(*start_xy), grid.world_to_grid(*goal_xy))
    if not cells:
        return None
    pts = np.array(cells, dtype=float) * grid.resolution
    return float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))


def sample_victim(world: World, start_xy, blocked: np.ndarray, grid: OccupancyGrid, seed: int):
    """Uniformly sample a reachable, non-trivial victim position."""
    rng = np.random.default_rng(10_000 + seed)
    for _ in range(200):
        vx = rng.uniform(0.7, world.width - 0.7)
        vy = rng.uniform(0.7, world.height - 0.7)
        if world.in_collision(vx, vy, radius=0.45):
            continue
        if math.hypot(vx - start_xy[0], vy - start_xy[1]) < 3.5:
            continue
        if oracle_path_length(blocked, grid, start_xy, (vx, vy)) is None:
            continue
        return vx, vy
    raise RuntimeError(f"could not sample a victim position for seed {seed}")


def run_episode(cfg: EpisodeConfig) -> dict:
    """Run one closed-loop episode and return a flat metrics row.

    Re-implements the bookkeeping of ``Navigator.run`` around ``Navigator.step``
    so per-step history (which holds a full grid copy per step) is *not*
    retained, and so detection latency and coverage growth can be extracted.
    """
    world, start = search_rescue_world()
    grid_ref = OccupancyGrid(world.width, world.height, resolution=0.1)
    blocked = ground_truth_blocked(world, grid_ref, ORACLE_INFLATE)

    if cfg.random_victim:
        vx, vy = sample_victim(world, (start.x, start.y), blocked, grid_ref, cfg.seed)
        world = World(world.width, world.height, world.obstacles, [HeatSource(vx, vy, 100.0)])

    victim = world.heat_sources[0].xy
    oracle_len = oracle_path_length(blocked, grid_ref, (start.x, start.y), victim)

    robot = DiffDriveRobot(start.x, start.y, start.theta, radius=0.25, v_max=0.8, omega_max=2.5)
    lidar = Lidar2D(
        num_beams=cfg.num_beams, fov=2 * math.pi, max_range=cfg.lidar_range,
        noise_std=cfg.lidar_noise, seed=cfg.seed,
    )
    thermal = ThermalSensor(
        fov=math.pi, max_range=cfg.thermal_range, reference_intensity=100.0,
        intensity_noise=cfg.intensity_noise, bearing_noise=cfg.bearing_noise, seed=cfg.seed,
    )
    controller = PurePursuit(lookahead=0.45, v_nominal=0.6, omega_max=2.5, goal_tolerance=0.25)
    nav = Navigator(
        robot, world, lidar, thermal,
        grid=OccupancyGrid(world.width, world.height, resolution=0.1),
        controller=controller, dt=DT, reach_radius=0.5, replan_every=5, safety_margin=0.2,
    )

    t0 = time.perf_counter()
    path_length = 0.0
    collisions = 0
    first_detect: int | None = None
    coverage: list[float] = []
    prev = robot.pose.xy
    steps = 0

    for i in range(cfg.max_steps):
        rec = nav.step()
        steps = i + 1
        if first_detect is None and rec.detections:
            first_detect = steps
        if i % COVERAGE_STRIDE == 0:
            coverage.append(float(nav.grid.known_mask.mean()))
        now = robot.pose.xy
        path_length += float(np.hypot(*(now - prev)))
        prev = now
        if world.in_collision(now[0], now[1], robot.radius):
            collisions += 1
        if nav.state in (NavState.REACHED, NavState.FAILED):
            break

    success = nav.state == NavState.REACHED
    reach_t = steps * DT if success else math.nan
    first_detect_t = first_detect * DT if first_detect is not None else math.nan
    est_err_cm = (
        float(np.hypot(*(nav.goal_estimate - victim))) * 100.0
        if nav.goal_estimate is not None
        else math.nan
    )

    row = {
        "kind": cfg.kind,
        "param": cfg.param,
        "value": cfg.value,
        "seed": cfg.seed,
        "success": int(success),
        "state": nav.state.value,
        "steps": steps,
        "sim_time": steps * DT,
        "path_length": path_length,
        "collisions": collisions,
        "first_detect_t": first_detect_t,
        "reach_t": reach_t,
        "homing_t": reach_t - first_detect_t,
        "est_err_cm": est_err_cm,
        "known_frac": float(nav.grid.known_mask.mean()),
        "oracle_len": oracle_len if oracle_len is not None else math.nan,
        # The robot stops reach_radius short of the victim, so subtract that
        # from the oracle for a like-for-like efficiency ratio.
        "path_eff": (
            (oracle_len - nav.reach_radius) / path_length
            if (oracle_len and path_length > 0)
            else math.nan
        ),
        "victim_x": float(victim[0]),
        "victim_y": float(victim[1]),
        "wall_s": time.perf_counter() - t0,
        "_coverage": coverage,  # stripped before CSV write
    }
    return row


def build_tasks(quick: bool) -> list[EpisodeConfig]:
    n_base = 20 if quick else 100
    n_sweep = 5 if quick else 20
    n_rand = 10 if quick else 60

    tasks = [EpisodeConfig("baseline", "", math.nan, s) for s in range(n_base)]

    sweeps: dict[str, list[float]] = {
        "bearing_noise": [0.0, 0.05, 0.15, 0.30, 0.50],
        "intensity_noise": [0.0, 0.10, 0.25, 0.50],
        "thermal_range": [3.0, 4.0, 6.0, 9.0, 15.0, 30.0],
        "lidar_noise": [0.0, 0.05, 0.10, 0.20, 0.30],
        "lidar_range": [2.0, 3.0, 4.5, 6.0, 8.0],
        "num_beams": [30, 60, 100, 140, 200],
    }
    for param, values in sweeps.items():
        for value in values:
            for s in range(n_sweep):
                override = {param: int(value) if param == "num_beams" else value}
                tasks.append(
                    EpisodeConfig("sweep", param, float(value), s, **override)
                )

    # Budget-extension study: re-run the failure-cliff levels with 3x the step
    # budget to separate "slow but solvable" from "terminally stuck".
    budget_levels: dict[str, list[float]] = {
        "lidar_range": [2.0, 3.0, 4.5],
        "thermal_range": [3.0, 4.0, 6.0],
    }
    for param, values in budget_levels.items():
        for value in values:
            for s in range(n_sweep):
                override = {param: value}
                tasks.append(
                    EpisodeConfig(
                        "budget", param, float(value), s,
                        max_steps=3 * MAX_STEPS, **override,
                    )
                )

    tasks += [
        EpisodeConfig("random_victim", "", math.nan, s, random_victim=True)
        for s in range(n_rand)
    ]
    return tasks


def run_all(tasks: list[EpisodeConfig], jobs: int) -> list[dict]:
    rows: list[dict] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for i, row in enumerate(pool.map(run_episode, tasks, chunksize=4), 1):
            rows.append(row)
            if i % 25 == 0 or i == len(tasks):
                print(
                    f"  {i}/{len(tasks)} episodes "
                    f"({time.perf_counter() - t0:.0f}s elapsed)",
                    flush=True,
                )
    return rows


def write_results(rows: list[dict]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    coverage = [r.pop("_coverage") for r in rows]
    base_cov = [c for r, c in zip(rows, coverage, strict=True) if r["kind"] == "baseline"]
    if base_cov:
        width = max(len(c) for c in base_cov)
        mat = np.full((len(base_cov), width), np.nan)
        for i, c in enumerate(base_cov):
            mat[i, : len(c)] = c
        np.savez_compressed(
            os.path.join(RESULTS_DIR, "baseline_coverage.npz"),
            coverage=mat,
            t=np.arange(width) * COVERAGE_STRIDE * DT,
        )

    path = os.path.join(RESULTS_DIR, "episodes.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} episodes)")


def load_results() -> list[dict]:
    with open(os.path.join(RESULTS_DIR, "episodes.csv"), newline="") as f:
        rows = []
        for raw in csv.DictReader(f):
            row: dict = dict(raw)
            for key in CSV_FIELDS:
                if key in ("kind", "param", "state"):
                    continue
                row[key] = float(raw[key]) if raw[key] != "" else math.nan
            rows.append(row)
        return rows


# --- aggregation ----------------------------------------------------------------
def _filter(rows, **eq):
    return [r for r in rows if all(r[k] == v for k, v in eq.items())]


def _col(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def _finite(a):
    return a[np.isfinite(a)]


def summarize(rows: list[dict]) -> str:
    """Aggregate the episode table into the printed analysis summary."""
    lines: list[str] = []

    def emit(s=""):
        lines.append(s)

    base = _filter(rows, kind="baseline")
    ok = [r for r in base if r["success"]]
    emit(f"=== Baseline (fixed world, {len(base)} seeds) ===")
    emit(f"  success rate     : {100 * len(ok) / len(base):.1f}%  ({len(ok)}/{len(base)})")
    if ok:
        for key, unit, scale in [
            ("reach_t", "s", 1), ("path_length", "m", 1), ("path_eff", "", 1),
            ("first_detect_t", "s", 1), ("homing_t", "s", 1), ("est_err_cm", "cm", 1),
        ]:
            v = _finite(_col(ok, key))
            emit(
                f"  {key:<17}: median {np.median(v) * scale:6.2f}{unit}   "
                f"mean {np.mean(v) * scale:6.2f}   "
                f"p10 {np.percentile(v, 10) * scale:6.2f}   "
                f"p90 {np.percentile(v, 90) * scale:6.2f}"
            )
        n_coll = int(_col(base, "collisions").sum())
        emit(f"  collisions       : total {n_coll} across all episodes")
        emit(f"  oracle length    : {ok[0]['oracle_len']:.2f} m")
        emit(f"  map known at end : {100 * np.mean(_col(ok, 'known_frac')):.1f}% (successes)")

    emit()
    emit("=== Sensitivity sweeps (success% | median reach_t of successes) ===")
    sweep = _filter(rows, kind="sweep")
    for param in sorted({r["param"] for r in sweep}):
        emit(f"  {param}:")
        prows = [r for r in sweep if r["param"] == param]
        for value in sorted({r["value"] for r in prows}):
            vrows = [r for r in prows if r["value"] == value]
            vok = [r for r in vrows if r["success"]]
            med = np.median(_col(vok, "reach_t")) if vok else math.nan
            n_coll = int(_col(vrows, "collisions").sum())
            emit(
                f"    {value:8.3g} -> {100 * len(vok) / len(vrows):5.1f}%   "
                f"median {med:6.1f}s   collisions {n_coll}"
            )

    budget = _filter(rows, kind="budget")
    if budget:
        emit()
        emit(f"=== Budget extension: cliff levels re-run at {3 * MAX_STEPS} steps ===")
        emit("    (success at 1x budget -> success at 3x budget)")
        sweep_all = _filter(rows, kind="sweep")
        for param in sorted({r["param"] for r in budget}):
            for value in sorted({r["value"] for r in budget if r["param"] == param}):
                brows = [r for r in budget if r["param"] == param and r["value"] == value]
                srows = [
                    r for r in sweep_all if r["param"] == param and r["value"] == value
                ]
                b_ok = 100 * sum(r["success"] for r in brows) / len(brows)
                s_ok = 100 * sum(r["success"] for r in srows) / len(srows) if srows else math.nan
                hard = sum(1 for r in brows if r["state"] == "failed")
                emit(
                    f"  {param}={value:g}: {s_ok:5.1f}% -> {b_ok:5.1f}%   "
                    f"(hard-stuck at 3x: {hard}/{len(brows)})"
                )

    rand = _filter(rows, kind="random_victim")
    if rand:
        rok = [r for r in rand if r["success"]]
        emit()
        emit(f"=== Random victim placement ({len(rand)} placements) ===")
        emit(f"  success rate : {100 * len(rok) / len(rand):.1f}%  ({len(rok)}/{len(rand)})")
        if rok:
            emit(f"  median reach : {np.median(_finite(_col(rok, 'reach_t'))):.1f}s")
            emit(f"  median eff.  : {np.median(_finite(_col(rok, 'path_eff'))):.2f}")
        emit(f"  collisions   : {int(_col(rand, 'collisions').sum())}")
        for r in rand:
            if not r["success"]:
                emit(
                    f"  FAILED seed {int(r['seed'])}: victim "
                    f"({r['victim_x']:.1f}, {r['victim_y']:.1f}), state={r['state']}"
                )
    return "\n".join(lines)


# --- figures ----------------------------------------------------------------------
def make_figures(rows: list[dict]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out: list[str] = []

    base_ok = [r for r in _filter(rows, kind="baseline") if r["success"]]

    # 1. Baseline distributions.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    (ax_t, ax_len), (ax_split, ax_err) = axes

    reach = _col(base_ok, "reach_t")
    ax_t.hist(reach, bins=18, color="#2a7fbf", edgecolor="white")
    ax_t.axvline(np.median(reach), color="k", ls="--", lw=1.2,
                 label=f"median {np.median(reach):.1f}s")
    ax_t.set(xlabel="time to rescue (s)", ylabel="episodes", title="Time to rescue")
    ax_t.legend()

    lengths = _col(base_ok, "path_length")
    oracle = base_ok[0]["oracle_len"]
    ax_len.hist(lengths, bins=18, color="#7fbf2a", edgecolor="white")
    ax_len.axvline(oracle, color="crimson", ls="--", lw=1.5,
                   label=f"A* oracle {oracle:.1f}m")
    med_eff = np.median(_finite(_col(base_ok, "path_eff")))
    ax_len.set(xlabel="path length (m)", ylabel="episodes",
               title=f"Path length (median efficiency {med_eff:.0%})")
    ax_len.legend()

    ax_split.scatter(_col(base_ok, "first_detect_t"), _col(base_ok, "homing_t"),
                     s=22, alpha=0.7, color="#bf2a7f")
    ax_split.set(xlabel="search phase: time to first thermal detection (s)",
                 ylabel="homing phase: detection → reach (s)",
                 title="Where the time goes: search vs homing")
    ax_split.grid(alpha=0.3)

    err = _finite(_col(base_ok, "est_err_cm"))
    ax_err.hist(err, bins=18, color="#bf7f2a", edgecolor="white")
    ax_err.axvline(np.median(err), color="k", ls="--", lw=1.2,
                   label=f"median {np.median(err):.0f}cm")
    ax_err.set(xlabel="final victim-estimate error (cm)", ylabel="episodes",
               title="Thermal localization error at arrival")
    ax_err.legend()

    fig.suptitle("PyroScout baseline: 100-seed Monte-Carlo on the three-room scenario")
    fig.tight_layout()
    p = os.path.join(FIGURES_DIR, "fig_baseline.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    out.append(p)

    # 2. Coverage growth.
    cov_path = os.path.join(RESULTS_DIR, "baseline_coverage.npz")
    if os.path.exists(cov_path):
        data = np.load(cov_path)
        cov, t = data["coverage"] * 100, data["t"]
        # Episodes end when they succeed, so late time points only contain the
        # slow stragglers; truncate where fewer than half are still running to
        # avoid survivor bias in the aggregate curve.
        alive = np.isfinite(cov).sum(axis=0)
        keep = alive >= 0.5 * cov.shape[0]
        cov, t = cov[:, keep], t[keep]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        med = np.nanmedian(cov, axis=0)
        lo, hi = np.nanpercentile(cov, [10, 90], axis=0)
        ax.fill_between(t, lo, hi, alpha=0.25, color="#2a7fbf", label="10–90% band")
        ax.plot(t, med, color="#2a7fbf", lw=2, label="median")
        det = np.median(_finite(_col(base_ok, "first_detect_t")))
        ax.axvline(det, color="crimson", ls="--", lw=1.2,
                   label=f"median first detection ({det:.0f}s)")
        ax.set(xlabel="simulated time (s)", ylabel="map observed (%)",
               title="Frontier exploration: map coverage over time (baseline)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(FIGURES_DIR, "fig_coverage.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        out.append(p)

    # 3. Sensitivity sweeps.
    sweep = _filter(rows, kind="sweep")
    params = [
        ("thermal_range", "thermal detection range (m)"),
        ("bearing_noise", "thermal bearing noise σ (rad)"),
        ("intensity_noise", "thermal intensity noise (fraction)"),
        ("lidar_range", "LIDAR max range (m)"),
        ("lidar_noise", "LIDAR range noise σ (m)"),
        ("num_beams", "LIDAR beam count"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8))
    for ax, (param, label) in zip(axes.ravel(), params, strict=True):
        prows = [r for r in sweep if r["param"] == param]
        values = sorted({r["value"] for r in prows})
        succ, med_t = [], []
        for value in values:
            vrows = [r for r in prows if r["value"] == value]
            vok = [r for r in vrows if r["success"]]
            succ.append(100 * len(vok) / len(vrows))
            med_t.append(np.median(_col(vok, "reach_t")) if vok else math.nan)
        ax.plot(values, succ, "o-", color="#2a7fbf", lw=2, label="success rate")
        ax.set_ylim(-5, 105)
        ax.set(xlabel=label, ylabel="success rate (%)")
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(values, med_t, "s--", color="#bf2a7f", lw=1.5, label="median time")
        ax2.set_ylabel("median time to rescue (s)")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8)
    fig.suptitle("Sensor degradation sweeps (20 seeds per point)")
    fig.tight_layout()
    p = os.path.join(FIGURES_DIR, "fig_sweeps.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    out.append(p)

    # 4. Random victim placements over the world map.
    rand = _filter(rows, kind="random_victim")
    if rand:
        world, start = search_rescue_world()
        fig, ax = plt.subplots(figsize=(9, 6.5))
        for obs in world.obstacles:
            ax.add_patch(plt.Rectangle((obs.x, obs.y), obs.w, obs.h, color="#444"))
        ax.add_patch(plt.Rectangle((0, 0), world.width, world.height,
                                   fill=False, ec="#444", lw=2))
        rok = [r for r in rand if r["success"]]
        rbad = [r for r in rand if not r["success"]]
        if rok:
            sc = ax.scatter(_col(rok, "victim_x"), _col(rok, "victim_y"),
                            c=_col(rok, "reach_t"), cmap="viridis", s=55,
                            edgecolor="k", lw=0.5, zorder=3, label="reached")
            fig.colorbar(sc, ax=ax, label="time to rescue (s)")
        if rbad:
            ax.scatter(_col(rbad, "victim_x"), _col(rbad, "victim_y"),
                       marker="X", color="crimson", s=90, zorder=4, label="failed")
        ax.plot(start.x, start.y, "*", color="#2a7fbf", ms=20, mec="k", label="start")
        rate = 100 * len(rok) / len(rand)
        ax.set(xlim=(-0.3, world.width + 0.3), ylim=(-0.3, world.height + 0.3),
               title=f"Random victim placement: {len(rand)} episodes, "
                     f"{rate:.0f}% reached",
               xlabel="x (m)", ylabel="y (m)")
        ax.set_aspect("equal")
        ax.legend(loc="upper left")
        fig.tight_layout()
        p = os.path.join(FIGURES_DIR, "fig_random_victims.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        out.append(p)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="reduced episode counts")
    parser.add_argument("--reuse", action="store_true", help="figures from existing CSV")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    args = parser.parse_args()

    if not args.reuse:
        tasks = build_tasks(args.quick)
        print(f"Running {len(tasks)} episodes on {args.jobs} workers ...")
        rows = run_all(tasks, args.jobs)
        write_results(rows)

    rows = load_results()
    print()
    print(summarize(rows))
    figures = make_figures(rows)
    print("\nFigures:")
    for p in figures:
        print(f"  {p}")


if __name__ == "__main__":
    main()
