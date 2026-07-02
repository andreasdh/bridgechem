"""Visualisation tests (headless: Agg backend, no live kernel)."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

import bridgechem as bc
from bridgechem import viz

# IPython.display is required by the live viewer; skip these if it's absent.
pytest.importorskip("IPython")


def test_in_notebook_false_under_pytest():
    assert viz.in_notebook() is False


def test_scene_builds_and_updates():
    system = bc.box(N=40, size=(15, 15), seed=0)
    fig, ax, coll, quiv, title = viz._setup_scene(
        system.Lx, system.Ly, system.radius, system.display_scale,
        vectors=True, color_by="speed", figsize=(4, 4),
        mean_speed=300.0, vmax=750.0,
    )
    viz._update_artists(coll, quiv, title, system.pos, system.vel, "speed", 1e-12)
    fig.canvas.draw()
    assert coll.get_offsets().shape == (40, 2)
    matplotlib.pyplot.close(fig)


def test_live_run_records_trajectory_and_conserves_energy():
    system = bc.box(N=50, size=(15, 15), temperature=300, seed=0)
    dt = system._auto_dt()
    tp, tv, ts, imp = viz.live_run(
        system, dt=dt, steps=1000, sample_every=100,
        vectors=True, color_by="speed", fps=0, figsize=(4, 4),
    )
    assert tp.shape == (11, 50, 2)
    assert tv.shape == tp.shape
    assert np.all(np.diff(ts) > 0)
    ke = (0.5 * system.mass * np.sum(tv ** 2, axis=-1)).sum(axis=-1)
    assert (ke.max() - ke.min()) / ke.mean() < 1e-9


def test_show_replay_runs(monkeypatch):
    system = bc.box(N=30, size=(15, 15), seed=0)
    sim = system.run(steps=600, sample_every=100, animate=False)
    sim.show(fps=0)  # should not raise under Agg / no kernel


def test_show_replay_accepts_speed(monkeypatch):
    system = bc.box(N=30, size=(15, 15), seed=0)
    sim = system.run(steps=600, sample_every=100, animate=False)
    sim.show(fps=0, speed=2.0)  # should not raise


def test_pick_sample_every_monotonic_in_speed():
    mean_speed, dt, Lx, Ly = 300.0, 1e-13, 40e-9, 40e-9
    slow = viz.pick_sample_every(mean_speed, dt, Lx, Ly, fps=30, speed=0.3)
    mid = viz.pick_sample_every(mean_speed, dt, Lx, Ly, fps=30, speed=1.0)
    fast = viz.pick_sample_every(mean_speed, dt, Lx, Ly, fps=30, speed=3.0)
    assert slow < mid < fast


def test_pick_sample_every_targets_crossing_time():
    mean_speed, dt, Lx, Ly = 300.0, 1e-13, 40e-9, 40e-9
    se = viz.pick_sample_every(mean_speed, dt, Lx, Ly, fps=30, speed=1.0)
    # wall-clock seconds for one box-crossing at this sample_every / fps
    crossing_time = min(Lx, Ly) / mean_speed
    frames_per_crossing = crossing_time / (se * dt)
    wallclock_per_crossing = frames_per_crossing / 30.0
    assert 4.0 < wallclock_per_crossing < 9.0  # near the 6s target


def test_run_default_speed_gives_smooth_frame_to_frame_motion():
    # per-frame displacement should stay within a couple of particle
    # diameters, i.e. particles glide rather than teleport across the box.
    system = bc.box(N=100, size=(30, 30), temperature=300, seed=0)
    sim = system.run(steps=6000, animate=False, speed=1.0)
    disp = np.sqrt(np.sum(np.diff(sim.pos, axis=0) ** 2, axis=-1))
    assert disp.max() < 3.0 * (2.0 * system.radius[0])


def test_run_higher_speed_gives_fewer_frames():
    system = bc.box(N=100, size=(30, 30), temperature=300, seed=0)
    slow_sim = bc.box(N=100, size=(30, 30), temperature=300, seed=0).run(
        steps=6000, animate=False, speed=0.3)
    fast_sim = system.run(steps=6000, animate=False, speed=3.0)
    assert fast_sim.n_frames < slow_sim.n_frames
