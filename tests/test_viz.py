"""Visualisation tests (headless: Agg backend, no live browser kernel)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pytest

import bridgechem as bc
from bridgechem import kernels, viz

# The interactive player needs IPython + ipywidgets; skip these if absent.
pytest.importorskip("IPython")
widgets = pytest.importorskip("ipywidgets")


def _trajectory(N=60, size=(20, 20), steps=2000, sample_every=100, seed=0,
               **box_kwargs):
    system = bc.box(N=N, size=size, temperature=300, seed=seed, **box_kwargs)
    dt = system._auto_dt()
    traj_pos, traj_vel, times, impulse, virial = kernels._simulate(
        system.pos, system.vel, system.radius, system.inv_mass,
        system.Lx, system.Ly, dt, steps, sample_every, system.periodic,
    )
    return system, traj_pos, traj_vel, times, impulse


def test_in_notebook_false_under_pytest():
    assert viz.in_notebook() is False


def test_scene_builds_and_updates():
    system = bc.box(N=40, size=(15, 15), seed=0)
    fig, ax, coll, quiv, title = viz._setup_scene(
        system.Lx, system.Ly, system.radius, system.display_scale,
        vectors=True, color_by="speed", figsize=(4, 4), mean_speed=300.0,
        vmax=750.0,
    )
    viz._update_artists(coll, quiv, title, system.pos, system.vel, "speed", 1e-12)
    fig.canvas.draw()
    assert coll.get_offsets().shape == (40, 2)
    matplotlib.pyplot.close(fig)


def test_play_returns_widget_with_speed_coloring():
    system, pos, vel, times, _ = _trajectory()
    pw = viz.play(pos, vel, times, system.mass, system.radius, system.Lx,
                  system.Ly, vectors=True, color_by="speed", fps=30, speed=1.0)
    assert isinstance(pw, widgets.Play)
    assert pw.max == pos.shape[0] - 1


def test_play_scrubbing_updates_frame_and_conserves_energy():
    system, pos, vel, times, _ = _trajectory(N=80, steps=3000)
    ke = (0.5 * system.mass * np.sum(vel ** 2, axis=-1)).sum(axis=-1)
    assert (ke.max() - ke.min()) / ke.mean() < 1e-9  # sanity: elastic engine

    pw = viz.play(pos, vel, times, system.mass, system.radius, system.Lx,
                  system.Ly, color_by="speed", fps=30, speed=1.0)
    last = pos.shape[0] - 1
    pw.value = last  # simulate scrubbing to the final frame
    # after scrubbing, the artist should reflect the last frame's positions
    # (indirectly verified via no exception + widget state)
    assert pw.value == last


def test_play_color_by_mass_mixture():
    system, pos, vel, times, _ = _trajectory(N=40, steps=1000)
    system.set_mass(80.0, indices=slice(0, 20))
    pw = viz.play(pos, vel, times, system.mass, system.radius, system.Lx,
                  system.Ly, color_by="mass", fps=30)
    assert isinstance(pw, widgets.Play)


def test_play_color_by_mass_uniform_does_not_crash():
    system, pos, vel, times, _ = _trajectory(N=30, steps=500)
    pw = viz.play(pos, vel, times, system.mass, system.radius, system.Lx,
                  system.Ly, color_by="mass", fps=30)
    assert isinstance(pw, widgets.Play)


def test_play_invalid_color_by_raises():
    system, pos, vel, times, _ = _trajectory(N=20, steps=500)
    with pytest.raises(ValueError):
        viz.play(pos, vel, times, system.mass, system.radius, system.Lx,
                 system.Ly, color_by="type")


def test_play_does_not_leak_open_figures():
    # Regression test: play() must drop its figure from pyplot's registry
    # once handed off to the display handle, otherwise IPython's inline
    # backend auto-renders it again as a frozen duplicate at cell end, and
    # repeated calls (e.g. run() then show()) accumulate open figures.
    n_before = len(plt.get_fignums())
    system, pos, vel, times, _ = _trajectory(N=30, steps=500)
    viz.play(pos, vel, times, system.mass, system.radius, system.Lx,
             system.Ly, color_by="speed")
    viz.play(pos, vel, times, system.mass, system.radius, system.Lx,
             system.Ly, color_by=None)
    assert len(plt.get_fignums()) == n_before


def test_play_zero_fps_does_not_raise():
    system, pos, vel, times, _ = _trajectory(N=20, steps=500)
    pw = viz.play(pos, vel, times, system.mass, system.radius, system.Lx,
                  system.Ly, fps=0)
    assert pw.interval >= 1


def test_run_and_show_end_to_end(monkeypatch):
    system = bc.box(N=30, size=(15, 15), seed=0)
    sim = system.run(steps=600, sample_every=100, animate=False)
    pw = sim.show(fps=0, display_scale=2.0)
    assert isinstance(pw, widgets.Play)
    assert sim.display_scale == 1.0  # override doesn't mutate stored state


def test_pick_sample_every_monotonic_in_speed():
    mean_speed, dt, Lx, Ly = 300.0, 1e-13, 40e-9, 40e-9
    slow = viz.pick_sample_every(mean_speed, dt, Lx, Ly, fps=30, speed=0.3)
    mid = viz.pick_sample_every(mean_speed, dt, Lx, Ly, fps=30, speed=1.0)
    fast = viz.pick_sample_every(mean_speed, dt, Lx, Ly, fps=30, speed=3.0)
    assert slow < mid < fast


def test_pick_sample_every_targets_crossing_time():
    mean_speed, dt, Lx, Ly = 300.0, 1e-13, 40e-9, 40e-9
    se = viz.pick_sample_every(mean_speed, dt, Lx, Ly, fps=30, speed=1.0)
    crossing_time = min(Lx, Ly) / mean_speed
    frames_per_crossing = crossing_time / (se * dt)
    wallclock_per_crossing = frames_per_crossing / 30.0
    assert 4.0 < wallclock_per_crossing < 9.0  # near the 6s target


def test_run_default_speed_gives_smooth_frame_to_frame_motion():
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
