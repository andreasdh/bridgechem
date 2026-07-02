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
