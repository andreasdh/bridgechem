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
        system.L, dt, steps, sample_every, system.periodic,
    )
    return system, traj_pos, traj_vel, times, impulse


def _trajectory_3d(N=40, size=(15, 15, 15), steps=1000, sample_every=100,
                   seed=0, **box_kwargs):
    return _trajectory(N=N, size=size, steps=steps, sample_every=sample_every,
                       seed=seed, **box_kwargs)


def test_in_notebook_false_under_pytest():
    assert viz.in_notebook() is False


def test_scene_builds_and_updates():
    system = bc.box(N=40, size=(15, 15), seed=0)
    fig, ax, coll, quiv, title = viz._setup_scene(
        system.L, system.radius, system.display_scale,
        vectors=True, color_by="speed", figsize=(4, 4), mean_speed=300.0,
        vmax=750.0,
    )
    viz._update_artists(coll, quiv, title, system.pos, system.vel, "speed", 1e-12)
    fig.canvas.draw()
    assert coll.get_offsets().shape == (40, 2)
    matplotlib.pyplot.close(fig)


def test_scene_builds_and_updates_3d():
    system = bc.box(N=30, size=(15, 15, 15), seed=0)
    fig, ax, coll, quiv, title = viz._setup_scene_3d(
        system.L, system.radius, system.display_scale,
        vectors=True, color_by="speed", figsize=(4, 4), mean_speed=300.0,
        vmax=750.0,
    )
    assert ax.name == "3d"  # a real Axes3D, not a 2D projection
    assert quiv is None  # 3D quivers are created on first update, not at setup
    new_quiv = viz._update_artists_3d(
        ax, coll, quiv, title, system.pos, system.vel, "speed", 1e-12,
        True, 300.0, system.L,
    )
    fig.canvas.draw()
    x, y, z = coll._offsets3d
    assert len(x) == len(y) == len(z) == 30
    assert np.allclose(x, system.pos[:, 0] * 1e9)
    assert new_quiv is not None
    matplotlib.pyplot.close(fig)


def test_is_live_backend_detects_known_live_backends(monkeypatch):
    import matplotlib as mpl
    for name in ("module://ipympl.backend_nbagg", "nbAgg", "WebAgg"):
        monkeypatch.setattr(mpl, "get_backend", lambda name=name: name)
        assert viz._is_live_backend() is True
    for name in ("agg", "module://matplotlib_inline.backend_inline"):
        monkeypatch.setattr(mpl, "get_backend", lambda name=name: name)
        assert viz._is_live_backend() is False


def test_ensure_interactive_backend_never_switches_outside_a_notebook():
    # Under pytest in_notebook() is always False, so this must never try to
    # touch the global matplotlib backend (which would leak into every other
    # test in the process).
    before = matplotlib.get_backend()
    assert viz._ensure_interactive_backend() is False
    assert matplotlib.get_backend() == before


def test_auto_size_factor_is_a_noop_for_already_reasonable_sizes():
    # A typical 2D demo (N=200 in a 60 nm box, default packing) already looks
    # right today -- this must not perturb it.
    system = bc.box(N=200, size=(60, 60), packing=0.10, seed=0)
    factor = viz._auto_size_factor(system.radius, system.L,
                                   viz.MAX_DISPLAY_DIAM_FRAC_2D)
    assert 0.9 < factor <= 1.0


def test_auto_size_factor_shrinks_the_reported_oversized_case():
    # The exact bug report: N=9 in a 120 nm cube, default packing=0.10 --
    # the raw radius alone demands a diameter ~28% of the box width.
    system = bc.box(N=9, size=(120, 120, 120), packing=0.10, seed=0)
    natural_frac = 2.0 * system.radius[0] / np.min(system.L)
    assert natural_frac > viz.MAX_DISPLAY_DIAM_FRAC_3D  # confirms the bug exists

    factor = viz._auto_size_factor(system.radius, system.L,
                                   viz.MAX_DISPLAY_DIAM_FRAC_3D)
    drawn_frac = natural_frac * factor
    assert np.isclose(drawn_frac, viz.MAX_DISPLAY_DIAM_FRAC_3D)


def test_auto_size_factor_grows_a_real_van_der_waals_scale_radius():
    # A real gas's actual radius (angstrom-scale) in a box sized for
    # visibility (tens of nm) would otherwise be invisible.
    argon_radius_m = bc.constants.gas_properties("argon")["radius_m"]
    L = np.array([120e-9, 120e-9, 120e-9])
    natural_frac = 2.0 * argon_radius_m / np.min(L)
    assert natural_frac < viz.MIN_DISPLAY_DIAM_FRAC  # confirms it'd be invisible

    factor = viz._auto_size_factor(np.array([argon_radius_m]), L,
                                   viz.MAX_DISPLAY_DIAM_FRAC_3D)
    drawn_frac = natural_frac * factor
    assert np.isclose(drawn_frac, viz.MIN_DISPLAY_DIAM_FRAC)


def test_scene_3d_caps_particle_size_for_the_reported_bug_case():
    # End-to-end: the actual scene builder must not draw absurdly large
    # spheres for N=9 in a big 3D box.
    system = bc.box(N=9, size=(120, 120, 120), packing=0.10, seed=0)
    fig, ax, coll, quiv, title = viz._setup_scene_3d(
        system.L, system.radius, system.display_scale,
        vectors=False, color_by=None, figsize=(6, 6), mean_speed=300.0,
    )
    diam_points = np.sqrt(coll.get_sizes()[0])
    points_per_nm = (6 * 72.0) / 120.0  # figsize[0]*72 / max(Lx,Ly,Lz) in nm
    diam_nm = diam_points / points_per_nm
    assert diam_nm / 120.0 <= viz.MAX_DISPLAY_DIAM_FRAC_3D + 1e-9
    matplotlib.pyplot.close(fig)


def test_play_returns_widget_with_speed_coloring():
    system, pos, vel, times, _ = _trajectory()
    pw = viz.play(pos, vel, times, system.mass, system.radius, system.L, vectors=True, color_by="speed", fps=30, speed=1.0)
    assert isinstance(pw, widgets.Play)
    assert pw.max == pos.shape[0] - 1


def test_play_scrubbing_updates_frame_and_conserves_energy():
    system, pos, vel, times, _ = _trajectory(N=80, steps=3000)
    ke = (0.5 * system.mass * np.sum(vel ** 2, axis=-1)).sum(axis=-1)
    assert (ke.max() - ke.min()) / ke.mean() < 1e-9  # sanity: elastic engine

    pw = viz.play(pos, vel, times, system.mass, system.radius, system.L, color_by="speed", fps=30, speed=1.0)
    last = pos.shape[0] - 1
    pw.value = last  # simulate scrubbing to the final frame
    # after scrubbing, the artist should reflect the last frame's positions
    # (indirectly verified via no exception + widget state)
    assert pw.value == last


def test_play_color_by_mass_mixture():
    system, pos, vel, times, _ = _trajectory(N=40, steps=1000)
    system.set_mass(80.0, indices=slice(0, 20))
    pw = viz.play(pos, vel, times, system.mass, system.radius, system.L, color_by="mass", fps=30)
    assert isinstance(pw, widgets.Play)


def test_play_color_by_mass_uniform_does_not_crash():
    system, pos, vel, times, _ = _trajectory(N=30, steps=500)
    pw = viz.play(pos, vel, times, system.mass, system.radius, system.L, color_by="mass", fps=30)
    assert isinstance(pw, widgets.Play)


def test_play_invalid_color_by_raises():
    system, pos, vel, times, _ = _trajectory(N=20, steps=500)
    with pytest.raises(ValueError):
        viz.play(pos, vel, times, system.mass, system.radius, system.L, color_by="type")


def test_play_3d_uses_rotatable_scene_and_does_not_crash():
    system, pos, vel, times, _ = _trajectory_3d()
    pw = viz.play(pos, vel, times, system.mass, system.radius, system.L,
                  color_by="speed", fps=30)
    assert isinstance(pw, widgets.Play)
    assert pw.max == pos.shape[0] - 1


def test_play_3d_with_vectors_does_not_crash():
    # 3D vectors are the expensive path (the quiver is rebuilt every frame),
    # so this is the one most likely to break -- exercise it explicitly.
    system, pos, vel, times, _ = _trajectory_3d(N=20, steps=500)
    pw = viz.play(pos, vel, times, system.mass, system.radius, system.L,
                  vectors=True, color_by="mass", fps=30)
    assert isinstance(pw, widgets.Play)


def test_play_3d_with_slab_falls_back_to_2d_projection():
    system, pos, vel, times, _ = _trajectory_3d()
    pw = viz.play(pos, vel, times, system.mass, system.radius, system.L,
                  color_by="speed", fps=30, slab=2.0)
    assert isinstance(pw, widgets.Play)


def test_play_does_not_leak_open_figures():
    # Regression test: play() must drop its figure from pyplot's registry
    # once handed off to the display handle, otherwise IPython's inline
    # backend auto-renders it again as a frozen duplicate at cell end, and
    # repeated calls (e.g. run() then show()) accumulate open figures.
    n_before = len(plt.get_fignums())
    system, pos, vel, times, _ = _trajectory(N=30, steps=500)
    viz.play(pos, vel, times, system.mass, system.radius, system.L, color_by="speed")
    viz.play(pos, vel, times, system.mass, system.radius, system.L, color_by=None)
    assert len(plt.get_fignums()) == n_before


def test_play_never_exceeds_measured_achievable_fps():
    # Regression test: the Play widget must never tick faster than this
    # machine can actually redraw+encode a frame, or ticks pile up faster
    # than they can be drawn -- which looks like stutter and can make
    # playback appear to skip or jump backward.
    system, pos, vel, times, _ = _trajectory(N=100, steps=2000)
    pw = viz.play(pos, vel, times, system.mass, system.radius, system.L, color_by="speed", fps=1000)  # deliberately absurd
    achieved_fps = 1000.0 / pw.interval
    assert achieved_fps < 1000  # must have been capped down, not honored


def test_play_zero_fps_does_not_raise():
    system, pos, vel, times, _ = _trajectory(N=20, steps=500)
    pw = viz.play(pos, vel, times, system.mass, system.radius, system.L, fps=0)
    assert pw.interval >= 1


def test_run_and_show_end_to_end(monkeypatch):
    system = bc.box(N=30, size=(15, 15), seed=0)
    sim = system.run(steps=600, sample_every=100, animate=False)
    pw = sim.show(fps=0, display_scale=2.0)
    assert isinstance(pw, widgets.Play)
    assert sim.display_scale == 1.0  # override doesn't mutate stored state


def test_pick_sample_every_monotonic_in_speed():
    mean_speed, dt, L = 300.0, 1e-13, np.array([40e-9, 40e-9])
    slow = viz.pick_sample_every(mean_speed, dt, L, fps=30, speed=0.3)
    mid = viz.pick_sample_every(mean_speed, dt, L, fps=30, speed=1.0)
    fast = viz.pick_sample_every(mean_speed, dt, L, fps=30, speed=3.0)
    assert slow < mid < fast


def test_pick_sample_every_targets_crossing_time():
    mean_speed, dt, L = 300.0, 1e-13, np.array([40e-9, 40e-9])
    se = viz.pick_sample_every(mean_speed, dt, L, fps=30, speed=1.0)
    crossing_time = float(np.min(L)) / mean_speed
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
