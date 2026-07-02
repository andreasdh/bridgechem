"""Real-time matplotlib visualisation for bridgechem.

The simulation animates *live* while it integrates, updating a single figure in
place via ``IPython.display`` (``display_id``). This works with the default
``%matplotlib inline`` backend -- no interactive backend, no ``to_jshtml`` HTML
blob, and nothing to click after ``run()``: the box appears and starts moving
immediately.

Particles are drawn as filled circles at their true collision size (times an
optional ``display_scale``), so what you see is what bounces. Velocity vectors
can be overlaid as arrows.
"""

from __future__ import annotations

import numpy as np


def _nm(x):
    return np.asarray(x) * 1e9


# A mean-speed particle crossing the box takes this many wall-clock seconds
# at speed=1 -- slow enough to actually watch collisions happen, not so slow
# it gets boring. Tune with the `speed` argument on Box.run()/Simulation.show().
SECONDS_PER_CROSSING = 6.0
MAX_LIVE_FRAMES = 3000  # safety cap on stored frames for very long/fast runs


def pick_sample_every(mean_speed, dt, Lx, Ly, *, fps=30, speed=1.0,
                      seconds_per_crossing=SECONDS_PER_CROSSING):
    """Choose how many physics steps to group into one displayed frame.

    Calibrated so a mean-speed particle crosses the shorter box dimension in
    about ``seconds_per_crossing / speed`` *wall-clock* seconds, independently
    of ``fps`` (raising fps makes playback smoother, not faster) and of the
    box/gas/temperature (a slow gas and a fast gas both look equally
    watchable). ``speed`` is a plain multiplier: 2.0 plays twice as fast,
    0.5 half as fast.
    """
    if mean_speed <= 0 or dt <= 0 or fps <= 0:
        return 50
    crossing_time = min(Lx, Ly) / mean_speed  # simulated seconds to cross the box
    wallclock_per_crossing = seconds_per_crossing / max(speed, 1e-9)
    sim_seconds_per_frame = crossing_time / (wallclock_per_crossing * fps)
    return max(1, round(sim_seconds_per_frame / dt))


def in_notebook() -> bool:
    """True if running inside a Jupyter/IPython kernel (not a plain shell)."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        return type(ip).__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# scene construction / per-frame updates
# --------------------------------------------------------------------------- #
def _setup_scene(Lx, Ly, radius, display_scale, *, vectors, color_by,
                 figsize, mean_speed, vmax):
    import matplotlib.pyplot as plt
    from matplotlib.collections import EllipseCollection

    Lx_nm, Ly_nm = Lx * 1e9, Ly * 1e9
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, Lx_nm)
    ax.set_ylim(0, Ly_nm)
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    diameters = 2.0 * np.asarray(radius) * 1e9 * display_scale  # nm
    coll = EllipseCollection(
        diameters, diameters, np.zeros_like(diameters), units="xy",
        offsets=np.zeros((len(diameters), 2)), offset_transform=ax.transData,
        edgecolors="black", linewidths=0.5, zorder=2,
    )
    if color_by == "speed":
        coll.set_cmap("plasma")
        coll.set_clim(0, vmax)
        coll.set_array(np.zeros(len(diameters)))
        cbar = fig.colorbar(coll, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("speed (m/s)")
    else:
        coll.set_facecolor("tab:blue")
    ax.add_collection(coll)

    quiv = None
    if vectors:
        # scale so a mean-speed arrow spans ~7% of the box
        target_nm = 0.07 * min(Lx_nm, Ly_nm)
        scale = (mean_speed / target_nm) if mean_speed > 0 else 1.0
        quiv = ax.quiver(
            np.zeros(len(diameters)), np.zeros(len(diameters)),
            np.zeros(len(diameters)), np.zeros(len(diameters)),
            angles="xy", scale_units="xy", scale=scale, width=0.004,
            color="black", zorder=3,
        )

    title = ax.set_title("")
    return fig, ax, coll, quiv, title


def _update_artists(coll, quiv, title, pos, vel, color_by, time_s):
    coll.set_offsets(_nm(pos))
    if color_by == "speed":
        coll.set_array(np.sqrt(np.sum(vel ** 2, axis=-1)))
    if quiv is not None:
        quiv.set_offsets(_nm(pos))
        quiv.set_UVC(vel[:, 0], vel[:, 1])
    if time_s is not None:
        title.set_text(f"t = {time_s * 1e12:.2f} ps")


# --------------------------------------------------------------------------- #
# live run (integrate + animate together)
# --------------------------------------------------------------------------- #
def live_run(system, *, dt, steps, sample_every, vectors, color_by, fps,
             figsize):
    """Integrate ``system`` while animating live; return the trajectory arrays.

    The system's own ``pos``/``vel`` arrays are advanced in place (same as the
    headless path), recording a frame every ``sample_every`` steps.
    """
    import time
    import matplotlib.pyplot as plt
    from IPython.display import display

    from . import analysis, kernels

    N = system.N
    n_frames = steps // sample_every + 1
    traj_pos = np.empty((n_frames, N, 2))
    traj_vel = np.empty((n_frames, N, 2))
    times = np.empty(n_frames)
    total_impulse = np.zeros(2)

    mean_v = float(analysis.speeds(system.vel).mean())
    vmax = 2.5 * mean_v if mean_v > 0 else 1.0
    fig, ax, coll, quiv, title = _setup_scene(
        system.Lx, system.Ly, system.radius, system.display_scale,
        vectors=vectors, color_by=color_by, figsize=figsize,
        mean_speed=mean_v, vmax=vmax,
    )

    traj_pos[0] = system.pos
    traj_vel[0] = system.vel
    times[0] = 0.0
    _update_artists(coll, quiv, title, system.pos, system.vel, color_by, 0.0)
    handle = display(fig, display_id=True)  # None outside a live kernel

    frame_budget = (1.0 / fps) if fps else 0.0
    for f in range(1, n_frames):
        t0 = time.time()
        imp = kernels._run_chunk(
            system.pos, system.vel, system.radius, system.inv_mass,
            system.Lx, system.Ly, dt, sample_every, system.periodic,
        )
        total_impulse += imp
        traj_pos[f] = system.pos
        traj_vel[f] = system.vel
        times[f] = f * sample_every * dt

        _update_artists(coll, quiv, title, system.pos, system.vel, color_by,
                        times[f])
        fig.canvas.draw_idle()
        if handle is not None:
            handle.update(fig)

        rest = frame_budget - (time.time() - t0)
        if rest > 0:
            time.sleep(rest)

    plt.close(fig)
    return traj_pos, traj_vel, times, total_impulse


# --------------------------------------------------------------------------- #
# replay a recorded trajectory
# --------------------------------------------------------------------------- #
def replay(sim, *, color_by="speed", vectors=False, fps=30, speed=1.0,
          figsize=(6, 6)):
    """Replay a stored :class:`Simulation` trajectory live (no HTML).

    ``speed`` rescales the wall-clock pacing of the (already recorded) frames:
    2.0 plays back twice as fast, 0.5 half as fast. It does not change what
    was recorded -- use ``speed`` on :meth:`Box.run` for that.
    """
    import time
    import matplotlib.pyplot as plt
    from IPython.display import display

    from . import analysis

    mean_v = float(analysis.speeds(sim.vel).mean())
    vmax = float(analysis.speeds(sim.vel).max()) if sim.vel.size else 1.0
    fig, ax, coll, quiv, title = _setup_scene(
        sim.Lx, sim.Ly, sim.radius, sim.display_scale,
        vectors=vectors, color_by=color_by, figsize=figsize,
        mean_speed=mean_v, vmax=vmax,
    )

    _update_artists(coll, quiv, title, sim.pos[0], sim.vel[0], color_by,
                    float(sim.times[0]))
    handle = display(fig, display_id=True)  # None outside a live kernel

    frame_budget = (1.0 / (fps * max(speed, 1e-9))) if fps else 0.0
    for f in range(1, sim.n_frames):
        t0 = time.time()
        _update_artists(coll, quiv, title, sim.pos[f], sim.vel[f], color_by,
                        float(sim.times[f]))
        fig.canvas.draw_idle()
        if handle is not None:
            handle.update(fig)
        rest = frame_budget - (time.time() - t0)
        if rest > 0:
            time.sleep(rest)

    plt.close(fig)


# --------------------------------------------------------------------------- #
# static histogram vs Maxwell-Boltzmann
# --------------------------------------------------------------------------- #
def histogram(speeds_array, *, temperature_K=None, mass_kg=None, dim=2,
              bins=40, ax=None, label="simulation"):
    """Plot a speed histogram, optionally overlaying Maxwell-Boltzmann."""
    import matplotlib.pyplot as plt

    from .analysis import maxwell_boltzmann_speed

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    speeds_array = np.asarray(speeds_array).ravel()
    ax.hist(speeds_array, bins=bins, density=True, alpha=0.6, label=label,
            color="tab:blue")
    if temperature_K is not None and mass_kg is not None:
        v = np.linspace(0, speeds_array.max() * 1.05, 400)
        mb = maxwell_boltzmann_speed(v, temperature_K, mass_kg, dim=dim)
        ax.plot(v, mb, "r-", lw=2,
                label=f"Maxwell-Boltzmann (T={temperature_K:.0f} K)")
    ax.set_xlabel("speed (m/s)")
    ax.set_ylabel("probability density")
    ax.legend()
    return ax
