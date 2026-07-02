"""Interactive matplotlib visualisation for bridgechem.

The whole trajectory is computed up front (fast, numba-accelerated), then
played back with play/pause/scrub controls via ``ipywidgets.Play`` -- the
standard Jupyter pattern for animations -- updating a single figure in place
via ``IPython.display`` (``display_id``). This works with the default
``%matplotlib inline`` backend: no interactive backend, no ``to_jshtml`` HTML
blob, and you can pause and scrub back to inspect a specific collision.

If ``ipywidgets`` isn't installed, playback falls back to a simple
forward-only autoplay (no pause/scrub). If there is no live notebook kernel at
all, nothing is displayed but the trajectory is still returned normally.

Particles are drawn as filled circles at their true collision size (times an
optional ``display_scale``), so what you see is what bounces. Velocity vectors
can be overlaid as arrows, and particles can be coloured by instantaneous
speed or by (fixed) mass -- handy for spotting a mixture set up with
:meth:`Box.set_mass`.
"""

from __future__ import annotations

import numpy as np

from .constants import AMU

VALID_COLOR_BY = (None, "speed", "mass")


def _nm(x):
    return np.asarray(x) * 1e9


# A mean-speed particle crossing the box takes this many wall-clock seconds
# at speed=1 -- slow enough to actually watch collisions happen, not so slow
# it gets boring. Tune with the `speed` argument on Box.run()/Simulation.show().
SECONDS_PER_CROSSING = 6.0
MAX_FRAMES = 3000  # safety cap on stored/played frames for very long/fast runs


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
                 figsize, mean_speed, color_static=None, vmin=0.0, vmax=1.0,
                 color_label=""):
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
    if color_by:
        coll.set_cmap("plasma")
        coll.set_clim(vmin, vmax)
        coll.set_array(color_static if color_static is not None
                       else np.zeros(len(diameters)))
        cbar = fig.colorbar(coll, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(color_label)
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
    # color_by == "mass" is static (set once at scene setup); nothing to do.
    if quiv is not None:
        quiv.set_offsets(_nm(pos))
        quiv.set_UVC(vel[:, 0], vel[:, 1])
    if time_s is not None:
        title.set_text(f"t = {time_s * 1e12:.2f} ps")


# --------------------------------------------------------------------------- #
# interactive playback (play / pause / scrub)
# --------------------------------------------------------------------------- #
def play(pos, vel, times, mass, radius, Lx, Ly, *, display_scale=1.0,
         vectors=False, color_by="speed", fps=30, speed=1.0, figsize=(6, 6)):
    """Play back a trajectory with play/pause/scrub controls (no HTML file).

    ``pos``/``vel`` are ``(n_frames, N, 2)`` arrays, ``times`` is ``(n_frames,)``.
    Uses ``ipywidgets.Play`` when available; falls back to a simple
    forward-only autoplay (no pause) if it isn't installed. Returns the
    ``ipywidgets.Play`` widget (for tests / further wiring), or ``None`` if
    nothing could be displayed (e.g. outside a notebook).
    """
    if color_by not in VALID_COLOR_BY:
        raise ValueError(f"color_by must be one of {VALID_COLOR_BY}")

    n_frames = pos.shape[0]
    color_static, vmin, vmax, color_label = None, 0.0, 1.0, ""
    color_by_render = color_by
    if color_by == "speed":
        all_speeds = np.sqrt(np.sum(vel ** 2, axis=-1))
        vmax = float(all_speeds.max()) if all_speeds.size else 1.0
        color_label = "speed (m/s)"
    elif color_by == "mass":
        mass_amu = np.asarray(mass) / AMU
        vmin, vmax = float(mass_amu.min()), float(mass_amu.max())
        color_label = "mass (amu)"
        if vmin == vmax:
            color_by_render = None  # uniform mass: nothing to colour by
        else:
            color_static = mass_amu

    mean_v = float(np.sqrt(np.sum(vel[0] ** 2, axis=-1)).mean()) if vel.size else 0.0
    fig, ax, coll, quiv, title = _setup_scene(
        Lx, Ly, radius, display_scale, vectors=vectors, color_by=color_by_render,
        figsize=figsize, mean_speed=mean_v, color_static=color_static,
        vmin=vmin, vmax=vmax, color_label=color_label,
    )
    _update_artists(coll, quiv, title, pos[0], vel[0], color_by_render,
                    float(times[0]) if times.size else None)

    import matplotlib.pyplot as plt
    try:
        from IPython.display import display
    except ImportError:
        plt.close(fig)
        return None  # nothing to display outside IPython

    handle = display(fig, display_id=True)  # None outside a live kernel
    # We keep updating `fig` in place via `handle` from here on, so drop it
    # from pyplot's own figure registry now -- otherwise IPython's inline
    # backend auto-displays every still-open figure again (as a frozen,
    # non-interactive duplicate) at the end of the cell.
    plt.close(fig)

    try:
        import ipywidgets as widgets
    except ImportError:
        _autoplay_fallback(fig, handle, coll, quiv, title, pos, vel, times,
                           color_by_render, fps, speed)
        return None

    interval_ms = (max(1, round(1000.0 / (fps * max(speed, 1e-9))))
                  if fps and fps > 0 else 1)
    play_widget = widgets.Play(min=0, max=n_frames - 1, step=1,
                               interval=interval_ms, value=0)
    slider = widgets.IntSlider(min=0, max=n_frames - 1, value=0,
                               description="frame")
    widgets.jslink((play_widget, "value"), (slider, "value"))

    def on_change(change):
        f = change["new"]
        _update_artists(coll, quiv, title, pos[f], vel[f], color_by_render,
                        float(times[f]) if times.size else None)
        fig.canvas.draw_idle()
        if handle is not None:
            handle.update(fig)

    play_widget.observe(on_change, names="value")
    display(widgets.HBox([play_widget, slider]))
    return play_widget


def _autoplay_fallback(fig, handle, coll, quiv, title, pos, vel, times,
                       color_by, fps, speed):
    """Forward-only autoplay used when ipywidgets isn't installed.

    ``fig`` is already closed (dropped from pyplot's figure registry) by the
    caller; we keep updating it in place via ``handle`` regardless.
    """
    import time

    print("Tip: install ipywidgets for play/pause/scrub controls "
          "(pip install ipywidgets).")
    frame_budget = (1.0 / (fps * max(speed, 1e-9))) if fps else 0.0
    for f in range(1, pos.shape[0]):
        t0 = time.time()
        _update_artists(coll, quiv, title, pos[f], vel[f], color_by,
                        float(times[f]) if times.size else None)
        fig.canvas.draw_idle()
        if handle is not None:
            handle.update(fig)
        rest = frame_budget - (time.time() - t0)
        if rest > 0:
            time.sleep(rest)


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
