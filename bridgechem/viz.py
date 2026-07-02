"""Interactive matplotlib visualisation for bridgechem.

The whole trajectory is computed up front (fast, numba-accelerated), then
played back with play/pause/scrub controls via ``ipywidgets.Play`` -- the
standard Jupyter pattern for animations.

Two rendering paths exist, chosen automatically:

* **Live canvas** (fast, smooth): if an interactive matplotlib backend is
  active (``ipympl``/``%matplotlib widget``, or the older ``nbagg``), the
  figure is a real live widget and each frame just restores a cached
  background + redraws the moving artists (matplotlib "blitting") -- no
  per-frame image encoding, so playback is as smooth as the browser can
  render. This is what you want for watching subtle phenomena (e.g. a slow
  phase transition) without the animation itself getting in the way.
* **Snapshot fallback** (default, works everywhere): with the default
  ``%matplotlib inline`` backend there's no live canvas to update -- every
  frame has to be rendered, PNG-encoded and shipped to the browser as a new
  image via ``IPython.display``. Blitting still cuts the *drawing* cost, but
  PNG encoding is an unavoidable ~tens-of-milliseconds-per-frame floor in
  this mode, so playback is capped to whatever this machine can actually
  encode+ship in time (measured automatically) rather than promising more.

Either way: no ``to_jshtml`` HTML blob, and you can pause and scrub back to
inspect a specific moment. If ``ipywidgets`` isn't installed, playback falls
back further to a simple forward-only autoplay (no pause/scrub). If there is
no live notebook kernel at all, nothing is displayed but the trajectory is
still returned normally.

Particles are drawn as filled circles at their true collision size (times an
optional ``display_scale``), so what you see is what bounces. Velocity
vectors can be overlaid as arrows, and particles can be coloured by
instantaneous speed or by (fixed) mass -- handy for spotting a mixture set up
with :meth:`Box.set_mass`. The title shows elapsed time and the instantaneous
temperature.
"""

from __future__ import annotations

import numpy as np

from .constants import AMU
from .analysis import temperature as _temperature

VALID_COLOR_BY = (None, "speed", "mass")


def _nm(x):
    return np.asarray(x) * 1e9


# A mean-speed particle crossing the box takes this many wall-clock seconds
# at speed=1 -- slow enough to actually watch collisions happen, not so slow
# it gets boring. Tune with the `speed` argument on Box.run()/Simulation.show().
SECONDS_PER_CROSSING = 6.0
MAX_FRAMES = 3000  # safety cap on stored/played frames for very long/fast runs


def pick_sample_every(mean_speed, dt, Lx, Ly, *, fps=15, speed=1.0,
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


def _is_interactive_backend() -> bool:
    """True if a live matplotlib canvas (ipympl/nbagg) is active.

    These backends support real blitting over a persistent connection, so
    playback can be smooth (no per-frame PNG encoding). Enable with
    ``pip install ipympl`` and ``%matplotlib widget`` at the top of a notebook
    (before creating any bridgechem simulations).
    """
    try:
        import matplotlib
        backend = matplotlib.get_backend().lower()
        return "ipympl" in backend or "nbagg" in backend
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

    # NOT animated=True: that flag tells normal draw()/savefig() calls to
    # skip this artist (expecting it to be blitted manually only) -- but the
    # snapshot path's initial frame and every PNG re-encode go through a
    # normal draw(), so the title (time/temperature) would silently vanish.
    title = ax.set_title("")
    return fig, ax, coll, quiv, title


def _update_artists(coll, quiv, title, pos, vel, color_by, time_s, *,
                    mass=None, dim=2):
    coll.set_offsets(_nm(pos))
    if color_by == "speed":
        coll.set_array(np.sqrt(np.sum(vel ** 2, axis=-1)))
    # color_by == "mass" is static (set once at scene setup); nothing to do.
    if quiv is not None:
        quiv.set_offsets(_nm(pos))
        quiv.set_UVC(vel[:, 0], vel[:, 1])
    label = ""
    if time_s is not None:
        label += f"t = {time_s * 1e12:.2f} ps"
    if mass is not None:
        T = _temperature(vel, mass, dim=dim)
        label += f"   T = {T:.1f} K"
    title.set_text(label)


def _draw_artists(canvas, coll, quiv, title):
    """Redraw just the dynamic artists (for blitting -- must follow a
    ``restore_region`` and precede a ``blit``).

    Draws via a renderer fetched from ``canvas`` directly (not
    ``ax.draw_artist``, which looks up the renderer through
    ``ax.get_figure().canvas`` -- unreliable once ``plt.close(fig)`` has
    replaced that attribute with a bare placeholder, see :func:`_play_snapshot`).
    """
    renderer = canvas.get_renderer()
    coll.draw(renderer)
    if quiv is not None:
        quiv.draw(renderer)
    title.draw(renderer)


# --------------------------------------------------------------------------- #
# interactive playback (play / pause / scrub)
# --------------------------------------------------------------------------- #
def play(pos, vel, times, mass, radius, Lx, Ly, *, display_scale=1.0,
         vectors=False, color_by="speed", fps=15, speed=1.0, dim=2,
         figsize=(6, 6)):
    """Play back a trajectory with play/pause/scrub controls (no HTML file).

    ``pos``/``vel`` are ``(n_frames, N, 2)`` arrays, ``times`` is ``(n_frames,)``.
    Uses ``ipywidgets.Play`` when available; falls back to a simple
    forward-only autoplay (no pause) if it isn't installed. Returns the
    ``ipywidgets.Play`` widget (for tests / further wiring), or ``None`` if
    nothing could be displayed (e.g. outside a notebook).

    Rendering picks the fastest path available -- see the module docstring.
    With the default inline backend, redrawing a matplotlib figure and
    shipping it to the browser as a PNG has real, fairly fixed overhead per
    frame; ``fps`` is a *target* there, capped to what this machine can
    actually render in time. Install ``ipympl`` and run ``%matplotlib
    widget`` before creating simulations for genuinely smooth playback.
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

    def frame_update(f):
        _update_artists(coll, quiv, title, pos[f], vel[f], color_by_render,
                        float(times[f]) if times.size else None, mass=mass,
                        dim=dim)

    import matplotlib.pyplot as plt
    try:
        from IPython.display import display
    except ImportError:
        plt.close(fig)
        return None  # nothing to display outside IPython

    # Capture a *clean* background for blitting -- with the dynamic artists
    # hidden, so no frame's particle positions get baked into what's supposed
    # to be the static part of the scene (axes/labels/colorbar only). Doing
    # this with real particle data already drawn (the previous bug) meant
    # every later frame showed a ghost of whichever frame got drawn first,
    # wherever its particles didn't exactly overlap the new ones.
    #
    # The captured/restored/blitted region is the *whole figure* (fig.bbox),
    # not just the axes (ax.bbox): the title sits above the axes' plotting
    # area, outside ax.bbox, so restoring/blitting only ax.bbox never erases
    # the previous frame's title text -- new text just gets drawn over the
    # old, leaving stray leftover digits wherever the two don't overlap.
    coll.set_visible(False)
    if quiv is not None:
        quiv.set_visible(False)
    title.set_visible(False)
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)
    coll.set_visible(True)
    if quiv is not None:
        quiv.set_visible(True)
    title.set_visible(True)
    frame_update(0)

    if _is_interactive_backend():
        return _play_live_canvas(fig, coll, quiv, title, background,
                                 frame_update, n_frames, fps, speed)
    return _play_snapshot(fig, coll, quiv, title, background, frame_update,
                          n_frames, fps, speed)


def _play_live_canvas(fig, coll, quiv, title, background, frame_update,
                      n_frames, fps, speed):
    """Fast path: a live interactive canvas (ipympl/nbagg) updated by
    blitting -- no per-frame image encoding, so this can be genuinely smooth.
    """
    from IPython.display import display

    canvas = fig.canvas

    def redraw():
        canvas.restore_region(background)
        _draw_artists(canvas, coll, quiv, title)
        canvas.blit(fig.bbox)

    redraw()  # paint frame 0 (already populated by play()) onto the canvas
    display(canvas)

    try:
        import ipywidgets as widgets
    except ImportError:
        print("Tip: install ipywidgets for play/pause/scrub controls "
              "(pip install ipywidgets).")
        _autoplay_loop(n_frames, fps, speed, frame_update, redraw)
        return None

    interval_ms = (max(1, round(1000.0 / (fps * max(speed, 1e-9))))
                  if fps and fps > 0 else 33)
    play_widget = widgets.Play(min=0, max=n_frames - 1, step=1,
                               interval=interval_ms, value=0)
    slider = widgets.IntSlider(min=0, max=n_frames - 1, value=0,
                               description="frame")
    widgets.jslink((play_widget, "value"), (slider, "value"))

    def on_change(change):
        frame_update(change["new"])
        redraw()

    play_widget.observe(on_change, names="value")
    display(widgets.HBox([play_widget, slider]))
    return play_widget


def _play_snapshot(fig, coll, quiv, title, background, frame_update,
                   n_frames, fps, speed):
    """Fallback path: no live canvas, so each frame is rendered, PNG-encoded
    and shipped via ``IPython.display`` -- blitting still speeds up the
    drawing step, but PNG encoding is an unavoidable per-frame floor here.
    """
    import io
    import time
    import matplotlib.pyplot as plt
    from IPython.display import display

    canvas = fig.canvas  # keep our own reference: plt.close() below replaces
                        # fig.canvas with a bare placeholder that can draw
                        # (via savefig) but has lost the blitting methods.

    # Measure the real redraw+encode cost directly via the Agg pipeline (not
    # by timing display() itself, which can short-circuit with no cost when
    # there's no live frontend to publish to -- e.g. outside a real kernel).
    # Frame 0 is already populated (by play()), so this also renders it.
    t0 = time.time()
    canvas.draw()
    fig.savefig(io.BytesIO(), format="png")
    render_time = time.time() - t0

    handle = display(fig, display_id=True)  # None outside a live kernel
    # We keep updating `fig` in place via `handle` from here on, so drop it
    # from pyplot's own figure registry now -- otherwise IPython's inline
    # backend auto-displays every still-open figure again (as a frozen,
    # non-interactive duplicate) at the end of the cell.
    plt.close(fig)

    def redraw():
        canvas.restore_region(background)
        _draw_artists(canvas, coll, quiv, title)
        canvas.blit(fig.bbox)
        if handle is not None:
            handle.update(fig)

    try:
        import ipywidgets as widgets
    except ImportError:
        print("Tip: install ipywidgets for play/pause/scrub controls "
              "(pip install ipywidgets), and ipympl (+ %matplotlib widget) "
              "for smoother playback.")
        _autoplay_loop(n_frames, fps, speed, frame_update, redraw,
                       frame_budget=_frame_budget(fps, speed, render_time))
        return None

    print("Tip: install ipympl and run %matplotlib widget at the top of "
          "your notebook (before creating simulations) for smoother, "
          "faster playback.")

    # Never tick faster than this machine can actually redraw+encode a frame
    # (measured above from the first frame), with a safety margin so a
    # slightly-more-expensive later frame doesn't immediately fall behind.
    achievable_fps = 0.8 / max(render_time, 1e-3)
    effective_fps = min(fps, achievable_fps) if fps and fps > 0 else achievable_fps
    interval_ms = max(1, round(1000.0 / (effective_fps * max(speed, 1e-9))))
    play_widget = widgets.Play(min=0, max=n_frames - 1, step=1,
                               interval=interval_ms, value=0)
    slider = widgets.IntSlider(min=0, max=n_frames - 1, value=0,
                               description="frame")
    widgets.jslink((play_widget, "value"), (slider, "value"))

    def on_change(change):
        frame_update(change["new"])
        redraw()

    play_widget.observe(on_change, names="value")
    display(widgets.HBox([play_widget, slider]))
    return play_widget


def _frame_budget(fps, speed, render_time):
    if not fps or fps <= 0:
        return 0.0
    return max(1.0 / (fps * max(speed, 1e-9)), render_time)


def _autoplay_loop(n_frames, fps, speed, frame_update, redraw, frame_budget=0.0):
    """Forward-only autoplay used when ipywidgets isn't installed."""
    import time

    for f in range(1, n_frames):
        t0 = time.time()
        frame_update(f)
        redraw()
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
