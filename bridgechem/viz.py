"""Interactive matplotlib visualisation for bridgechem.

The whole trajectory is computed up front (fast, numba-accelerated), then
played back with play/pause/scrub controls via ``ipywidgets.Play``.

**Live playback.** Inside a real Jupyter kernel, :func:`play` switches to the
``ipympl`` (``%matplotlib widget``) backend the first time it's called (unless
you've already picked a backend yourself) and drives a *live* canvas instead
of the default ``%matplotlib inline`` behaviour of re-rendering the whole
figure to a PNG and shipping a fresh image every frame. Each tick does a
direct, synchronous ``fig.canvas.draw()`` -- see :func:`_make_live_push` for
why that's the reliable choice over both ``draw_idle()`` and manual
blitting, which were each tried first and each misbehaved specifically on
ipympl. Outside a live kernel (a script, a test, or if ``ipympl`` isn't
installed) playback falls back to the old PNG-per-frame approach, which
still works everywhere.

**2D vs 3D.** The box's dimension (``len(size)`` at construction) decides the
scene: a 2D box gets a flat view, a 3D box gets a real, mouse-rotatable
``Axes3D`` scatter -- an actual 3D scene, not a 2D projection that drops the
z-coordinate (a full-3D box no longer hides particles behind each other the
way a projection would; nothing is being flattened, only drawn in
perspective). Pass ``slab=<nm>`` to force the old thin-slice 2D view of a 3D
run instead, when you specifically want to see collisions in a single plane
rather than a rotatable overview.

If ``ipywidgets`` isn't installed, playback falls back to a simple
forward-only autoplay (no pause/scrub). If there is no live notebook kernel at
all, nothing is displayed but the trajectory is still returned normally.

**Particle size.** Particles are drawn at their true collision radius by
default -- except when that would be a bad picture: a `packing`-derived
radius spread over very few particles can demand a size approaching a third
of the box (especially in 3D), and a physically real radius (a gas's actual
van der Waals radius, a couple of angstrom) is too small to see at all in a
box sized for visibility. :func:`_auto_size_factor` keeps the drawn size
within a sane range for both ends without touching the *relative* spacing
between particles, so a dense (liquid-like) arrangement still reads as
denser than a sparse (gas-like) one -- only the common scale changes.
``display_scale`` still multiplies on top of this, uncapped, so you can push
the result bigger or smaller yourself.
"""

from __future__ import annotations

import time

import numpy as np

from .constants import AMU

VALID_COLOR_BY = (None, "speed", "mass")


def _nm(x):
    return np.asarray(x) * 1e9


def _project(pos, slab_window=None):
    """Screen coordinates: the x-y components, in nm.

    Used for the 2D scene, and for the ``slab``-restricted 2D view of a 3D
    box. If ``slab_window`` is given as ``(z_low, z_high)`` in metres,
    particles outside that slice are moved to NaN, which matplotlib simply
    does not draw -- so what remains on screen is a thin sheet of gas whose
    apparent collisions are real ones.
    """
    xy = _nm(pos[:, :2])
    if slab_window is not None and pos.shape[1] > 2:
        outside = (pos[:, 2] < slab_window[0]) | (pos[:, 2] > slab_window[1])
        xy = xy.copy()
        xy[outside] = np.nan
    return xy


def _slab_window(L, slab):
    """A slab of thickness ``slab`` nm through the middle of the box (metres)."""
    if slab is None or len(L) < 3:
        return None
    half = 0.5 * slab * 1e-9
    middle = 0.5 * float(L[2])
    return (middle - half, middle + half)


# A mean-speed particle crossing the box takes this many wall-clock seconds
# at speed=1 -- slow enough to actually watch collisions happen, not so slow
# it gets boring. Tune with the `speed` argument on Box.run()/Simulation.show().
SECONDS_PER_CROSSING = 6.0
MAX_FRAMES = 3000  # safety cap on stored/played frames for very long/fast runs

# Bounds on how big a particle's *drawn* diameter is allowed to be, as a
# fraction of the box's shortest side -- see _auto_size_factor().
MIN_DISPLAY_DIAM_FRAC = 0.015
MAX_DISPLAY_DIAM_FRAC_2D = 0.12
MAX_DISPLAY_DIAM_FRAC_3D = 0.08  # 3D reads busier at the same relative size


def _auto_size_factor(radius, L, max_frac, min_frac=MIN_DISPLAY_DIAM_FRAC):
    """Multiplier that keeps the drawn diameter within a sane visible range.

    Particles are drawn at their true collision size by default (factor 1) --
    *unless* that would be imperceptibly small or absurdly large relative to
    the box, in which case the drawn size is scaled toward a floor or
    ceiling instead. Both ends are real cases here: ``radius`` defaults to a
    `packing` fraction of the box (see :class:`bridgechem.Box`), which for a
    handful of particles in a big 3D box can demand a radius approaching a
    third of the box width -- and a *physically real* particle radius (a
    gas's actual van der Waals radius, a couple of angstrom) is genuinely too
    small to see at all in a box sized for visibility.

    The factor is uniform across every particle in the run (there's only one
    ``radius`` to begin with -- bridgechem doesn't support per-particle
    radii), so it never distorts *relative* crowding: a denser arrangement
    still reads as denser than a sparser one, only the common scale changes.
    This runs before the user's own ``display_scale``, which still applies
    on top uncapped -- an explicit request to draw things bigger or smaller
    is never second-guessed.
    """
    r = float(np.max(radius))
    shortest = float(np.min(L))
    if r <= 0 or shortest <= 0:
        return 1.0
    natural_frac = 2.0 * r / shortest
    target_frac = min(max(natural_frac, min_frac), max_frac)
    return target_frac / natural_frac


def pick_sample_every(mean_speed, dt, L, *, fps=15, speed=1.0,
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
    crossing_time = float(np.min(L)) / mean_speed  # simulated s to cross the box
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
# backend management: switch to a live (ipympl) canvas when we can
# --------------------------------------------------------------------------- #
_backend_switch_attempted = False


def _is_live_backend() -> bool:
    """True if matplotlib is already on an interactive-canvas backend.

    ``ipympl``/``nbAgg``/``WebAgg`` all push incremental updates to a canvas
    that stays alive in the notebook, as opposed to ``inline``, which
    re-renders a static PNG per figure. That live canvas is what lets us
    blit (2D) or just redraw in place (3D) instead of re-encoding an image
    every frame.
    """
    import matplotlib
    backend = matplotlib.get_backend().lower()
    return any(tag in backend for tag in ("ipympl", "nbagg", "webagg"))


def _ensure_interactive_backend() -> bool:
    """Switch to the ``ipympl`` backend if we safely can; return whether live.

    Only acts inside a real Jupyter kernel (:func:`in_notebook`), and only
    when the backend is still the plain Jupyter default -- so a user who has
    deliberately chosen a backend (``%matplotlib inline`` explicitly, a GUI
    backend, ``Agg`` for a headless script, ...) is never overridden. Tried
    at most once per process: if ``ipympl`` isn't installed we don't keep
    retrying (or re-printing the tip) on every subsequent call.
    """
    global _backend_switch_attempted
    if _is_live_backend():
        return True
    if not in_notebook():
        return False
    if _backend_switch_attempted:
        return False
    _backend_switch_attempted = True

    import matplotlib
    if "inline" not in matplotlib.get_backend().lower():
        return False  # user picked something else on purpose; leave it alone

    try:
        import ipympl  # noqa: F401
        import matplotlib.pyplot as plt
        plt.switch_backend("module://ipympl.backend_nbagg")
        return True
    except Exception:
        print("Tip: install ipympl for smooth, rotatable playback "
              "(pip install ipympl), then restart the kernel.")
        return False


# --------------------------------------------------------------------------- #
# 2D scene: particles as an EllipseCollection (true collision-size circles)
# --------------------------------------------------------------------------- #
def _setup_scene(L, radius, display_scale, *, vectors, color_by,
                 figsize, mean_speed, color_static=None, vmin=0.0, vmax=1.0,
                 color_label=""):
    import matplotlib.pyplot as plt
    from matplotlib.collections import EllipseCollection

    Lx_nm, Ly_nm = float(L[0]) * 1e9, float(L[1]) * 1e9
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, Lx_nm)
    ax.set_ylim(0, Ly_nm)
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    auto = _auto_size_factor(radius, L, MAX_DISPLAY_DIAM_FRAC_2D)
    diameters = 2.0 * np.asarray(radius) * 1e9 * display_scale * auto  # nm
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


def _update_artists(coll, quiv, title, pos, vel, color_by, time_s,
                    slab_window=None):
    xy = _project(pos, slab_window)
    coll.set_offsets(xy)
    if color_by == "speed":
        coll.set_array(np.sqrt(np.sum(vel ** 2, axis=-1)))
    # color_by == "mass" is static (set once at scene setup); nothing to do.
    if quiv is not None:
        quiv.set_offsets(xy)
        quiv.set_UVC(vel[:, 0], vel[:, 1])
    if time_s is not None:
        title.set_text(f"t = {time_s * 1e12:.2f} ps")


# --------------------------------------------------------------------------- #
# 3D scene: a real, rotatable Axes3D scatter -- not a projection
# --------------------------------------------------------------------------- #
def _setup_scene_3d(L, radius, display_scale, *, vectors, color_by,
                    figsize, mean_speed, color_static=None, vmin=0.0, vmax=1.0,
                    color_label=""):
    import matplotlib.pyplot as plt

    Lx_nm, Ly_nm, Lz_nm = (float(x) * 1e9 for x in L)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(0, Lx_nm)
    ax.set_ylim(0, Ly_nm)
    ax.set_zlim(0, Lz_nm)
    ax.set_box_aspect((Lx_nm, Ly_nm, Lz_nm))  # box drawn true-to-scale, not cubical
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_zlabel("z (nm)")

    # matplotlib's 3D scatter sizes markers in points^2 (screen space), not
    # data space -- unlike the 2D EllipseCollection it cannot draw a true
    # data-space circle, so size can't track radius exactly as you zoom or
    # rotate. This picks a size that looks right at the initial view (a
    # known mplot3d limitation, not a bug).
    points_per_nm = (figsize[0] * 72.0) / max(Lx_nm, Ly_nm, Lz_nm)
    auto = _auto_size_factor(radius, L, MAX_DISPLAY_DIAM_FRAC_3D)
    diam_nm = 2.0 * np.asarray(radius) * 1e9 * display_scale * auto
    sizes = (diam_nm * points_per_nm) ** 2

    n = len(radius)
    zeros = np.zeros(n)
    # depthshade=True darkens far particles -- a cheap but real depth cue on
    # top of the perspective projection, which is what actually makes this
    # read as 3D rather than a cloud of same-looking dots.
    coll = ax.scatter(zeros, zeros, zeros, s=sizes, edgecolors="black",
                      linewidths=0.5, depthshade=True)
    if color_by:
        coll.set_cmap("plasma")
        coll.set_clim(vmin, vmax)
        coll.set_array(color_static if color_static is not None
                       else np.zeros(n))
        cbar = fig.colorbar(coll, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(color_label)
    else:
        coll.set_color("tab:blue")

    title = ax.set_title("")
    return fig, ax, coll, None, title


def _update_artists_3d(ax, coll, quiv, title, pos, vel, color_by, time_s,
                       vectors, mean_speed, L):
    """Update the 3D scene for one frame; returns the (possibly new) quiver.

    mplot3d offers no in-place update for a 3D quiver, so unlike the 2D
    ``quiv.set_UVC``, a 3D velocity-vector quiver has to be removed and
    recreated every frame. That's cheap next to a full redraw but not free,
    so leave ``vectors=False`` (the default) for the smoothest 3D playback.
    """
    xyz = _nm(pos)
    coll._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
    if color_by == "speed":
        coll.set_array(np.sqrt(np.sum(vel ** 2, axis=-1)))

    if quiv is not None:
        quiv.remove()
    new_quiv = None
    if vectors:
        Lx_nm = float(L[0]) * 1e9
        target_nm = 0.07 * Lx_nm
        scale = (target_nm / mean_speed) if mean_speed > 0 else 1.0
        new_quiv = ax.quiver(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            vel[:, 0] * scale, vel[:, 1] * scale, vel[:, 2] * scale,
            color="black", linewidth=1.0, arrow_length_ratio=0.3,
        )
    if time_s is not None:
        title.set_text(f"t = {time_s * 1e12:.2f} ps")
    return new_quiv


# --------------------------------------------------------------------------- #
# live-canvas push (2D and 3D alike): a direct, synchronous full redraw
# --------------------------------------------------------------------------- #
def _make_live_push(fig):
    """Set up the live-canvas push mechanism. Returns ``(push, first_render_time)``.

    Deliberately a plain, full ``fig.canvas.draw()`` every frame, not
    ``draw_idle()`` and not manual blitting (``restore_region``/
    ``draw_artist``/``blit``) -- both were tried and both misbehaved on the
    ipympl backend specifically:

    - ``draw_idle()`` doesn't render anything itself; it only sends a
      "please redraw" request to the *frontend* and waits for it to ask
      back (see ``FigureCanvasWebAggCore.draw_idle``/``send_event``). That
      round trip can stall under a fast run of ticks -- playback silently
      freezes while the driving widget's own counter keeps advancing.
    - Manual blitting marks the particle artists ``animated=True`` so the
      normal draw traversal skips them, then paints just those artists back
      in via ``draw_artist`` each frame. That part works fine on the Python
      side (the rendered buffer itself never accumulates anything stale --
      confirmed by inspecting it directly), but playback on ipympl still
      visibly filled up with every past frame's particles, growing steadily
      slower as it went. However that happens exactly, it isn't reliable
      here, and diagnosing it further would mean debugging ipympl's JS/comm
      diff protocol blind, without a real browser to inspect.

    A full ``draw()`` sidesteps the *rendering* half of both problems above:
    nothing is marked animated, so there's no "which artists does the normal
    draw skip" bookkeeping to get wrong, and it renders and pushes directly
    like ``blit()`` does. But switching from blit() to draw() alone turned
    out not to be enough -- particles kept visibly trailing regardless,
    which makes sense in hindsight: draw() and blit() share the exact same
    push mechanism *downstream* of rendering (``_png_is_old = True`` then
    ``manager.refresh_all()``), so whatever caused the trailing was never in
    which one triggered it. ``refresh_all()`` sends a *diff* against the
    previous frame by default (``get_diff_image()``), computed and applied
    asynchronously over the kernel comm -- if a frontend hasn't finished
    applying frame N-1's diff before frame N's arrives (very possible once
    ticks start arriving every few tens of ms), frame N's diff, computed
    against a buffer state the frontend hasn't reached yet, can land wrong
    without ever raising an error -- a mismatched diff overlaying stale
    content looks exactly like a trail. A full frame doesn't have this
    failure mode: whichever one a client processes last simply replaces the
    canvas outright, no matter what order they arrive in. Forcing
    ``_force_full`` before every push trades a little bandwidth (a full
    frame instead of a diff -- trivial at this figure size) to remove that
    failure mode entirely, rather than relying on frames arriving in order.
    mplot3d doesn't support ``draw_artist``/blitting at all, so a full
    ``draw()`` was already the only option for 3D; using it for 2D too keeps
    both paths identical and equally trustworthy.
    """
    def _draw():
        fig.canvas._force_full = True
        fig.canvas.draw()

    t0 = time.time()
    _draw()
    render_time = time.time() - t0

    return _draw, render_time


# --------------------------------------------------------------------------- #
# interactive playback (play / pause / scrub)
# --------------------------------------------------------------------------- #
def play(pos, vel, times, mass, radius, L, *, display_scale=1.0,
         vectors=False, color_by="speed", fps=15, speed=1.0, figsize=(6, 6),
         slab=None):
    """Play back a trajectory with play/pause/scrub controls (no HTML file).

    ``pos``/``vel`` are ``(n_frames, N, dim)`` arrays and ``L`` is ``(dim,)``.
    A 3D run gets a real, rotatable 3D scene by default; pass ``slab`` (nm) to
    instead view a thin 2D slice through the middle of the box, where every
    apparent collision on screen really is one.

    Uses ``ipywidgets.Play`` when available; falls back to a simple
    forward-only autoplay (no pause) if it isn't installed. Returns the
    ``ipywidgets.Play`` widget (for tests / further wiring), or ``None`` if
    nothing could be displayed (e.g. outside a notebook).

    Inside a live Jupyter kernel this switches to the ``ipympl`` backend (see
    :func:`_ensure_interactive_backend`) and drives its live canvas directly
    with a synchronous redraw each frame (see :func:`_make_live_push`) --
    skipping the per-frame PNG encode/transfer that caused stutter under the
    plain inline backend. Outside a live kernel (a script, a test, or
    without ``ipympl``) playback falls back to that PNG-per-frame approach,
    unchanged: we measure how long the first frame actually takes to
    render+encode on this machine and cap the ``Play`` widget's tick rate
    accordingly, so it never queues frames faster than they can be drawn.
    """
    if color_by not in VALID_COLOR_BY:
        raise ValueError(f"color_by must be one of {VALID_COLOR_BY}")

    n_frames = pos.shape[0]
    L = np.atleast_1d(np.asarray(L, dtype=float))
    dim = L.size
    full_3d = dim == 3 and slab is None

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
    slab_window = _slab_window(L, slab)
    live = _ensure_interactive_backend()

    scene_kwargs = dict(display_scale=display_scale, vectors=vectors,
                        color_by=color_by_render, figsize=figsize,
                        mean_speed=mean_v, color_static=color_static,
                        vmin=vmin, vmax=vmax, color_label=color_label)
    if full_3d:
        fig, ax, coll, quiv, title = _setup_scene_3d(L, radius, **scene_kwargs)
    else:
        fig, ax, coll, quiv, title = _setup_scene(L, radius, **scene_kwargs)

    quiv_holder = [quiv]

    def update(f):
        t = float(times[f]) if times.size else None
        if full_3d:
            quiv_holder[0] = _update_artists_3d(
                ax, coll, quiv_holder[0], title, pos[f], vel[f],
                color_by_render, t, vectors, mean_v, L)
        else:
            _update_artists(coll, quiv_holder[0], title, pos[f], vel[f],
                            color_by_render, t, slab_window)

    update(0)

    import matplotlib.pyplot as plt
    try:
        from IPython.display import display
    except ImportError:
        plt.close(fig)
        return None  # nothing to display outside IPython

    if live:
        # Live canvas: no PNG round trip -- see _make_live_push. Frame 0 must
        # be rendered *before* display(fig.canvas): the widget mounts showing
        # whatever's already in the canvas buffer, so displaying it first
        # means it mounts blank and only catches up on the next tick.
        push_frame, render_time = _make_live_push(fig)

        def push(f):
            push_frame()

        display(fig.canvas)
    else:
        # Static (inline / headless) fallback: redraw the whole figure to a
        # PNG and push it as a fresh display update. Real, fairly fixed cost
        # (tens of ms) with nothing to do with the physics -- ticking the
        # `Play` widget faster than this machine can redraw+encode just makes
        # frames queue up, which looks like stutter or out-of-order playback.
        import io

        t0 = time.time()
        fig.canvas.draw()
        fig.savefig(io.BytesIO(), format="png")
        render_time = time.time() - t0

        handle = display(fig, display_id=True)  # None outside a live kernel
        # We keep updating `fig` in place via `handle` from here on, so drop
        # it from pyplot's own figure registry now -- otherwise IPython's
        # inline backend auto-displays every still-open figure again (as a
        # frozen, non-interactive duplicate) at the end of the cell.
        plt.close(fig)

        def push(f):
            fig.canvas.draw_idle()
            if handle is not None:
                handle.update(fig)

    try:
        import ipywidgets as widgets
    except ImportError:
        _autoplay_fallback(update, push, n_frames, fps, speed)
        return None

    # Never tick faster than this machine can actually push a frame (measured
    # above), with a safety margin so a slightly-more-expensive later frame
    # doesn't immediately fall behind.
    achievable_fps = 0.8 / max(render_time, 1e-3)
    effective_fps = min(fps, achievable_fps) if fps and fps > 0 else achievable_fps
    interval_ms = max(1, round(1000.0 / (effective_fps * max(speed, 1e-9))))
    play_widget = widgets.Play(min=0, max=n_frames - 1, step=1,
                               interval=interval_ms, value=0)
    slider = widgets.IntSlider(min=0, max=n_frames - 1, value=0,
                               description="frame")
    widgets.jslink((play_widget, "value"), (slider, "value"))

    def on_change(change):
        f = change["new"]
        update(f)
        push(f)

    play_widget.observe(on_change, names="value")
    display(widgets.HBox([play_widget, slider]))
    return play_widget


def _autoplay_fallback(update, push, n_frames, fps, speed):
    """Forward-only autoplay used when ipywidgets isn't installed."""
    print("Tip: install ipywidgets for play/pause/scrub controls "
          "(pip install ipywidgets).")
    frame_budget = (1.0 / (fps * max(speed, 1e-9))) if fps else 0.0
    for f in range(1, n_frames):
        t0 = time.time()
        update(f)
        push(f)
        rest = frame_budget - (time.time() - t0)
        if rest > 0:
            time.sleep(rest)


# --------------------------------------------------------------------------- #
# live playback: step and render together, as you watch (animate="live")
# --------------------------------------------------------------------------- #
def play_live(box, *, dt, steps, sample_every, vectors=False,
             color_by="speed", fps=15, speed=1.0, display_scale=None,
             figsize=(6, 6), thermostat=False, T_start=0.0, T_target=0.0,
             rate=0.0):
    """Step and render live, one frame at a time -- nothing precomputed.

    Unlike :func:`play`, there's no trajectory to hand in: each displayed
    frame is ``sample_every`` physics steps run right before it's drawn,
    using -- and mutating -- the box's *current* live state
    (``box.pos``/``box.vel``), exactly like :meth:`bridgechem.Box.advance`.
    ``steps`` is the total step budget for the run (playback stops once
    reached), matching the ``steps``/``t`` already resolved by
    :meth:`bridgechem.Box.run`.

    This needs a live Jupyter kernel with ``ipympl`` -- there's no
    meaningful "replay" fallback for a run that's never actually recorded
    up front, so this raises rather than silently doing something that
    wouldn't look live.

    ``thermostat``/``T_start``/``T_target``/``rate`` carry a pending
    :meth:`bridgechem.Box.set_temperature` ramp through to each
    :meth:`~bridgechem.Box.advance` call -- so cooling a gas to watch it
    condense works the same way live as it does precomputed.

    Returns a :class:`bridgechem.simulation.LiveRun` handle: its ``Play``
    widget is shown *without* a slider (there's no "scrub back" for a live
    physics step -- only :meth:`~bridgechem.simulation.LiveRun.pause` /
    :meth:`~bridgechem.simulation.LiveRun.resume`), and ``.simulation``
    gives a normal :class:`Simulation` over whatever's been recorded so
    far, at any point, complete with full scrubbing over that recording.
    """
    from .simulation import LiveRun

    if not _ensure_interactive_backend():
        raise RuntimeError(
            "animate='live' needs a live Jupyter kernel with ipympl "
            "installed -- there's no meaningful way to stream a live "
            "simulation into a static image. Use the default precomputed "
            "mode instead (animate=True), or install ipympl "
            "(pip install ipympl) and run this in a real notebook kernel."
        )
    if color_by not in VALID_COLOR_BY:
        raise ValueError(f"color_by must be one of {VALID_COLOR_BY}")

    L = np.atleast_1d(np.asarray(box.L, dtype=float))
    dim = L.size
    full_3d = dim == 3
    ds = display_scale if display_scale is not None else box.display_scale
    n_frames = int(steps) // int(sample_every) + 1

    color_static, vmin, vmax, color_label = None, 0.0, 1.0, ""
    color_by_render = color_by
    if color_by == "speed":
        # No full trajectory to draw the true max from yet, so estimate from
        # the current speed distribution with generous headroom -- a
        # heating/cooling run can otherwise saturate or wash out the colour
        # scale as it goes, since (unlike precomputed playback) this can't
        # be recalibrated from data that doesn't exist yet.
        current_speeds = np.sqrt(np.sum(box.vel ** 2, axis=-1))
        vmax = 2.0 * float(current_speeds.max()) if current_speeds.size else 1.0
        color_label = "speed (m/s)"
    elif color_by == "mass":
        mass_amu = np.asarray(box.mass) / AMU
        vmin, vmax = float(mass_amu.min()), float(mass_amu.max())
        color_label = "mass (amu)"
        if vmin == vmax:
            color_by_render = None
        else:
            color_static = mass_amu

    mean_v = (float(np.sqrt(np.sum(box.vel ** 2, axis=-1)).mean())
             if box.vel.size else 0.0)

    scene_kwargs = dict(display_scale=ds, vectors=vectors,
                        color_by=color_by_render, figsize=figsize,
                        mean_speed=mean_v, color_static=color_static,
                        vmin=vmin, vmax=vmax, color_label=color_label)
    if full_3d:
        fig, ax, coll, quiv, title = _setup_scene_3d(L, box.radius, **scene_kwargs)
    else:
        fig, ax, coll, quiv, title = _setup_scene(L, box.radius, **scene_kwargs)

    quiv_holder = [quiv]
    live_run = LiveRun(box, mass=box.mass, radius=box.radius, L=L,
                       periodic=box.periodic, display_scale=ds)
    live_run.pos.append(box.pos.copy())
    live_run.vel.append(box.vel.copy())
    live_run.times.append(0.0)
    live_run.impulse.append(np.zeros(dim))
    elapsed = [0.0]
    virial_total = np.zeros(1)

    def render(time_s):
        if full_3d:
            quiv_holder[0] = _update_artists_3d(
                ax, coll, quiv_holder[0], title, box.pos, box.vel,
                color_by_render, time_s, vectors, mean_v, L)
        else:
            _update_artists(coll, quiv_holder[0], title, box.pos, box.vel,
                            color_by_render, time_s, None)

    render(0.0)

    from IPython.display import display
    push_frame, render_time = _make_live_push(fig)
    display(fig.canvas)

    try:
        import ipywidgets as widgets
    except ImportError as exc:
        raise RuntimeError(
            "animate='live' needs ipywidgets for its Play/Pause control "
            "(pip install ipywidgets)."
        ) from exc

    achievable_fps = 0.8 / max(render_time, 1e-3)
    effective_fps = min(fps, achievable_fps) if fps and fps > 0 else achievable_fps
    interval_ms = max(1, round(1000.0 / (effective_fps * max(speed, 1e-9))))
    play_widget = widgets.Play(min=0, max=n_frames - 1, step=1,
                               interval=interval_ms, value=0, repeat=False)
    live_run._play_widget = play_widget

    def on_change(change):
        f = change["new"]
        frame_impulse = np.zeros(dim)
        box.advance(dt=dt, steps=sample_every, impulse=frame_impulse,
                    virial=virial_total, thermostat=thermostat,
                    T_start=T_start, T_target=T_target, rate=rate,
                    t_elapsed=elapsed[0])
        elapsed[0] += dt * sample_every
        live_run.pos.append(box.pos.copy())
        live_run.vel.append(box.vel.copy())
        live_run.times.append(elapsed[0])
        live_run.impulse.append(frame_impulse)
        live_run.virial = float(virial_total[0])
        render(elapsed[0])
        push_frame()
        if f >= n_frames - 1:
            play_widget.playing = False
            live_run.finished = True

    play_widget.observe(on_change, names="value")
    display(play_widget)
    return live_run


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
