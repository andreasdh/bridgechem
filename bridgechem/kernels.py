"""Numba-accelerated hard-sphere kernels.

The hot loops (collision detection, boundary handling, time stepping) live here
as plain functions operating on raw NumPy arrays. They are JIT-compiled with
numba when it is available. If numba is not installed the ``njit`` decorator
degrades to a no-op so the package still runs (much more slowly) in pure Python
-- handy for small examples and for environments where numba is unavailable.

Conventions used throughout:

* ``pos``, ``vel`` are ``(N, 2)`` float64 arrays (metres, metres/second).
* ``inv_mass`` is ``(N,)`` = 1 / mass (1/kg).
* ``radius`` is ``(N,)`` in metres.
* ``Lx``, ``Ly`` are the box side lengths in metres.
* ``periodic`` is a bool: wrap-around (True) or reflective walls (False).
* wall impulse is accumulated as ``[|p| transferred at x-walls, at y-walls]``
  in kg*m/s and later turned into a pressure.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised implicitly depending on environment
    from numba import njit as _njit

    HAVE_NUMBA = True

    def njit(*args, **kwargs):
        # Sensible defaults; callers may still pass their own.
        kwargs.setdefault("cache", True)
        kwargs.setdefault("fastmath", True)
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return _njit(args[0])
        return _njit(*args, **kwargs)

except Exception:  # numba missing -> transparent fallback
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def wrap(func):
            return func

        return wrap


@njit
def _apply_boundaries(pos, vel, radius, inv_mass, Lx, Ly, periodic, impulse):
    """Keep particles inside the box; accumulate wall impulse for pressure."""
    N = pos.shape[0]
    if periodic:
        for i in range(N):
            x = pos[i, 0]
            if x < 0.0:
                pos[i, 0] = x + Lx * (int(-x / Lx) + 1)
            elif x >= Lx:
                pos[i, 0] = x - Lx * int(x / Lx)
            y = pos[i, 1]
            if y < 0.0:
                pos[i, 1] = y + Ly * (int(-y / Ly) + 1)
            elif y >= Ly:
                pos[i, 1] = y - Ly * int(y / Ly)
    else:
        for i in range(N):
            r = radius[i]
            m = 1.0 / inv_mass[i]
            if pos[i, 0] < r:
                pos[i, 0] = r
                impulse[0] += 2.0 * m * abs(vel[i, 0])
                vel[i, 0] = -vel[i, 0]
            elif pos[i, 0] > Lx - r:
                pos[i, 0] = Lx - r
                impulse[0] += 2.0 * m * abs(vel[i, 0])
                vel[i, 0] = -vel[i, 0]
            if pos[i, 1] < r:
                pos[i, 1] = r
                impulse[1] += 2.0 * m * abs(vel[i, 1])
                vel[i, 1] = -vel[i, 1]
            elif pos[i, 1] > Ly - r:
                pos[i, 1] = Ly - r
                impulse[1] += 2.0 * m * abs(vel[i, 1])
                vel[i, 1] = -vel[i, 1]


@njit
def _resolve_collisions(pos, vel, radius, inv_mass, Lx, Ly, periodic):
    """Detect overlapping pairs and resolve them as elastic collisions.

    O(N^2). Momentum and kinetic energy are conserved exactly per collision
    (restitution coefficient e = 1). Overlapping pairs are also nudged apart to
    avoid particles sticking together at higher densities.
    """
    N = pos.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            if periodic:
                # minimum-image convention
                if dx > 0.5 * Lx:
                    dx -= Lx
                elif dx < -0.5 * Lx:
                    dx += Lx
                if dy > 0.5 * Ly:
                    dy -= Ly
                elif dy < -0.5 * Ly:
                    dy += Ly
            dist2 = dx * dx + dy * dy
            rsum = radius[i] + radius[j]
            if dist2 < rsum * rsum and dist2 > 1e-30:
                dist = np.sqrt(dist2)
                nx = dx / dist
                ny = dy / dist
                rvx = vel[i, 0] - vel[j, 0]
                rvy = vel[i, 1] - vel[j, 1]
                vn = rvx * nx + rvy * ny
                if vn < 0.0:  # only resolve if approaching
                    imp = -2.0 * vn / (inv_mass[i] + inv_mass[j])
                    vel[i, 0] += imp * inv_mass[i] * nx
                    vel[i, 1] += imp * inv_mass[i] * ny
                    vel[j, 0] -= imp * inv_mass[j] * nx
                    vel[j, 1] -= imp * inv_mass[j] * ny
                # positional de-overlap (does not change velocities/energy)
                overlap = rsum - dist
                if overlap > 0.0:
                    push = 0.5 * overlap
                    pos[i, 0] += push * nx
                    pos[i, 1] += push * ny
                    pos[j, 0] -= push * nx
                    pos[j, 1] -= push * ny


@njit
def _step(pos, vel, radius, inv_mass, Lx, Ly, dt, periodic, impulse):
    """Advance the system by a single time step, in place."""
    N = pos.shape[0]
    for i in range(N):
        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt
    _apply_boundaries(pos, vel, radius, inv_mass, Lx, Ly, periodic, impulse)
    _resolve_collisions(pos, vel, radius, inv_mass, Lx, Ly, periodic)
    if not periodic:
        # de-overlap can nudge a particle past a wall; clamp back inside
        # (position only, so velocities/energy/pressure are unaffected).
        for i in range(N):
            r = radius[i]
            if pos[i, 0] < r:
                pos[i, 0] = r
            elif pos[i, 0] > Lx - r:
                pos[i, 0] = Lx - r
            if pos[i, 1] < r:
                pos[i, 1] = r
            elif pos[i, 1] > Ly - r:
                pos[i, 1] = Ly - r
    else:
        # de-overlap can nudge a particle just outside [0, L); re-wrap it
        # (pushes are tiny, so a single shift suffices).
        for i in range(N):
            if pos[i, 0] < 0.0:
                pos[i, 0] += Lx
            elif pos[i, 0] >= Lx:
                pos[i, 0] -= Lx
            if pos[i, 1] < 0.0:
                pos[i, 1] += Ly
            elif pos[i, 1] >= Ly:
                pos[i, 1] -= Ly


@njit
def _simulate(pos, vel, radius, inv_mass, Lx, Ly, dt, n_steps, sample_every,
              periodic):
    """Run ``n_steps`` steps, recording the state every ``sample_every`` steps.

    Returns (traj_pos, traj_vel, times, impulse) where the trajectory arrays
    have shape ``(n_frames, N, 2)``. ``impulse`` is the total momentum handed
    to the x- and y-walls over the whole run (used to compute pressure).
    """
    N = pos.shape[0]
    n_frames = n_steps // sample_every + 1
    traj_pos = np.empty((n_frames, N, 2))
    traj_vel = np.empty((n_frames, N, 2))
    times = np.empty(n_frames)
    impulse = np.zeros(2)

    traj_pos[0] = pos
    traj_vel[0] = vel
    times[0] = 0.0
    frame = 1

    for step in range(1, n_steps + 1):
        _step(pos, vel, radius, inv_mass, Lx, Ly, dt, periodic, impulse)
        if step % sample_every == 0 and frame < n_frames:
            traj_pos[frame] = pos
            traj_vel[frame] = vel
            times[frame] = step * dt
            frame += 1

    return traj_pos[:frame], traj_vel[:frame], times[:frame], impulse
