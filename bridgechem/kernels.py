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
  in kg*m/s -- turned into the "wall" pressure.
* the collisional virial ``sum(|r_ij| * impulse_ij)`` over all particle-particle
  collisions is accumulated as a length-1 array, in kg*m**2/s -- turned into
  the "virial" pressure (see :mod:`bridgechem.analysis`). Both give the same
  answer for a box with reflective walls; only virial works for periodic
  boundaries, since there are no walls to measure momentum transfer at.
"""

from __future__ import annotations

import numpy as np

from .constants import K_B

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
def _resolve_collisions(pos, vel, radius, inv_mass, Lx, Ly, periodic, virial):
    """Detect overlapping pairs and resolve them as elastic collisions.

    O(N^2). Momentum and kinetic energy are conserved exactly per collision
    (restitution coefficient e = 1). Overlapping pairs are also nudged apart to
    avoid particles sticking together at higher densities.

    Accumulates the collisional virial ``sum(|r_ij| * impulse)`` into
    ``virial[0]`` -- the impulse is exchanged along the line of centres
    (frictionless hard spheres), so ``r_ij . impulse_ij = dist * imp`` exactly.
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
                    virial[0] += dist * imp
                # positional de-overlap (does not change velocities/energy)
                overlap = rsum - dist
                if overlap > 0.0:
                    push = 0.5 * overlap
                    pos[i, 0] += push * nx
                    pos[i, 1] += push * ny
                    pos[j, 0] -= push * nx
                    pos[j, 1] -= push * ny


@njit
def _current_temperature(vel, inv_mass, dim):
    """Instantaneous temperature (K) from equipartition: <m v^2> = dim k_B T."""
    N = vel.shape[0]
    total = 0.0
    for i in range(N):
        m = 1.0 / inv_mass[i]
        total += m * (vel[i, 0] * vel[i, 0] + vel[i, 1] * vel[i, 1])
    return total / (N * dim * K_B)


@njit
def _rescale_velocities(vel, factor):
    N = vel.shape[0]
    for i in range(N):
        vel[i, 0] *= factor
        vel[i, 1] *= factor


@njit
def _ramp_target_temperature(T_start, T_target, rate, t_elapsed):
    """Linearly ramp from ``T_start`` toward ``T_target`` at ``rate`` (K/s),
    clamped so it never overshoots. ``rate <= 0`` means jump immediately."""
    if rate <= 0.0:
        return T_target
    if T_target >= T_start:
        return min(T_start + rate * t_elapsed, T_target)
    return max(T_start - rate * t_elapsed, T_target)


@njit
def _apply_thermostat(vel, inv_mass, dim, T_start, T_target, rate, t_elapsed):
    """Rescale velocities toward the ramped target temperature, in place."""
    T_now = _ramp_target_temperature(T_start, T_target, rate, t_elapsed)
    T_current = _current_temperature(vel, inv_mass, dim)
    if T_current > 0.0:
        _rescale_velocities(vel, np.sqrt(T_now / T_current))


@njit
def _step(pos, vel, radius, inv_mass, Lx, Ly, dt, periodic, impulse, virial):
    """Advance the system by a single time step, in place."""
    N = pos.shape[0]
    for i in range(N):
        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt
    _apply_boundaries(pos, vel, radius, inv_mass, Lx, Ly, periodic, impulse)
    _resolve_collisions(pos, vel, radius, inv_mass, Lx, Ly, periodic, virial)
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
              periodic, thermostat=False, T_start=0.0, T_target=0.0,
              rate=0.0, dim=2):
    """Run ``n_steps`` steps, recording the state every ``sample_every`` steps.

    Returns ``(traj_pos, traj_vel, times, impulse, virial)`` where the
    trajectory arrays have shape ``(n_frames, N, 2)``. ``impulse`` is the total
    momentum handed to the x- and y-walls over the whole run, and ``virial``
    the collisional virial sum -- see :mod:`bridgechem.analysis` for how each
    turns into a pressure. If ``thermostat`` is True, velocities are rescaled
    every step toward a target temperature ramped from ``T_start`` to
    ``T_target`` at ``rate`` (K/s; <= 0 means jump immediately).
    """
    N = pos.shape[0]
    n_frames = n_steps // sample_every + 1
    traj_pos = np.empty((n_frames, N, 2))
    traj_vel = np.empty((n_frames, N, 2))
    times = np.empty(n_frames)
    impulse = np.zeros(2)
    virial = np.zeros(1)

    traj_pos[0] = pos
    traj_vel[0] = vel
    times[0] = 0.0
    frame = 1

    for step in range(1, n_steps + 1):
        _step(pos, vel, radius, inv_mass, Lx, Ly, dt, periodic, impulse, virial)
        if thermostat:
            _apply_thermostat(vel, inv_mass, dim, T_start, T_target, rate,
                              step * dt)
        if step % sample_every == 0 and frame < n_frames:
            traj_pos[frame] = pos
            traj_vel[frame] = vel
            times[frame] = step * dt
            frame += 1

    return traj_pos[:frame], traj_vel[:frame], times[:frame], impulse, virial[0]


# --------------------------------------------------------------------------- #
# Lennard-Jones forces + velocity-Verlet integration
# --------------------------------------------------------------------------- #
@njit
def _lj_forces(pos, Lx, Ly, periodic, epsilon, sigma2, r_cut2, u_shift, forces):
    """Accumulate pairwise LJ forces into ``forces`` (N,2); return (PE, virial).

    O(N^2) with a cutoff (pairs beyond ``r_cut2`` are skipped). The potential
    is shifted so U(r_cut) = 0 (continuous energy at the cutoff, avoiding a
    small energy jump each time a pair crosses it). ``virial`` is the
    instantaneous ``sum(r_ij . F_ij)`` (J); the caller integrates
    ``virial * dt`` over the run for the virial pressure (see
    :func:`bridgechem.analysis.pressure_virial`).
    """
    N = pos.shape[0]
    forces[:, :] = 0.0
    pe = 0.0
    virial = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            if periodic:
                if dx > 0.5 * Lx:
                    dx -= Lx
                elif dx < -0.5 * Lx:
                    dx += Lx
                if dy > 0.5 * Ly:
                    dy -= Ly
                elif dy < -0.5 * Ly:
                    dy += Ly
            r2 = dx * dx + dy * dy
            if r2 < r_cut2 and r2 > 1e-30:
                inv_r2 = 1.0 / r2
                sr6 = sigma2 * sigma2 * sigma2 * inv_r2 * inv_r2 * inv_r2
                sr12 = sr6 * sr6
                f_over_r2 = 24.0 * epsilon * (2.0 * sr12 - sr6) * inv_r2
                fx = f_over_r2 * dx
                fy = f_over_r2 * dy
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy
                pe += 4.0 * epsilon * (sr12 - sr6) - u_shift
                virial += dx * fx + dy * fy
    return pe, virial


@njit
def _step_lj(pos, vel, forces, radius, inv_mass, Lx, Ly, dt, periodic,
            epsilon, sigma2, r_cut2, u_shift, impulse, virial):
    """One velocity-Verlet step under LJ forces (+ elastic wall reflection).

    ``forces`` holds the force at the start of the step (from the previous
    call, or the initial evaluation) and is updated in place to the force at
    the new positions, ready for the next call. Returns the potential energy
    at the new positions.
    """
    N = pos.shape[0]
    for i in range(N):
        ax = forces[i, 0] * inv_mass[i]
        ay = forces[i, 1] * inv_mass[i]
        pos[i, 0] += vel[i, 0] * dt + 0.5 * ax * dt * dt
        pos[i, 1] += vel[i, 1] * dt + 0.5 * ay * dt * dt
        vel[i, 0] += 0.5 * ax * dt
        vel[i, 1] += 0.5 * ay * dt

    _apply_boundaries(pos, vel, radius, inv_mass, Lx, Ly, periodic, impulse)

    pe, virial_instant = _lj_forces(pos, Lx, Ly, periodic, epsilon, sigma2,
                                    r_cut2, u_shift, forces)
    virial[0] += virial_instant * dt

    for i in range(N):
        ax = forces[i, 0] * inv_mass[i]
        ay = forces[i, 1] * inv_mass[i]
        vel[i, 0] += 0.5 * ax * dt
        vel[i, 1] += 0.5 * ay * dt
    return pe


@njit
def _simulate_lj(pos, vel, radius, inv_mass, Lx, Ly, dt, n_steps, sample_every,
                 periodic, epsilon, sigma, r_cut2, u_shift,
                 thermostat, T_start, T_target, rate, dim):
    """Run ``n_steps`` velocity-Verlet steps under LJ forces.

    Returns ``(traj_pos, traj_vel, traj_pe, times, impulse, virial)``. If
    ``thermostat`` is True, velocities are rescaled every step toward a
    target temperature ramped from ``T_start`` to ``T_target`` at ``rate``
    (K/s; <= 0 means jump immediately) -- see
    :func:`bridgechem.kernels._apply_thermostat`.
    """
    N = pos.shape[0]
    n_frames = n_steps // sample_every + 1
    traj_pos = np.empty((n_frames, N, 2))
    traj_vel = np.empty((n_frames, N, 2))
    traj_pe = np.empty(n_frames)
    times = np.empty(n_frames)
    impulse = np.zeros(2)
    virial = np.zeros(1)
    forces = np.zeros((N, 2))
    sigma2 = sigma * sigma

    pe0, _ = _lj_forces(pos, Lx, Ly, periodic, epsilon, sigma2, r_cut2,
                        u_shift, forces)

    traj_pos[0] = pos
    traj_vel[0] = vel
    traj_pe[0] = pe0
    times[0] = 0.0
    frame = 1

    for step in range(1, n_steps + 1):
        pe = _step_lj(pos, vel, forces, radius, inv_mass, Lx, Ly, dt, periodic,
                      epsilon, sigma2, r_cut2, u_shift, impulse, virial)
        if thermostat:
            _apply_thermostat(vel, inv_mass, dim, T_start, T_target, rate,
                              step * dt)
        if step % sample_every == 0 and frame < n_frames:
            traj_pos[frame] = pos
            traj_vel[frame] = vel
            traj_pe[frame] = pe
            times[frame] = step * dt
            frame += 1

    return (traj_pos[:frame], traj_vel[:frame], traj_pe[:frame],
            times[:frame], impulse, virial[0])
