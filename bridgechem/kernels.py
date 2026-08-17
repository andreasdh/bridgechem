"""Numba-accelerated hard-sphere and Lennard-Jones kernels.

The hot loops (collision detection, boundary handling, time stepping) live here
as plain functions operating on raw NumPy arrays. numba is a required
dependency, so every one of them is JIT-compiled -- ``_simulate``/
``_simulate_lj`` themselves are ``@njit``, so an entire trajectory runs inside
one compiled function with no per-step Python overhead. If numba somehow isn't
importable at runtime (e.g. an unsupported platform), the ``njit`` decorator
degrades to a no-op so the package still runs, just much more slowly, in pure
Python -- a safety net, not the intended path.

Everything is written for a general number of dimensions: the physics is
expressed once, as a loop ``for d in range(dim)`` over the components of each
vector, and ``dim`` is simply ``pos.shape[1]``. A 2D box and a 3D box run the
exact same code.

Conventions used throughout:

* ``pos``, ``vel`` are ``(N, dim)`` float64 arrays (metres, metres/second).
* ``inv_mass`` is ``(N,)`` = 1 / mass (1/kg).
* ``radius`` is ``(N,)`` in metres.
* ``L`` is ``(dim,)``: the box side length along each axis, in metres.
* ``periodic`` is a bool: wrap-around (True) or reflective walls (False).
* wall impulse is accumulated per axis as ``(dim,)`` in kg*m/s -- the momentum
  handed to the pair of walls perpendicular to that axis. Turned into the
  "wall" pressure by :func:`bridgechem.analysis.pressure_wall`.
* the collisional virial ``sum(r_ij . impulse_ij)`` over all particle-particle
  collisions is accumulated as a length-1 array, in kg*m**2/s -- turned into
  the "virial" pressure. Both give the same answer for a box with reflective
  walls; only virial works for periodic boundaries, since there are no walls
  to measure momentum transfer at.
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
def _apply_boundaries(pos, vel, radius, inv_mass, L, periodic, impulse):
    """Keep particles inside the box; accumulate wall impulse for pressure."""
    N = pos.shape[0]
    dim = pos.shape[1]
    if periodic:
        for i in range(N):
            for d in range(dim):
                x = pos[i, d]
                if x < 0.0:
                    pos[i, d] = x + L[d] * (int(-x / L[d]) + 1)
                elif x >= L[d]:
                    pos[i, d] = x - L[d] * int(x / L[d])
    else:
        for i in range(N):
            r = radius[i]
            m = 1.0 / inv_mass[i]
            for d in range(dim):
                if pos[i, d] < r:
                    pos[i, d] = r
                    impulse[d] += 2.0 * m * abs(vel[i, d])
                    vel[i, d] = -vel[i, d]
                elif pos[i, d] > L[d] - r:
                    pos[i, d] = L[d] - r
                    impulse[d] += 2.0 * m * abs(vel[i, d])
                    vel[i, d] = -vel[i, d]


@njit
def _resolve_collisions(pos, vel, radius, inv_mass, L, periodic, virial):
    """Detect overlapping pairs and resolve them as elastic collisions.

    O(N^2). Momentum and kinetic energy are conserved exactly per collision
    (restitution coefficient e = 1). Overlapping pairs are also nudged apart to
    avoid particles sticking together at higher densities.

    Accumulates the collisional virial ``sum(|r_ij| * impulse)`` into
    ``virial[0]`` -- the impulse is exchanged along the line of centres
    (frictionless hard spheres), so ``r_ij . impulse_ij = dist * imp`` exactly.
    """
    N = pos.shape[0]
    dim = pos.shape[1]
    sep = np.empty(dim)  # vector from j to i, reused for every pair
    for i in range(N):
        for j in range(i + 1, N):
            dist2 = 0.0
            for d in range(dim):
                delta = pos[i, d] - pos[j, d]
                if periodic:  # minimum-image convention
                    if delta > 0.5 * L[d]:
                        delta -= L[d]
                    elif delta < -0.5 * L[d]:
                        delta += L[d]
                sep[d] = delta
                dist2 += delta * delta
            rsum = radius[i] + radius[j]
            if dist2 >= rsum * rsum or dist2 <= 1e-30:
                continue

            # they are touching: resolve the collision along the line of
            # centres, n = r_ij / |r_ij|
            dist = np.sqrt(dist2)
            vn = 0.0  # relative velocity along that line
            for d in range(dim):
                vn += (vel[i, d] - vel[j, d]) * sep[d] / dist
            if vn < 0.0:  # only resolve if approaching
                imp = -2.0 * vn / (inv_mass[i] + inv_mass[j])
                for d in range(dim):
                    n_d = sep[d] / dist
                    vel[i, d] += imp * inv_mass[i] * n_d
                    vel[j, d] -= imp * inv_mass[j] * n_d
                virial[0] += dist * imp
            # positional de-overlap (does not change velocities/energy)
            overlap = rsum - dist
            if overlap > 0.0:
                push = 0.5 * overlap
                for d in range(dim):
                    n_d = sep[d] / dist
                    pos[i, d] += push * n_d
                    pos[j, d] -= push * n_d


@njit
def _current_temperature(vel, inv_mass):
    """Instantaneous temperature (K) from equipartition: <m v^2> = dim k_B T.

    ``dim`` is read off the velocity array rather than passed in, so it can
    never disagree with the number of components actually summed.
    """
    N = vel.shape[0]
    dim = vel.shape[1]
    total = 0.0
    for i in range(N):
        m = 1.0 / inv_mass[i]
        v2 = 0.0
        for d in range(dim):
            v2 += vel[i, d] * vel[i, d]
        total += m * v2
    return total / (N * dim * K_B)


@njit
def _rescale_velocities(vel, factor):
    N = vel.shape[0]
    dim = vel.shape[1]
    for i in range(N):
        for d in range(dim):
            vel[i, d] *= factor


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
def _apply_thermostat(vel, inv_mass, T_start, T_target, rate, t_elapsed):
    """Rescale velocities toward the ramped target temperature, in place."""
    T_now = _ramp_target_temperature(T_start, T_target, rate, t_elapsed)
    T_current = _current_temperature(vel, inv_mass)
    if T_current > 0.0:
        _rescale_velocities(vel, np.sqrt(T_now / T_current))


@njit
def _clamp_inside(pos, radius, L, periodic):
    """Fix up positions after de-overlap may have nudged a particle outside.

    Position only, so velocities/energy/pressure are unaffected. The pushes are
    tiny, so a single shift suffices for the periodic case.
    """
    N = pos.shape[0]
    dim = pos.shape[1]
    if periodic:
        for i in range(N):
            for d in range(dim):
                if pos[i, d] < 0.0:
                    pos[i, d] += L[d]
                elif pos[i, d] >= L[d]:
                    pos[i, d] -= L[d]
    else:
        for i in range(N):
            r = radius[i]
            for d in range(dim):
                if pos[i, d] < r:
                    pos[i, d] = r
                elif pos[i, d] > L[d] - r:
                    pos[i, d] = L[d] - r


@njit
def _step(pos, vel, radius, inv_mass, L, dt, periodic, impulse, virial):
    """Advance the system by a single time step, in place."""
    N = pos.shape[0]
    dim = pos.shape[1]
    for i in range(N):
        for d in range(dim):
            pos[i, d] += vel[i, d] * dt
    _apply_boundaries(pos, vel, radius, inv_mass, L, periodic, impulse)
    _resolve_collisions(pos, vel, radius, inv_mass, L, periodic, virial)
    _clamp_inside(pos, radius, L, periodic)


@njit
def _simulate(pos, vel, radius, inv_mass, L, dt, n_steps, sample_every,
              periodic, thermostat=False, T_start=0.0, T_target=0.0,
              rate=0.0):
    """Run ``n_steps`` steps, recording the state every ``sample_every`` steps.

    Returns ``(traj_pos, traj_vel, times, impulse, virial)`` where the
    trajectory arrays have shape ``(n_frames, N, dim)``. ``impulse`` has shape
    ``(n_frames, dim)`` and holds the momentum handed to the walls *during each
    recorded frame*, so pressure can be followed over time (and the
    equilibration period discarded) rather than only averaged over the whole
    run. ``virial`` is the collisional virial sum over the run -- see
    :mod:`bridgechem.analysis` for how each turns into a pressure. If
    ``thermostat`` is True, velocities are rescaled every step toward a target
    temperature ramped from ``T_start`` to ``T_target`` at ``rate`` (K/s;
    <= 0 means jump immediately).
    """
    N = pos.shape[0]
    dim = pos.shape[1]
    n_frames = n_steps // sample_every + 1
    traj_pos = np.empty((n_frames, N, dim))
    traj_vel = np.empty((n_frames, N, dim))
    times = np.empty(n_frames)
    impulse = np.zeros((n_frames, dim))
    frame_impulse = np.zeros(dim)
    virial = np.zeros(1)

    traj_pos[0] = pos
    traj_vel[0] = vel
    times[0] = 0.0
    frame = 1

    for step in range(1, n_steps + 1):
        _step(pos, vel, radius, inv_mass, L, dt, periodic, frame_impulse,
              virial)
        if thermostat:
            _apply_thermostat(vel, inv_mass, T_start, T_target, rate,
                              step * dt)
        if step % sample_every == 0 and frame < n_frames:
            traj_pos[frame] = pos
            traj_vel[frame] = vel
            times[frame] = step * dt
            for d in range(dim):
                impulse[frame, d] = frame_impulse[d]
                frame_impulse[d] = 0.0
            frame += 1

    return traj_pos[:frame], traj_vel[:frame], times[:frame], impulse[:frame], virial[0]


# --------------------------------------------------------------------------- #
# Lennard-Jones forces + velocity-Verlet integration
# --------------------------------------------------------------------------- #
@njit
def _lj_forces(pos, L, periodic, epsilon, sigma2, r_cut2, u_shift, forces):
    """Accumulate pairwise LJ forces into ``forces`` (N,dim); return (PE, virial).

    O(N^2) with a cutoff (pairs beyond ``r_cut2`` are skipped). The potential
    is shifted so U(r_cut) = 0 (continuous energy at the cutoff, avoiding a
    small energy jump each time a pair crosses it). ``virial`` is the
    instantaneous ``sum(r_ij . F_ij)`` (J); the caller integrates
    ``virial * dt`` over the run for the virial pressure (see
    :func:`bridgechem.analysis.pressure_virial`).
    """
    N = pos.shape[0]
    dim = pos.shape[1]
    sep = np.empty(dim)  # vector from j to i, reused for every pair
    forces[:, :] = 0.0
    pe = 0.0
    virial = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            r2 = 0.0
            for d in range(dim):
                delta = pos[i, d] - pos[j, d]
                if periodic:  # minimum-image convention
                    if delta > 0.5 * L[d]:
                        delta -= L[d]
                    elif delta < -0.5 * L[d]:
                        delta += L[d]
                sep[d] = delta
                r2 += delta * delta
            if r2 >= r_cut2 or r2 <= 1e-30:
                continue

            # inside the cutoff: F = -dU/dr along r_ij, split per component
            inv_r2 = 1.0 / r2
            sr6 = sigma2 * sigma2 * sigma2 * inv_r2 * inv_r2 * inv_r2
            sr12 = sr6 * sr6
            f_over_r2 = 24.0 * epsilon * (2.0 * sr12 - sr6) * inv_r2
            for d in range(dim):
                f_d = f_over_r2 * sep[d]
                forces[i, d] += f_d
                forces[j, d] -= f_d
                virial += sep[d] * f_d
            pe += 4.0 * epsilon * (sr12 - sr6) - u_shift
    return pe, virial


@njit
def _step_lj(pos, vel, forces, radius, inv_mass, L, dt, periodic,
             epsilon, sigma2, r_cut2, u_shift, impulse, virial):
    """One velocity-Verlet step under LJ forces (+ elastic wall reflection).

    ``forces`` holds the force at the start of the step (from the previous
    call, or the initial evaluation) and is updated in place to the force at
    the new positions, ready for the next call. Returns the potential energy
    at the new positions.
    """
    N = pos.shape[0]
    dim = pos.shape[1]
    for i in range(N):
        for d in range(dim):
            a_d = forces[i, d] * inv_mass[i]
            pos[i, d] += vel[i, d] * dt + 0.5 * a_d * dt * dt
            vel[i, d] += 0.5 * a_d * dt

    _apply_boundaries(pos, vel, radius, inv_mass, L, periodic, impulse)

    pe, virial_instant = _lj_forces(pos, L, periodic, epsilon, sigma2,
                                    r_cut2, u_shift, forces)
    virial[0] += virial_instant * dt

    for i in range(N):
        for d in range(dim):
            vel[i, d] += 0.5 * forces[i, d] * inv_mass[i] * dt
    return pe


@njit
def _simulate_lj(pos, vel, radius, inv_mass, L, dt, n_steps, sample_every,
                 periodic, epsilon, sigma, r_cut2, u_shift,
                 thermostat, T_start, T_target, rate):
    """Run ``n_steps`` velocity-Verlet steps under LJ forces.

    Returns ``(traj_pos, traj_vel, traj_pe, times, impulse, virial)``, with
    ``impulse`` recorded per frame exactly as in :func:`_simulate`. If
    ``thermostat`` is True, velocities are rescaled every step toward a
    target temperature ramped from ``T_start`` to ``T_target`` at ``rate``
    (K/s; <= 0 means jump immediately) -- see
    :func:`bridgechem.kernels._apply_thermostat`.
    """
    N = pos.shape[0]
    dim = pos.shape[1]
    n_frames = n_steps // sample_every + 1
    traj_pos = np.empty((n_frames, N, dim))
    traj_vel = np.empty((n_frames, N, dim))
    traj_pe = np.empty(n_frames)
    times = np.empty(n_frames)
    impulse = np.zeros((n_frames, dim))
    frame_impulse = np.zeros(dim)
    virial = np.zeros(1)
    forces = np.zeros((N, dim))
    sigma2 = sigma * sigma

    pe0, _ = _lj_forces(pos, L, periodic, epsilon, sigma2, r_cut2,
                        u_shift, forces)

    traj_pos[0] = pos
    traj_vel[0] = vel
    traj_pe[0] = pe0
    times[0] = 0.0
    frame = 1

    for step in range(1, n_steps + 1):
        pe = _step_lj(pos, vel, forces, radius, inv_mass, L, dt, periodic,
                      epsilon, sigma2, r_cut2, u_shift, frame_impulse, virial)
        if thermostat:
            _apply_thermostat(vel, inv_mass, T_start, T_target, rate,
                              step * dt)
        if step % sample_every == 0 and frame < n_frames:
            traj_pos[frame] = pos
            traj_vel[frame] = vel
            traj_pe[frame] = pe
            times[frame] = step * dt
            for d in range(dim):
                impulse[frame, d] = frame_impulse[d]
                frame_impulse[d] = 0.0
            frame += 1

    return (traj_pos[:frame], traj_vel[:frame], traj_pe[:frame],
            times[:frame], impulse[:frame], virial[0])
