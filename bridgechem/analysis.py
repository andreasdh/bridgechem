"""Post-processing helpers: temperatures, speeds, pressure and the
Maxwell-Boltzmann distribution.

All functions take and return SI quantities, and all of them work in 2D and
3D alike -- the dimension is either read off the array or passed as ``dim``.

A note on pressure units. In 3D, pressure is force per area and comes out in
pascal. In 2D there is no area to push on, only the *lines* that bound the
box, so "pressure" is force per length and comes out in N/m. The physics is
the same; only the geometry of the container differs.
"""

from __future__ import annotations

import numpy as np

from .constants import K_B


def speeds(velocities: np.ndarray) -> np.ndarray:
    """Speeds (m/s) from a ``(..., dim)`` velocity array."""
    return np.sqrt(np.sum(velocities ** 2, axis=-1))


def kinetic_energy(velocities: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """Total kinetic energy (J) per frame.

    ``velocities`` may be ``(N, dim)`` (one frame) or ``(n_frames, N, dim)``.
    ``mass`` is a ``(N,)`` array in kg.
    """
    v2 = np.sum(velocities ** 2, axis=-1)  # (..., N)
    return 0.5 * np.sum(v2 * mass, axis=-1)


def temperature(velocities: np.ndarray, mass: np.ndarray, dim: int = None) -> np.ndarray:
    """Instantaneous temperature (K) from the equipartition theorem.

    Each particle has ``dim`` translational degrees of freedom, each carrying
    an average of (1/2) k_B T, so

        <KE> per particle = (dim/2) k_B T   ->   T = <m v^2> / (dim k_B).

    In 2D that means <KE> = k_B T, in 3D the familiar (3/2) k_B T. ``dim``
    defaults to the number of velocity components, which is what you want
    unless you are deliberately asking a different question.

    Returns a scalar for a single frame or a ``(n_frames,)`` array.
    """
    velocities = np.asarray(velocities)
    if dim is None:
        dim = velocities.shape[-1]
    v2 = np.sum(velocities ** 2, axis=-1)  # (..., N)
    mean_m_v2 = np.mean(v2 * mass, axis=-1)
    return mean_m_v2 / (dim * K_B)


def pressure_wall(impulse: np.ndarray, total_time: float,
                  L: np.ndarray) -> float:
    """"wall" method: pressure from momentum transferred to the walls.

    This is the most direct, operational definition of pressure -- literally
    what a pressure gauge mounted on the container wall would read: force per
    unit wall area (or, in 2D, per unit wall length).

    ``impulse`` is the total momentum handed to the walls perpendicular to
    each axis, either ``(dim,)`` for a whole run or ``(n_frames, dim)`` per
    frame (in which case it is summed first). ``L`` is ``(dim,)``.

    For axis ``d`` there are two opposite walls, each of extent
    ``V / L[d]``, so the pressure read off those walls is

        P_d = impulse_d / (total_time * 2 * V / L[d]),

    and we average over the axes for an isotropic estimate.

    Only meaningful for **reflective** boundaries -- with periodic boundaries
    particles never touch a wall, so ``impulse`` stays zero and this method
    cannot be used (see :func:`pressure_virial` instead).
    """
    if total_time <= 0.0:
        return 0.0
    impulse = np.asarray(impulse, dtype=float)
    if impulse.ndim > 1:
        impulse = impulse.sum(axis=0)
    L = np.asarray(L, dtype=float)
    volume = float(np.prod(L))
    wall_extent = volume / L          # (dim,) area (3D) or length (2D) per wall
    per_axis = impulse / (total_time * 2.0 * wall_extent)
    return float(np.mean(per_axis))


def pressure_virial(n_particles: int, temperature_K: float, volume: float,
                    virial: float, total_time: float, dim: int = 2) -> float:
    """"virial" method: pressure from the Clausius virial theorem.

        P = [N k_B T + (1/dim) <sum_{collisions} r_ij . impulse_ij> / t] / V

    The kinetic (ideal-gas) term N k_B T / V is corrected by the time-averaged
    virial of the collisional forces: every particle-particle collision
    delivers an impulse along the line connecting the two centres, so its
    contribution to the sum is exactly ``|r_ij| * impulse``, which is what
    :func:`bridgechem.kernels._resolve_collisions` accumulates into
    ``virial``. Collisions are always repulsive here, so this term is always
    >= 0 -- finite-size particles exclude volume from each other and therefore
    raise the pressure above the ideal-gas value (like a hard-sphere/van der
    Waals gas).

    Unlike :func:`pressure_wall`, this needs no walls at all: it works
    identically for reflective *and* periodic boundaries, which is why it's
    the standard technique for periodic (wall-less) molecular dynamics. For a
    reflective box the two methods measure the same physical pressure by two
    independent routes and should broadly agree -- a good sanity check on a
    simulation. They are not identical, though: the centres of finite-size
    particles cannot come closer to a wall than one radius, so the volume
    actually accessible to them is slightly smaller than ``V``, and the wall
    reading sits a little high. The gap shrinks as the particles do.

    ``volume`` is the box volume in 3D and the box area in 2D.
    """
    if total_time <= 0.0:
        return 0.0
    ideal_term = n_particles * K_B * temperature_K
    virial_term = virial / (dim * total_time)
    return (ideal_term + virial_term) / volume


def ideal_gas_pressure(n_particles: int, temperature_K: float,
                       volume: float) -> float:
    """"ideal" method: the textbook ideal-gas estimate P = N k_B T / V.

    A theoretical reference value, not something measured from the dynamics
    (it ignores particle size and all collisions) -- useful as a baseline to
    compare :func:`pressure_wall` / :func:`pressure_virial` against.
    ``volume`` is the box volume in 3D and the box area in 2D.
    """
    return n_particles * K_B * temperature_K / volume


def maxwell_boltzmann_speed(v, temperature_K: float, mass_kg: float,
                            dim: int = 2):
    """Maxwell-Boltzmann *speed* probability density at speed(s) ``v``.

    In 2D this is the Rayleigh distribution
        f(v) = (m / kT) v exp(-m v^2 / 2kT),
    in 3D the familiar
        f(v) = 4 pi (m/2 pi kT)^{3/2} v^2 exp(-m v^2 / 2kT).
    """
    v = np.asarray(v, dtype=float)
    a = mass_kg / (K_B * temperature_K)
    if dim == 2:
        return a * v * np.exp(-0.5 * a * v ** 2)
    elif dim == 3:
        return (4.0 * np.pi * (a / (2.0 * np.pi)) ** 1.5
                * v ** 2 * np.exp(-0.5 * a * v ** 2))
    raise ValueError("dim must be 2 or 3")


def mean_speed(temperature_K: float, mass_kg: float, dim: int = 2) -> float:
    """Analytic mean speed of the Maxwell-Boltzmann distribution (m/s)."""
    if dim == 2:
        return np.sqrt(np.pi * K_B * temperature_K / (2.0 * mass_kg))
    elif dim == 3:
        return np.sqrt(8.0 * K_B * temperature_K / (np.pi * mass_kg))
    raise ValueError("dim must be 2 or 3")


def rms_speed(temperature_K: float, mass_kg: float, dim: int = 2) -> float:
    """Analytic root-mean-square speed: sqrt(dim k_B T / m) (m/s)."""
    return np.sqrt(dim * K_B * temperature_K / mass_kg)
