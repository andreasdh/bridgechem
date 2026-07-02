"""Post-processing helpers: temperatures, speeds, pressure and the
Maxwell-Boltzmann distribution.

All functions take and return SI quantities.
"""

from __future__ import annotations

import numpy as np

from .constants import K_B


def speeds(velocities: np.ndarray) -> np.ndarray:
    """Speeds (m/s) from a ``(..., 2)`` velocity array."""
    return np.sqrt(np.sum(velocities ** 2, axis=-1))


def kinetic_energy(velocities: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """Total kinetic energy (J) per frame.

    ``velocities`` may be ``(N, 2)`` (one frame) or ``(n_frames, N, 2)``.
    ``mass`` is a ``(N,)`` array in kg.
    """
    v2 = np.sum(velocities ** 2, axis=-1)  # (..., N)
    return 0.5 * np.sum(v2 * mass, axis=-1)


def temperature(velocities: np.ndarray, mass: np.ndarray, dim: int = 2) -> np.ndarray:
    """Instantaneous temperature (K) from the equipartition theorem.

    <KE> per particle = (dim/2) k_B T, so T = <m v^2> / (dim k_B).
    Returns a scalar for a single frame or a ``(n_frames,)`` array.
    """
    v2 = np.sum(velocities ** 2, axis=-1)  # (..., N)
    mean_m_v2 = np.mean(v2 * mass, axis=-1)
    return mean_m_v2 / (dim * K_B)


def pressure_wall(impulse: np.ndarray, total_time: float, Lx: float,
                  Ly: float) -> float:
    """"wall" method: 2D pressure (N/m) from momentum transferred to the walls.

    This is the most direct, operational definition of pressure -- literally
    what a pressure gauge mounted on the container wall would read: force per
    unit length = (momentum delivered) / (time x wall length).

    ``impulse`` = total momentum transferred to [x-walls, y-walls] over the
    run. Each pair of opposite walls has combined length 2*L_perp, so the
    pressure on the x-walls is impulse_x / (total_time * 2 * Ly), and
    similarly for y; we average the two for an isotropic estimate.

    Only meaningful for **reflective** boundaries -- with periodic boundaries
    particles never touch a wall, so ``impulse`` stays zero and this method
    cannot be used (see :func:`pressure_virial` instead).
    """
    if total_time <= 0.0:
        return 0.0
    p_x = impulse[0] / (total_time * 2.0 * Ly)
    p_y = impulse[1] / (total_time * 2.0 * Lx)
    return 0.5 * (p_x + p_y)


def pressure_virial(n_particles: int, temperature_K: float, area: float,
                    virial: float, total_time: float, dim: int = 2) -> float:
    """"virial" method: 2D pressure (N/m) from the Clausius virial theorem.

        P = [N k_B T + (1/dim) <sum_{collisions} r_ij . impulse_ij> / t] / A

    The kinetic (ideal-gas) term N k_B T / A is corrected by the time-averaged
    virial of the collisional forces: every particle-particle collision
    delivers an impulse along the line connecting the two centres, so its
    contribution to the sum is exactly ``|r_ij| * impulse``, which is what
    :func:`bridgechem.kernels._resolve_collisions` accumulates into
    ``virial``. Collisions are always repulsive here, so this term is always
    >= 0 -- finite-size particles exclude area from each other and therefore
    raise the pressure above the ideal-gas value (like a hard-disk/van der
    Waals gas).

    Unlike :func:`pressure_wall`, this needs no walls at all: it works
    identically for reflective *and* periodic boundaries, which is why it's
    the standard technique for periodic (wall-less) molecular dynamics. For a
    reflective box the two methods measure the same physical pressure by two
    independent routes and should agree -- a good sanity check on a
    simulation.
    """
    if total_time <= 0.0:
        return 0.0
    ideal_term = n_particles * K_B * temperature_K
    virial_term = virial / (dim * total_time)
    return (ideal_term + virial_term) / area


def ideal_gas_pressure(n_particles: int, temperature_K: float,
                       area: float) -> float:
    """"ideal" method: the textbook ideal-gas estimate P = N k_B T / A (N/m).

    A theoretical reference value, not something measured from the dynamics
    (it ignores particle size and all collisions) -- useful as a baseline to
    compare :func:`pressure_wall` / :func:`pressure_virial` against.
    """
    return n_particles * K_B * temperature_K / area


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
