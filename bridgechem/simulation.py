"""The :class:`Simulation` object returned by :meth:`Box.run`.

It holds the recorded trajectory and exposes analysis and visualisation:

    sim = system.run(steps=20000)
    sim.show()                       # animation
    sim.calculate("pressure")        # 2D pressure (N/m)
    sim.histogram("speeds")          # speed histogram vs Maxwell-Boltzmann
"""

from __future__ import annotations

import numpy as np

from . import analysis, viz

PRESSURE_METHODS = ("wall", "virial", "ideal")


class Simulation:
    def __init__(self, traj_pos, traj_vel, times, impulse, virial=0.0, *,
                 mass, radius, Lx, Ly, dim=2, periodic=False,
                 potential_energy=None, display_scale=1.0):
        self.pos = np.asarray(traj_pos)          # (n_frames, N, 2), m
        self.vel = np.asarray(traj_vel)          # (n_frames, N, 2), m/s
        self.times = np.asarray(times)           # (n_frames,), s
        self.impulse = np.asarray(impulse)       # (2,), kg*m/s
        self.virial = float(virial)              # kg*m**2/s, see analysis.pressure_virial
        self.mass = np.asarray(mass)             # (N,), kg
        self.radius = np.asarray(radius)         # (N,), m
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.dim = dim
        self.periodic = periodic
        self.display_scale = float(display_scale)
        # (n_frames,) J -- zero for a pure hard-sphere run (no interactions)
        self.potential_energy = (np.asarray(potential_energy)
                                 if potential_energy is not None
                                 else np.zeros(self.pos.shape[0]))

    # -- basic properties ---------------------------------------------------
    @property
    def n_frames(self) -> int:
        return self.pos.shape[0]

    @property
    def n_particles(self) -> int:
        return self.pos.shape[1]

    @property
    def area(self) -> float:
        return self.Lx * self.Ly

    @property
    def total_time(self) -> float:
        return float(self.times[-1]) if self.times.size else 0.0

    # -- analysis -----------------------------------------------------------
    def calculate(self, quantity: str, method: str = None):
        """Compute a derived quantity from the trajectory.

        Supported: ``velocities``, ``speeds``, ``temperature``,
        ``kinetic_energy``, ``potential_energy``, ``total_energy``,
        ``pressure``, ``mean_speed``.

        For ``"pressure"``, ``method`` picks *how* it's computed -- see
        :meth:`pressure` for the physics behind each option.
        """
        q = quantity.lower()
        if q in ("velocities", "velocity"):
            return self.vel
        if q in ("speeds", "speed"):
            return analysis.speeds(self.vel)
        if q in ("temperature", "temp"):
            return analysis.temperature(self.vel, self.mass, dim=self.dim)
        if q in ("kinetic_energy", "ke"):
            return analysis.kinetic_energy(self.vel, self.mass)
        if q in ("potential_energy", "pe"):
            return self.potential_energy
        if q in ("total_energy", "energy"):
            return analysis.kinetic_energy(self.vel, self.mass) + self.potential_energy
        if q == "pressure":
            return self.pressure(method=method)
        if q == "mean_speed":
            return float(np.mean(analysis.speeds(self.vel)))
        raise ValueError(
            f"Unknown quantity {quantity!r}. Try: velocities, speeds, "
            "temperature, kinetic_energy, potential_energy, total_energy, "
            "pressure, mean_speed."
        )

    def pressure(self, method: str = None) -> float:
        """2D pressure (N/m), computed one of three ways.

        method="wall"
            Momentum transferred to the container walls per unit time and
            length -- literally what a pressure gauge on the wall would read.
            Needs **reflective** boundaries (raises otherwise: with periodic
            boundaries nothing ever touches a wall).
        method="virial"
            The Clausius virial theorem applied to particle-particle
            collisions: P = [N k_B T + (virial term from collisions)] / A.
            Works for reflective *or* periodic boundaries. For a reflective
            box this should agree with ``method="wall"`` -- two independent
            ways of measuring the same physical pressure, a good sanity check.
        method="ideal"
            The textbook ideal-gas estimate P = N k_B T / A. A theoretical
            reference, not a measurement from the dynamics (ignores particle
            size and collisions entirely) -- see also :meth:`ideal_gas_pressure`.

        Default: ``"wall"`` for reflective boundaries, ``"virial"`` for
        periodic (the only one that works there).
        """
        if method is None:
            method = "virial" if self.periodic else "wall"
        if method not in PRESSURE_METHODS:
            raise ValueError(f"method must be one of {PRESSURE_METHODS}")
        if method == "wall":
            if self.periodic:
                raise ValueError(
                    "method='wall' needs reflective walls to measure "
                    "momentum transfer -- this box is periodic, so nothing "
                    "ever touches a wall. Use method='virial' instead (or "
                    "method='ideal' for the theoretical estimate)."
                )
            return analysis.pressure_wall(self.impulse, self.total_time,
                                          self.Lx, self.Ly)
        if method == "virial":
            temperature_K = float(np.mean(self.calculate("temperature")))
            return analysis.pressure_virial(
                self.n_particles, temperature_K, self.area, self.virial,
                self.total_time, dim=self.dim,
            )
        return self.ideal_gas_pressure()  # method == "ideal"

    def ideal_gas_pressure(self, temperature_K=None) -> float:
        """Reference 2D ideal-gas pressure for comparison with ``calculate('pressure')``."""
        if temperature_K is None:
            temperature_K = float(np.mean(self.calculate("temperature")))
        return analysis.ideal_gas_pressure(self.n_particles, temperature_K,
                                           self.area)

    # -- visualisation ------------------------------------------------------
    def show(self, color_by="speed", vectors=False, fps=15, speed=1.0,
             display_scale=None, figsize=(6, 6)):
        """Play back the recorded trajectory with play/pause/scrub controls.

        Uses ``ipywidgets.Play`` when available (arrow/slider controls, like a
        media player, including scrubbing back to inspect a collision); falls
        back to a simple forward-only autoplay if ipywidgets isn't installed.
        No HTML file either way. ``speed`` rescales the playback pace (2.0 =
        twice as fast, 0.5 = half as fast). ``color_by`` is ``None``,
        ``"speed"`` or ``"mass"``. ``display_scale`` overrides the particle
        draw size for this call only (default: the size set on the ``Box``).
        ``fps`` is a target frame rate. With the default inline backend,
        redrawing has real, roughly fixed cost per frame, so actual playback
        is capped to whatever this machine can redraw+encode in time
        (measured automatically). For genuinely smooth playback, install
        ``ipympl`` and run ``%matplotlib widget`` before creating simulations
        -- this switches to a live canvas with no per-frame encoding cost.
        The title shows elapsed time and instantaneous temperature.
        """
        ds = display_scale if display_scale is not None else self.display_scale
        return viz.play(self.pos, self.vel, self.times, self.mass, self.radius,
                        self.Lx, self.Ly, display_scale=ds, vectors=vectors,
                        color_by=color_by, fps=fps, speed=speed, dim=self.dim,
                        figsize=figsize)

    def histogram(self, quantity="speeds", frame=-1, bins=40,
                  compare_maxwell_boltzmann=True, ax=None):
        """Histogram of speeds at a given frame, vs Maxwell-Boltzmann.

        ``frame`` selects a trajectory frame (default the last). Use
        ``frame='all'`` to pool every recorded frame together for smoother
        statistics.
        """
        if quantity.lower() not in ("speeds", "speed"):
            raise ValueError("histogram currently supports quantity='speeds'")
        if frame == "all":
            vel = self.vel
        else:
            vel = self.vel[frame]
        spd = analysis.speeds(vel)

        T = m = None
        if compare_maxwell_boltzmann:
            T = float(np.mean(analysis.temperature(vel, self.mass, dim=self.dim)))
            m = float(np.mean(self.mass))
        return viz.histogram(spd, temperature_K=T, mass_kg=m, dim=self.dim,
                             bins=bins, ax=ax)

    def __repr__(self):
        return (f"<Simulation N={self.n_particles} frames={self.n_frames} "
                f"t={self.total_time * 1e12:.2f} ps>")
