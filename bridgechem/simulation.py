"""The :class:`Simulation` object returned by :meth:`Box.run`.

It holds the recorded trajectory and exposes analysis and visualisation:

    sim = system.run(steps=20000)
    sim.show()                          # animation
    sim.calculate("pressure")           # N/m in 2D, Pa in 3D
    sim.calculate("velocities")         # raw (n_frames, N, dim) array
    sim.pressure(per_frame=True)        # pressure as a time series
    sim.histogram("speeds")             # speed histogram vs Maxwell-Boltzmann
"""

from __future__ import annotations

import numpy as np

from . import analysis, viz

PRESSURE_METHODS = ("wall", "virial", "ideal")


class Simulation:
    def __init__(self, traj_pos, traj_vel, times, impulse, virial=0.0, *,
                 mass, radius, L, periodic=False,
                 potential_energy=None, display_scale=1.0):
        self.pos = np.asarray(traj_pos)          # (n_frames, N, dim), m
        self.vel = np.asarray(traj_vel)          # (n_frames, N, dim), m/s
        self.times = np.asarray(times)           # (n_frames,), s
        self.impulse = np.asarray(impulse)       # (n_frames, dim), kg*m/s
        self.virial = float(virial)              # kg*m**2/s
        self.mass = np.asarray(mass)             # (N,), kg
        self.radius = np.asarray(radius)         # (N,), m
        self.L = np.asarray(L, dtype=float)      # (dim,), m
        self.dim = int(self.L.size)
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
    def volume(self) -> float:
        """Box volume in 3D, box area in 2D (m**dim)."""
        return float(np.prod(self.L))

    @property
    def area(self) -> float:
        """Box area -- 2D only. In 3D, ask for :attr:`volume` instead."""
        if self.dim != 2:
            raise AttributeError(
                f"`area` is only defined in 2D; this run is {self.dim}D. "
                "Use `.volume` instead."
            )
        return self.volume

    @property
    def Lx(self) -> float:
        return float(self.L[0])

    @property
    def Ly(self) -> float:
        return float(self.L[1])

    @property
    def Lz(self) -> float:
        if self.dim < 3:
            raise AttributeError(f"this run is {self.dim}D and has no z axis")
        return float(self.L[2])

    @property
    def pressure_unit(self) -> str:
        """"Pa" in 3D (force per area), "N/m" in 2D (force per length)."""
        return "Pa" if self.dim == 3 else "N/m"

    @property
    def total_time(self) -> float:
        return float(self.times[-1]) if self.times.size else 0.0

    # -- analysis -----------------------------------------------------------
    def calculate(self, quantity: str, method: str = None):
        """Compute a derived quantity from the trajectory.

        Supported: ``velocities``, ``speeds``, ``temperature``,
        ``kinetic_energy``, ``potential_energy``, ``total_energy``,
        ``pressure``, ``mean_speed``.

        ``velocities`` and ``speeds`` return the raw per-particle arrays, so
        you can do the statistics yourself rather than taking the library's
        word for it.

        For ``"pressure"``, ``method`` picks *how* it's computed -- see
        :meth:`pressure` for the physics behind each option.
        """
        q = quantity.lower()
        if q in ("velocities", "velocity"):
            return self.vel
        if q in ("speeds", "speed"):
            return analysis.speeds(self.vel)
        if q in ("temperature", "temp"):
            return analysis.temperature(self.vel, self.mass)
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

    def wall_collisions(self, per_frame: bool = False):
        """Momentum handed to the walls, per axis (kg m/s).

        ``(dim,)`` totals for the whole run, or the ``(n_frames, dim)``
        per-frame record with ``per_frame=True`` -- the raw material for
        working out the pressure yourself:

            impulse = sim.wall_collisions()          # kg m/s per axis
            P = impulse.sum() / (t * total_wall_extent)
        """
        return self.impulse if per_frame else self.impulse.sum(axis=0)

    def pressure(self, method: str = None, per_frame: bool = False):
        """Pressure, computed one of three ways. N/m in 2D, Pa in 3D.

        method="wall"
            Momentum transferred to the container walls per unit time and
            wall area (2D: per unit length) -- literally what a pressure
            gauge on the wall would read. Needs **reflective** boundaries
            (raises otherwise: with periodic boundaries nothing ever touches
            a wall).
        method="virial"
            The Clausius virial theorem applied to particle-particle
            collisions: P = [N k_B T + (virial term)] / V. Works for
            reflective *or* periodic boundaries. For a reflective box this
            should broadly agree with ``method="wall"`` -- two independent
            routes to the same physical pressure. Expect a small gap at high
            packing: particle centres cannot reach closer than one radius to
            a wall, so the accessible volume is a little smaller than V and
            the wall reading sits a little high.
        method="ideal"
            The textbook estimate P = N k_B T / V. A theoretical reference,
            not a measurement from the dynamics (it ignores particle size and
            collisions entirely).

        Default: ``"wall"`` for reflective boundaries, ``"virial"`` for
        periodic (the only one that works there).

        With ``per_frame=True`` the ``"wall"`` method returns a
        ``(n_frames,)`` time series instead of a single number, so you can
        plot P(t), throw away the equilibration period, and watch the
        fluctuations shrink as N grows.
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
            if not per_frame:
                return analysis.pressure_wall(self.impulse, self.total_time,
                                              self.L)
            if self.n_frames < 2:
                return np.zeros(self.n_frames)
            frame_dt = np.diff(self.times, prepend=self.times[0] -
                               (self.times[1] - self.times[0]))
            return np.array([
                analysis.pressure_wall(self.impulse[f], float(frame_dt[f]),
                                       self.L)
                for f in range(self.n_frames)
            ])

        if per_frame:
            raise ValueError(
                "per_frame=True is only available for method='wall' (the "
                "virial is accumulated over the whole run)."
            )
        if method == "virial":
            temperature_K = float(np.mean(self.calculate("temperature")))
            return analysis.pressure_virial(
                self.n_particles, temperature_K, self.volume, self.virial,
                self.total_time, dim=self.dim,
            )
        return self.ideal_gas_pressure()  # method == "ideal"

    def ideal_gas_pressure(self, temperature_K=None) -> float:
        """Reference ideal-gas pressure N k_B T / V, for comparison."""
        if temperature_K is None:
            temperature_K = float(np.mean(self.calculate("temperature")))
        return analysis.ideal_gas_pressure(self.n_particles, temperature_K,
                                           self.volume)

    # -- visualisation ------------------------------------------------------
    def show(self, color_by="speed", vectors=False, fps=15, speed=1.0,
             display_scale=None, figsize=(6, 6), slab=None):
        """Play back the recorded trajectory with play/pause/scrub controls.

        Uses ``ipywidgets.Play`` when available (arrow/slider controls, like a
        media player, including scrubbing back to inspect a collision); falls
        back to a simple forward-only autoplay if ipywidgets isn't installed.

        A 3D run is drawn as its x-y projection. Particles at different depths
        then overlap on screen, so an apparent "collision" may just be a
        near-miss in z. Pass ``slab=2.0`` to show only the particles within a
        2 nm thick slice through the middle of the box, where what you see
        really is what collides.
        """
        ds = display_scale if display_scale is not None else self.display_scale
        return viz.play(self.pos, self.vel, self.times, self.mass, self.radius,
                        self.L, display_scale=ds, vectors=vectors,
                        color_by=color_by, fps=fps, speed=speed,
                        figsize=figsize, slab=slab)

    def histogram(self, quantity="speeds", frame=-1, bins=40,
                  compare_maxwell_boltzmann=True, ax=None):
        """Histogram of speeds at a given frame, vs Maxwell-Boltzmann.

        ``frame`` selects a trajectory frame (default the last). Use
        ``frame='all'`` to pool every recorded frame together for smoother
        statistics.
        """
        if quantity.lower() not in ("speeds", "speed"):
            raise ValueError("histogram currently supports quantity='speeds'")
        vel = self.vel if frame == "all" else self.vel[frame]
        spd = analysis.speeds(vel)

        T = m = None
        if compare_maxwell_boltzmann:
            T = float(np.mean(analysis.temperature(vel, self.mass)))
            m = float(np.mean(self.mass))
        return viz.histogram(spd, temperature_K=T, mass_kg=m, dim=self.dim,
                             bins=bins, ax=ax)

    def __repr__(self):
        return (f"<Simulation {self.dim}D N={self.n_particles} "
                f"frames={self.n_frames} t={self.total_time * 1e12:.2f} ps>")
