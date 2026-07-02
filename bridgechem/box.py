"""The simulation box -- the main user-facing object.

    import bridgechem as bc
    system = bc.box(N=1000, size=(40, 40))     # 40 nm x 40 nm, argon at 300 K
    sim = system.run(steps=20000)
    sim.show()

Without interactions, particles are hard spheres: they fly ballistically and
collide elastically with each other and with the walls (or wrap around, with
periodic boundaries). Calling :meth:`Box.add_interactions` switches to
Lennard-Jones forces integrated with velocity-Verlet -- the steep repulsive
core of LJ prevents overlap continuously, so there's no separate collision
step once interactions are on. :meth:`Box.set_temperature` ramps the
temperature during the next ``run()``, for watching phase transitions.
"""

from __future__ import annotations

import numpy as np

from . import kernels
from .constants import AMU, K_B, NM, DEFAULT_GAS, gas_properties
from .simulation import Simulation

VALID_INTERACTIONS = ("LJ", "dispersion")  # "dispersion" is an alias for LJ
DEFAULT_LJ_CUTOFF = 2.5  # in units of sigma, standard truncation for LJ MD


def _lattice_positions(N, Lx, Ly, r_max):
    """Place N particles on a regular grid that fits inside the box."""
    aspect = Lx / Ly
    ncols = max(1, int(np.ceil(np.sqrt(N * aspect))))
    nrows = int(np.ceil(N / ncols))
    dx = Lx / ncols
    dy = Ly / nrows
    if dx <= 2.0 * r_max or dy <= 2.0 * r_max:
        raise ValueError(
            "Too many particles for this box/radius: they cannot be placed "
            "without overlap. Increase `size`, decrease `N`, or use a smaller "
            "`radius`."
        )
    pos = np.empty((N, 2))
    k = 0
    for j in range(nrows):
        for i in range(ncols):
            if k >= N:
                break
            pos[k, 0] = (i + 0.5) * dx
            pos[k, 1] = (j + 0.5) * dy
            k += 1
    return pos


class Box:
    """A 2D box of particles.

    Parameters
    ----------
    N : int
        Number of particles.
    size : tuple(float, float)
        Box side lengths in **nanometres**.
    gas : str
        Reference gas providing the default particle mass ("argon", "helium", ...).
    mass : float, optional
        Particle mass in **amu**, overrides the gas default.
    radius : float, optional
        Hard-sphere radius in **nanometres**. If omitted the radius is chosen
        automatically from ``packing`` so the particles are big and easy to see
        (this is what you collide with *and* what is drawn). For a physically
        dilute point-like gas, pass a small explicit radius.
    packing : float
        Target fraction of the box area covered by particles, used to pick the
        default radius (ignored if ``radius`` is given). ~0.10 gives a lively,
        clearly visible gas.
    temperature : float
        Initial temperature in K (used to sample velocities).
    boundary : {"reflective", "periodic"}
        Wall behaviour. Reflective walls also let us measure pressure.
    velocity_init : {"thermal", "uniform_speed"}
        "thermal" samples a Maxwell-Boltzmann distribution at ``temperature``;
        "uniform_speed" gives every particle the same speed (random direction)
        -- handy for watching a distribution relax to Maxwell-Boltzmann.
    display_scale : float
        Visual size multiplier for drawing particles (1.0 = draw at true
        collision size).
    seed : int, optional
        Seed for reproducible initial velocities.
    """

    def __init__(self, N, size=(20.0, 20.0), *, gas=DEFAULT_GAS, mass=None,
                 radius=None, packing=0.10, temperature=300.0,
                 boundary="reflective", velocity_init="thermal",
                 display_scale=1.0, dim=2, seed=None):
        if dim != 2:
            raise NotImplementedError("bridgechem currently supports 2D only.")
        self.N = int(N)
        self.dim = 2
        self.Lx = float(size[0]) * NM
        self.Ly = float(size[1]) * NM
        self.temperature = float(temperature)
        self.display_scale = float(display_scale)
        self._gas = gas

        props = gas_properties(gas)
        m_kg = mass * AMU if mass is not None else props["mass_kg"]
        if radius is not None:
            r_m = radius * NM
        else:
            # size the particles so they cover ~`packing` of the box area
            r_m = np.sqrt(packing * self.area / (self.N * np.pi))
        self.mass = np.full(self.N, m_kg, dtype=float)
        self.radius = np.full(self.N, r_m, dtype=float)
        self.inv_mass = 1.0 / self.mass

        if boundary not in ("reflective", "periodic"):
            raise ValueError("boundary must be 'reflective' or 'periodic'")
        self.boundary = boundary
        self.periodic = boundary == "periodic"

        self._rng = np.random.default_rng(seed)
        self.pos = _lattice_positions(self.N, self.Lx, self.Ly, r_m)
        self.vel = self._init_velocities(velocity_init)

        # set via add_interactions() / set_temperature()
        self._interaction = None
        self._has_interactions = False
        self._thermostat = None

    # -- initialisation -----------------------------------------------------
    def _init_velocities(self, mode):
        m = self.mass[0]
        if mode == "thermal":
            sigma = np.sqrt(K_B * self.temperature / m)
            vel = self._rng.normal(0.0, sigma, size=(self.N, 2))
        elif mode == "uniform_speed":
            v_rms = np.sqrt(2.0 * K_B * self.temperature / m)  # 2D rms speed
            angle = self._rng.uniform(0.0, 2.0 * np.pi, size=self.N)
            vel = v_rms * np.stack([np.cos(angle), np.sin(angle)], axis=1)
        else:
            raise ValueError("velocity_init must be 'thermal' or 'uniform_speed'")
        # remove centre-of-mass drift and rescale to the exact target T
        vel -= vel.mean(axis=0)
        vel = self._rescale_to_temperature(vel, self.temperature)
        return vel

    def _rescale_to_temperature(self, vel, target_T):
        v2 = np.sum(vel ** 2, axis=1)
        current_T = np.mean(self.mass * v2) / (self.dim * K_B)
        if current_T > 0:
            vel = vel * np.sqrt(target_T / current_T)
        return vel

    # -- convenience --------------------------------------------------------
    @property
    def area(self):
        return self.Lx * self.Ly

    def set_mass(self, mass=None, *, gas=None, indices=None):
        """Set the mass of some or all particles, for mixtures.

        Pass exactly one of ``mass`` (a number or per-particle array, in amu)
        or ``gas`` (a reference gas name, e.g. ``"helium"``). ``indices``
        selects which particles to change (default: all) -- e.g. an array or
        boolean mask to make half the gas heavier and watch differential
        diffusion. Existing velocities are left untouched, so a particle's
        kinetic energy (and its contribution to the measured temperature)
        changes immediately; collisions will re-equilibrate the mixture over
        time.
        """
        if (mass is None) == (gas is None):
            raise ValueError("pass exactly one of `mass` or `gas`")
        m_kg = gas_properties(gas)["mass_kg"] if gas is not None else np.asarray(mass, dtype=float) * AMU
        if np.any(np.asarray(m_kg) <= 0):
            raise ValueError("mass must be positive")
        if indices is None:
            self.mass[:] = m_kg
        else:
            self.mass[indices] = m_kg
        self.inv_mass = 1.0 / self.mass
        return self

    def _auto_dt(self, safety=0.2, safety_curvature=0.01, safety_speed=0.01):
        """Pick a time step small enough to integrate accurately.

        Without interactions: small enough that particles can't tunnel
        through each other between hard-sphere collision checks.

        With LJ interactions, two effects can each demand a smaller step, so
        we take whichever is stricter: (1) resolving the oscillation near the
        potential well, timescale ``sigma * sqrt(mass / epsilon)``; (2) fast
        (hot) particles can plough deep into the steep repulsive core within
        a single step, which needs ``dt`` to shrink with speed too, not just
        with the well's curvature. Both fractions were tuned empirically
        (see tests) to keep energy drift well under 1% from 30 K to 3000 K.
        Reflective walls clamp a particle's position on the same step forces
        are recomputed, a small non-symplectic correction that (unlike a
        periodic box) leaks energy over many bounces -- roughly linearly in
        dt -- so we shrink dt further in that case too.
        """
        if self._has_interactions:
            ia = self._interaction
            tau = ia["sigma"] * np.sqrt(self.mass.min() / ia["epsilon"])
            dt_curvature = safety_curvature * tau
            vmax = np.sqrt(np.sum(self.vel ** 2, axis=1)).max()
            dt = dt_curvature if vmax <= 0 else min(
                dt_curvature, safety_speed * ia["sigma"] / vmax)
            return dt if self.periodic else 0.25 * dt
        vmax = np.sqrt(np.sum(self.vel ** 2, axis=1)).max()
        rmin = self.radius.min()
        if vmax <= 0:
            # no motion; fall back to a sane molecular timescale
            return 1e-14
        return safety * rmin / vmax

    # -- live stepping (loop-style API) ------------------------------------
    def advance(self, dt=None, steps=1):
        """Advance the live system state in place by ``steps`` steps.

        Enables the explicit-loop style::

            while t < t_end:
                system.advance()
                t += system.last_dt
        """
        if dt is None:
            dt = self._auto_dt()
        self.last_dt = dt
        impulse = np.zeros(2)
        virial = np.zeros(1)
        if self._has_interactions:
            ia = self._interaction
            forces = np.zeros((self.N, 2))
            kernels._lj_forces(self.pos, self.Lx, self.Ly, self.periodic,
                               ia["epsilon"], ia["sigma"] ** 2, ia["r_cut2"],
                               ia["u_shift"], forces)
            for _ in range(int(steps)):
                kernels._step_lj(self.pos, self.vel, forces, self.radius,
                                 self.inv_mass, self.Lx, self.Ly, dt,
                                 self.periodic, ia["epsilon"], ia["sigma"] ** 2,
                                 ia["r_cut2"], ia["u_shift"], impulse, virial)
        else:
            for _ in range(int(steps)):
                kernels._step(self.pos, self.vel, self.radius, self.inv_mass,
                              self.Lx, self.Ly, dt, self.periodic, impulse,
                              virial)
        return self

    # ``integrate`` is kept as an alias so the loop sketch in the design notes
    # works; for the hard-sphere engine a step is ballistic move + collisions.
    integrate = advance

    # -- batch run ----------------------------------------------------------
    def run(self, steps=1000, *, t=None, dt=None, sample_every=None,
            method=None, animate=None, vectors=False,
            color_by="speed", fps=15, speed=1.0, display_scale=None,
            figsize=(6, 6)):
        """Run the simulation and return a :class:`Simulation` with the trajectory.

        The whole trajectory is computed first (numba-accelerated, typically
        well under a second), then -- in a Jupyter notebook -- displayed with
        play/pause/scrub controls (no HTML file, nothing extra to install
        beyond ``ipywidgets``). Outside a notebook it just runs headless.

        Parameters
        ----------
        steps : int
            Number of integration steps (``t`` is accepted as an alias).
        dt : float, optional
            Time step in seconds. Chosen automatically if omitted.
        sample_every : int, optional
            Record a frame every this many steps. Chosen automatically from
            ``speed`` if omitted -- prefer tuning ``speed`` over this.
        method : str, optional
            "hard-sphere" or "velocity-verlet". Chosen automatically:
            "velocity-verlet" once :meth:`add_interactions` has been called
            (required, since forces need integrating), "hard-sphere"
            otherwise.
        animate : bool, optional
            Display the trajectory with play/pause controls. Defaults to True
            inside a notebook, False otherwise.
        vectors : bool
            Draw a velocity arrow on each particle.
        color_by : None, "speed" or "mass"
            Colour particles by their instantaneous speed, or by (fixed)
            particle mass -- handy after :meth:`set_mass` to spot a mixture.
        fps : float
            Target frames per second (visual smoothness only -- does not
            change how fast the simulation *looks* like it's moving; use
            ``speed`` for that). Redrawing has real, roughly fixed cost per
            frame, so actual playback is capped to whatever this machine can
            redraw+encode in time (measured automatically) -- if it's still
            choppy, try a lower ``fps``.
        speed : float
            Pedagogical playback speed. At the default ``speed=1`` a
            mean-speed particle takes a few seconds to cross the box, slow
            enough to actually follow collisions. ``speed=3`` plays three
            times faster, ``speed=0.3`` about three times slower. This does
            not change the physics, only how many physics steps are grouped
            into each displayed frame.
        display_scale : float, optional
            Visual size multiplier for drawn particles, overriding the box's
            default for this call.
        """
        if t is not None:
            steps = t
        steps = int(steps)
        if method is None:
            method = "velocity-verlet" if self._has_interactions else "hard-sphere"
        if method not in ("hard-sphere", "velocity-verlet"):
            raise ValueError("method must be 'hard-sphere' or 'velocity-verlet'")
        if method == "hard-sphere" and self._has_interactions:
            raise ValueError(
                "this box has interactions (add_interactions was called); "
                "method must be 'velocity-verlet' (or leave method unset)."
            )
        if dt is None:
            dt = self._auto_dt()

        from . import viz
        if sample_every is None:
            mean_speed = float(np.sqrt(np.sum(self.vel ** 2, axis=1)).mean())
            sample_every = viz.pick_sample_every(
                mean_speed, dt, self.Lx, self.Ly, fps=fps, speed=speed,
            )
            # cap total stored frames for very long or very fast-playing runs
            if steps // sample_every + 1 > viz.MAX_FRAMES:
                sample_every = max(sample_every, -(-steps // viz.MAX_FRAMES))

        th = self._thermostat
        thermostat = th is not None
        T_start = th["T_start"] if thermostat else 0.0
        T_target = th["T_target"] if thermostat else 0.0
        rate = th["rate"] if thermostat else 0.0

        if self._has_interactions:
            ia = self._interaction
            (traj_pos, traj_vel, traj_pe, times, impulse,
             virial) = kernels._simulate_lj(
                self.pos, self.vel, self.radius, self.inv_mass,
                self.Lx, self.Ly, dt, steps, sample_every, self.periodic,
                ia["epsilon"], ia["sigma"], ia["r_cut2"], ia["u_shift"],
                thermostat, T_start, T_target, rate, self.dim,
            )
        else:
            traj_pos, traj_vel, times, impulse, virial = kernels._simulate(
                self.pos, self.vel, self.radius, self.inv_mass,
                self.Lx, self.Ly, dt, steps, sample_every, self.periodic,
                thermostat, T_start, T_target, rate, self.dim,
            )
            traj_pe = None
        self._thermostat = None  # thermostat, if any, only applies for one run()

        # update live state to the end of the run
        self.pos = traj_pos[-1].copy()
        self.vel = traj_vel[-1].copy()

        sim = Simulation(
            traj_pos, traj_vel, times, impulse, virial,
            mass=self.mass, radius=self.radius, Lx=self.Lx, Ly=self.Ly,
            dim=self.dim, periodic=self.periodic, potential_energy=traj_pe,
            display_scale=display_scale if display_scale is not None else self.display_scale,
        )

        if animate is None:
            animate = viz.in_notebook()
        if animate:
            sim.show(color_by=color_by, vectors=vectors, fps=fps, speed=speed,
                    figsize=figsize)
        return sim

    # -- interactions & thermostat -------------------------------------------
    def add_interactions(self, interaction="LJ", *, epsilon=None, sigma=None,
                         gas=None, cutoff=DEFAULT_LJ_CUTOFF):
        """Switch on Lennard-Jones interactions between particles.

        Once added, ``run()`` integrates with velocity-Verlet under the LJ
        force instead of treating particles as hard spheres that bounce off
        each other -- the steep repulsive part of the LJ potential does that
        job continuously, so there's no separate collision step. Walls are
        unaffected (still elastic bounces for reflective boundaries).

        Parameters
        ----------
        interaction : str
            ``"LJ"`` or ``"dispersion"`` (an alias) -- the only interaction
            implemented so far.
        epsilon : float, optional
            Well depth in **kelvin** (i.e. epsilon / k_B), the usual way LJ
            parameters are tabulated. Defaults to the box's gas.
        sigma : float, optional
            Distance in **nm** where the potential crosses zero. Defaults to
            the box's gas.
        gas : str, optional
            Look up epsilon/sigma from a different reference gas than the one
            the box was constructed with.
        cutoff : float
            Truncate the potential beyond ``cutoff * sigma`` (shifted so it's
            continuous there). 2.5 is standard for LJ molecular dynamics.
        """
        interaction_key = interaction if interaction != "dispersion" else "LJ"
        if interaction not in VALID_INTERACTIONS:
            raise ValueError(
                f"interaction must be one of {VALID_INTERACTIONS}; custom "
                "potentials aren't implemented yet."
            )
        props = gas_properties(gas if gas is not None else self._gas)
        epsilon_J = epsilon * K_B if epsilon is not None else props["epsilon_J"]
        sigma_m = sigma * NM if sigma is not None else props["sigma_m"]
        if epsilon_J <= 0 or sigma_m <= 0:
            raise ValueError("epsilon and sigma must be positive")

        r_cut = cutoff * sigma_m
        r_cut2 = r_cut * r_cut
        sr6_cut = (sigma_m / r_cut) ** 6
        u_shift = 4.0 * epsilon_J * (sr6_cut ** 2 - sr6_cut)

        self._interaction = {
            "kind": interaction_key, "epsilon": epsilon_J, "sigma": sigma_m,
            "cutoff": cutoff, "r_cut2": r_cut2, "u_shift": u_shift,
        }
        self._has_interactions = True
        # draw particles at their LJ equilibrium size (2^(1/6) sigma between
        # centres); this is cosmetic only -- LJ forces don't use `radius`.
        self.radius = np.full(self.N, 2.0 ** (1.0 / 6.0) * sigma_m / 2.0)
        return self

    def set_temperature(self, target_temperature, rate=None):
        """Ramp the temperature toward ``target_temperature`` during the next ``run()``.

        Velocities are rescaled every step to track a target that moves
        linearly from the box's current temperature to ``target_temperature``
        at ``rate`` kelvin per picosecond. Omit ``rate`` to jump immediately
        instead of ramping. Combine with :meth:`add_interactions` to watch a
        gas condense as it cools (a phase transition) -- an ideal (hard-sphere,
        no interactions) gas has no phase transition, so this is mostly useful
        once interactions are present. The thermostat applies to the *next*
        ``run()`` call only.
        """
        if target_temperature <= 0:
            raise ValueError("target_temperature must be positive")
        current_T = float(
            np.mean(np.sum(self.vel ** 2, axis=1) * self.mass) / (self.dim * K_B)
        )
        rate_per_s = (rate / 1e-12) if rate is not None else 0.0  # K/ps -> K/s
        self._thermostat = {
            "T_start": current_T, "T_target": float(target_temperature),
            "rate": rate_per_s,
        }
        return self

    def __repr__(self):
        return (f"<Box N={self.N} size=({self.Lx / NM:.1f}, {self.Ly / NM:.1f}) nm "
                f"T={self.temperature:.0f} K boundary={self.boundary}>")


# lower-case alias so the intended API reads ``bc.box(N=...)``
box = Box
