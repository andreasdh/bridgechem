# bridgechem

Simulations to bridge the gap between the microscopic, macroscopic and symbolic
level in chemistry.

`bridgechem` is a small, teaching-oriented particle-simulation library. You fill
a box with `N` particles and watch them evolve in a Jupyter notebook, then pull
out speed distributions, temperature and pressure and compare them with the
theory students meet in physical chemistry.

The goal is **didactic**, not to be a production MD package — but the simple
systems it produces (velocity distributions, pressures, …) do agree with the
theory for simple real systems.

## Install

```bash
pip install -e ".[dev]"     # numpy, matplotlib, numba, pytest
```

numba is optional but strongly recommended: the hot loops fall back to pure
Python if it is missing (much slower, but everything still runs).

## Quick start

```python
import bridgechem as bc

system = bc.box(N=1000, size=(80, 80))   # 80 nm x 80 nm box of argon at 300 K
sim = system.run(steps=20000)            # numba-accelerated
sim.show()                               # live animation in the notebook

sim.histogram("speeds")                  # compare with Maxwell-Boltzmann
sim.calculate("pressure")                # 2D pressure (N/m) from wall collisions
sim.calculate("temperature")             # per-frame temperature (K)
```

If you prefer an explicit loop, the same engine is available step by step:

```python
while t < t_end:
    system.advance()      # ballistic move + elastic collisions
    t += system.last_dt
```

## Units

Everything is computed and returned in **SI units** (m/s, K, J, and — in 2D —
pressure as force per length, N/m). For convenience the constructor takes a
couple of chemistry-friendly *input* units, converted to SI immediately:

| Input        | Unit |
|--------------|------|
| box `size`   | nm   |
| `radius`     | nm   |
| `mass`       | amu  |
| `temperature`| K    |

## What milestone 1 covers

- A 2D box of hard spheres with **reflective** (default) or **periodic** walls.
- Elastic particle–particle and particle–wall collisions (energy- and
  momentum-conserving).
- Velocities initialised from Maxwell–Boltzmann, or all at the same speed to
  watch a distribution *relax* to equilibrium.
- Analysis: speeds, temperature, kinetic energy, and pressure (with the 2D
  ideal-gas law `P = N k_B T / A` for comparison — the few-percent excess you
  see is the excluded-area effect of finite particles, real physics).
- Reference gases: argon, helium, neon, krypton, xenon.

See [`examples/demo.ipynb`](examples/demo.ipynb) for a guided tour.

## Roadmap

- **Interactions**: `system.add_interactions("LJ")` / `"dispersion"` / custom
  potentials, velocity-Verlet integration, virial pressure, phase transitions.
- **Thermostatting**: `system.set_temperature(...)` and gradual cooling.
- **3D** boxes and richer visualisation.

## Development

```bash
pytest        # physics + API tests
```
