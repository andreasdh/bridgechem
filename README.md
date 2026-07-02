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
pip install -e ".[dev]"     # numpy, matplotlib, ipywidgets, numba, pytest
```

numba is optional but strongly recommended: the hot loops fall back to pure
Python if it is missing (much slower, but everything still runs). ipywidgets
is needed for play/pause controls; without it, playback falls back to a
simple forward-only autoplay.

## Quick start

```python
import bridgechem as bc

system = bc.box(N=200, size=(40, 40))    # 40 nm x 40 nm box of argon at 300 K
sim = system.run(steps=20000, vectors=True)
```

The whole trajectory is computed first (numba-accelerated, typically under a
second), then displayed in a Jupyter notebook with **play / pause / scrub**
controls (an `ipywidgets.Play` widget) — no HTML file, no separate `show()`
call needed, and you can pause and drag the slider back to inspect a specific
collision. Particles are auto-sized to be big and easy to see (and drawn at
their true collision size); `vectors=True` overlays velocity arrows.

Real gas particles move at hundreds of m/s, far too fast to watch, so playback is
paced by a `speed` knob rather than shown at true speed: at the default `speed=1`
a typical particle takes a few seconds to cross the box. `speed=3` plays three
times faster, `speed=0.3` about three times slower -- this only changes the
*display* pace, never the underlying physics (energy, pressure, temperature are
computed from the real SI dynamics regardless of `speed`).

`run()` still returns a `Simulation` you can analyse:

```python
sim.histogram("speeds")                  # compare with Maxwell-Boltzmann
sim.calculate("pressure")                # 2D pressure (N/m) from wall collisions
sim.calculate("temperature")             # per-frame temperature (K)
sim.show()                               # replay the recorded run, with controls
sim.show(color_by="mass", display_scale=1.5)  # colour by mass, bigger particles
```

Use `animate=False` to run headless at full speed (e.g. for pressure statistics),
and the explicit-loop style is available too:

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
- **Interactive playback** in Jupyter (default inline backend, no HTML): play,
  pause, and scrub through the trajectory with an `ipywidgets.Play` widget,
  with big, auto-sized particles and optional velocity-vector arrows.
- Colour particles by instantaneous **speed** or by (fixed) **mass**
  (`color_by="mass"`, after `system.set_mass(...)`).
- `system.set_mass(mass, indices=...)` to build a mixture -- e.g. a light/heavy
  pair to watch differential collision behaviour.
- Velocities initialised from Maxwell–Boltzmann, or all at the same speed to
  watch a distribution *relax* to equilibrium.
- Analysis: speeds, temperature, kinetic energy, and pressure (with the 2D
  ideal-gas law `P = N k_B T / A` for comparison — larger particles read a bit
  high, the excluded-area effect of finite size, real physics).
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
