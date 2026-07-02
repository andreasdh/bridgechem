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

Redrawing a figure and shipping it to the browser has real, fairly fixed cost
(tens of milliseconds) that has nothing to do with the physics. `fps` is
therefore only a *target*: on the first frame we measure how fast this machine
can actually redraw+encode, and never promise more than that -- asking the
`Play` widget to tick faster than the kernel can draw is what causes stutter
(and can make played-back frames appear to arrive out of order). If playback
still looks choppy, pass a lower `fps` explicitly.

`run()` still returns a `Simulation` you can analyse:

```python
sim.histogram("speeds")                  # compare with Maxwell-Boltzmann
sim.calculate("pressure")                # 2D pressure (N/m) -- see "Pressure" below
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

## Pressure

`sim.calculate("pressure", method=...)` supports three methods, each teaching
something different about where "pressure" comes from:

| `method`     | How it's computed | Needs | Notes |
|--------------|--------------------|-------|-------|
| `"wall"`     | Momentum transferred to the container walls per unit time and length -- literally what a pressure gauge on the wall would read. | reflective boundaries | Default for reflective boxes; raises a clear error on a periodic box instead of silently returning zero. |
| `"virial"`   | The Clausius virial theorem from particle-particle collisions/forces: `P = [N k_B T + virial_term] / A`. | works either way | Default for periodic boxes (the only one that works there); should agree with `"wall"` for a reflective box -- two independent measurements of the same physical pressure. |
| `"ideal"`    | The textbook estimate `P = N k_B T / A`. | nothing | A theoretical reference, not a measurement -- ignores particle size and collisions entirely. Same as `sim.ideal_gas_pressure()`. |

## Interactions and phase transitions

By default particles are hard spheres with no forces between them (an ideal
gas). `add_interactions` switches on Lennard-Jones forces and moves the engine
to velocity-Verlet integration -- the steep repulsive core of LJ keeps
particles apart continuously, so there's no separate collision step once
interactions are on:

```python
system = bc.box(N=200, size=(7, 7), gas="argon", temperature=300, boundary="periodic")
system.add_interactions("LJ")             # epsilon/sigma default to the box's gas
sim = system.run(steps=20000)             # method="velocity-verlet" is chosen automatically
```

Pass `epsilon` (kelvin, i.e. epsilon/k_B) and/or `sigma` (nm) to override the
gas defaults, or `gas=` to borrow another gas's parameters. Periodic boundaries
are recommended for interacting systems -- it's the standard choice for bulk
gas/liquid MD, and reflective walls have a small, expected energy-conservation
cost from clamping a particle's position at the instant it bounces (`_auto_dt`
compensates automatically, but periodic still conserves energy better).

`set_temperature` ramps the temperature during the *next* `run()` call --
combine it with interactions to cool a gas and watch it condense, a real phase
transition (an ideal gas without interactions has no phase transition, so this
is mostly useful once LJ is on):

```python
system.set_temperature(target_temperature=20, rate=50)  # cool to 20 K at 50 K/ps
sim = system.run(steps=40000)
```

Omit `rate` to jump to the target immediately instead of ramping. Track the
condensation with `sim.calculate("potential_energy")` (drops sharply as
particles bind together) alongside `sim.calculate("temperature")`.

## What's implemented

- A 2D box of hard spheres with **reflective** (default) or **periodic** walls,
  or (once `add_interactions` is called) continuous Lennard-Jones forces
  integrated with velocity-Verlet.
- Elastic particle–particle and particle–wall collisions (energy- and
  momentum-conserving) for the hard-sphere engine.
- **Interactive playback** in Jupyter (default inline backend, no HTML): play,
  pause, and scrub through the trajectory with an `ipywidgets.Play` widget,
  with big, auto-sized particles and optional velocity-vector arrows.
- Colour particles by instantaneous **speed** or by (fixed) **mass**
  (`color_by="mass"`, after `system.set_mass(...)`).
- `system.set_mass(mass, indices=...)` to build a mixture -- e.g. a light/heavy
  pair to watch differential collision behaviour.
- `system.add_interactions("LJ")` (or the alias `"dispersion"`) for
  Lennard-Jones forces, with configurable `epsilon`/`sigma`/`cutoff`.
- `system.set_temperature(target, rate=...)` to ramp temperature during a run
  -- watch a gas condense as it cools.
- Velocities initialised from Maxwell–Boltzmann, or all at the same speed
  (`velocity_init="uniform_speed"`) to watch a distribution relax under collisions.
- Analysis: speeds, temperature, kinetic/potential/total energy, and pressure
  via three methods (see "Pressure" above) -- larger particles read a bit high,
  the excluded-area effect of finite size, real physics.
- Reference gases (mass, hard-sphere radius, and LJ epsilon/sigma): argon,
  helium, neon, krypton, xenon.

See [`examples/demo.ipynb`](examples/demo.ipynb) for a guided tour.

## Roadmap

- Custom pairwise potentials beyond Lennard-Jones.
- **3D** boxes and richer visualisation.

## Development

```bash
pytest        # physics + API tests
```
