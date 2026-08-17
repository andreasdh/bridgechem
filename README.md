# bridgechem

Simulations to bridge the gap between the microscopic, macroscopic and symbolic
level in chemistry.

`bridgechem` is a small, teaching-oriented particle-simulation library. You fill
a box with `N` particles and watch them evolve in a Jupyter notebook, then pull
out speed distributions, temperature and pressure and compare them with the
theory students meet in physical chemistry.

Boxes can be **2D or 3D**. The number of side lengths you give decides which:
two numbers give a flat box, three give a cube. Everything downstream follows
from that -- equipartition uses `dim/2 k_B T` per particle, the speed
distribution uses the 2D (Rayleigh) or 3D Maxwell-Boltzmann form, and the
pressure comes out in N/m or Pa.

The goal is **didactic**, not to be a production MD package — but the simple
systems it produces (velocity distributions, pressures, …) do agree with the
theory for simple real systems.

## Install

```bash
pip install -e ".[dev]"     # numpy, matplotlib, ipywidgets, numba, ipympl, pytest
```

numba and ipympl are required: numba JIT-compiles the physics (a pure-Python
fallback exists as a last resort, but is much slower and not the intended
path), and ipympl is what makes playback live and 3D scenes rotatable inside
Jupyter -- see "Playback" below. ipywidgets is needed for play/pause/scrub
controls; without it, playback falls back to a simple forward-only autoplay.

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
can actually push a frame, and never promise more than that -- asking the
`Play` widget to tick faster than the kernel can draw is what causes stutter
(and can make played-back frames appear to arrive out of order). If playback
still looks choppy, pass a lower `fps` explicitly.

### Playback: live 2D, rotatable 3D

Inside a real Jupyter kernel, `bridgechem` switches to the `ipympl`
(`%matplotlib widget`) backend the first time you call `run()`/`show()`
(unless you've already picked a backend yourself) and drives its live canvas
directly, instead of the default `%matplotlib inline` behaviour of
re-rendering the whole figure to a PNG and shipping a fresh image every
frame. That per-frame PNG round trip -- not the physics -- is what causes
choppy playback; on the live canvas, 2D playback is true-blitted (only the
particles redraw, not the axes/labels/colorbar) and 3D playback redraws in
place without ever leaving Python.

A **3D box gets a real, mouse-rotatable 3D scene** by default -- drag to spin
it, even mid-playback -- not a 2D projection that flattens the z-axis (which
is why particles at different depths used to pile up into a chaotic mess on
screen: nothing actually overlapped in 3D, the projection just discarded the
coordinate that would have kept them apart). Pass `slab=2.0` to
`run()`/`show()` instead to view a thin 2 nm slice through the middle of the
box in 2D, when a full 3D view is too busy to read at a glance and you'd
rather see a single plane where every collision on screen really is one.

Particles are drawn at their true collision radius by default -- except when
that would be a bad picture. A `packing`-driven radius spread over very few
particles can demand a size approaching a third of the box (especially in
3D); a real, physically accurate radius (a gas's actual van der Waals
radius, a couple of angstrom) is too small to see at all. Drawn size is
auto-balanced into a sane range for both ends without touching *relative*
spacing between particles -- a dense, liquid-like arrangement still reads as
denser than a sparse, gas-like one, only the common scale changes. Pass
`display_scale` to push the result bigger or smaller yourself.

Outside a live kernel (a script, a test, or without `ipympl` installed)
playback falls back to the old PNG-per-frame approach, which still works
everywhere -- just without blitting or rotation.

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

- A 2D or 3D box of hard spheres with **reflective** (default) or **periodic**
  walls, or (once `add_interactions` is called) continuous Lennard-Jones
  forces integrated with velocity-Verlet.
- Elastic particle–particle and particle–wall collisions (energy- and
  momentum-conserving) for the hard-sphere engine.
- **Interactive playback** in Jupyter (live `ipympl` canvas, no HTML file):
  play, pause, and scrub through the trajectory with an `ipywidgets.Play`
  widget, with big, auto-sized particles and optional velocity-vector arrows.
  2D playback is true-blitted; a 3D box gets a real, mouse-rotatable 3D scene.
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

## Development

```bash
pytest        # physics + API tests
```
