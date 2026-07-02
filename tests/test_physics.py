"""Physics validation for the hard-sphere gas engine.

These double as didactic sanity checks: energy/momentum conservation, the 2D
ideal-gas law, and relaxation to the Maxwell-Boltzmann distribution.
"""

import numpy as np
import pytest

import bridgechem as bc
from bridgechem.analysis import speeds, mean_speed


def test_energy_and_temperature_conserved_reflective():
    system = bc.box(N=300, size=(40, 40), temperature=300, seed=0)
    sim = system.run(steps=25000, animate=False)

    ke = sim.calculate("kinetic_energy")
    assert (ke.max() - ke.min()) / ke.mean() < 1e-9  # elastic -> exact

    temp = sim.calculate("temperature")
    assert np.allclose(temp, 300.0, rtol=1e-9)


def test_particles_stay_inside_box():
    system = bc.box(N=300, size=(40, 40), temperature=300, seed=0)
    sim = system.run(steps=20000, animate=False)
    assert sim.pos.min() >= 0.0
    assert sim.pos[..., 0].max() <= system.Lx + 1e-18
    assert sim.pos[..., 1].max() <= system.Ly + 1e-18


def test_ideal_gas_law_2d():
    # a dilute (small-radius) gas recovers the point-particle ideal gas law
    system = bc.box(N=300, size=(40, 40), temperature=300, radius=0.1, seed=0)
    sim = system.run(steps=25000, animate=False)
    p = sim.calculate("pressure")
    p_ideal = sim.ideal_gas_pressure()
    assert 0.9 < p / p_ideal < 1.12


def test_finite_size_raises_pressure():
    # larger particles (excluded area) push the pressure above ideal, like a
    # 2D van der Waals / hard-disk gas -- real physics, not a bug.
    dilute = bc.box(N=300, size=(40, 40), temperature=300, radius=0.1, seed=0)
    dense = bc.box(N=300, size=(40, 40), temperature=300, packing=0.15, seed=0)
    r_dilute = dilute.run(steps=20000, animate=False)
    r_dense = dense.run(steps=20000, animate=False)
    z_dilute = r_dilute.calculate("pressure") / r_dilute.ideal_gas_pressure()
    z_dense = r_dense.calculate("pressure") / r_dense.ideal_gas_pressure()
    assert z_dense > z_dilute > 0.95


def test_wall_and_virial_pressure_agree_reflective():
    # two independent measurements of the same physical pressure should
    # roughly agree (a few percent discretization-driven discrepancy is
    # expected: collisions are detected a step after exact contact).
    system = bc.box(N=250, size=(35, 35), temperature=300, packing=0.10, seed=0)
    sim = system.run(steps=20000, animate=False)
    p_wall = sim.calculate("pressure", method="wall")
    p_virial = sim.calculate("pressure", method="virial")
    assert 0.85 < p_virial / p_wall < 1.15
    assert sim.calculate("pressure") == p_wall  # default for reflective


def test_wall_pressure_raises_for_periodic():
    system = bc.box(N=100, size=(30, 30), boundary="periodic", seed=0)
    sim = system.run(steps=2000, animate=False)
    with pytest.raises(ValueError):
        sim.calculate("pressure", method="wall")


def test_virial_pressure_works_for_periodic_and_matches_reflective_bulk():
    # bulk pressure of a homogeneous gas shouldn't depend on whether the
    # boundary happens to be a wall or periodic -- both measure the same
    # density/temperature via the virial (only "wall" needs an actual wall).
    common = dict(N=250, size=(35, 35), temperature=300, packing=0.10, seed=0)
    reflective = bc.box(boundary="reflective", **common).run(steps=20000, animate=False)
    periodic = bc.box(boundary="periodic", **common).run(steps=20000, animate=False)
    p_reflective = reflective.calculate("pressure", method="virial")
    p_periodic = periodic.calculate("pressure", method="virial")
    assert 0.85 < p_periodic / p_reflective < 1.15
    assert periodic.calculate("pressure") == p_periodic  # default for periodic


def test_pressure_invalid_method_raises():
    system = bc.box(N=50, size=(20, 20), seed=0)
    sim = system.run(steps=1000, animate=False)
    with pytest.raises(ValueError):
        sim.calculate("pressure", method="nonsense")


def test_momentum_conserved_periodic():
    system = bc.box(N=300, size=(40, 40), temperature=300,
                    boundary="periodic", seed=5)
    p0 = (system.mass[:, None] * system.vel).sum(axis=0)
    sim = system.run(steps=8000, animate=False)
    pf = (sim.mass[:, None] * sim.vel[-1]).sum(axis=0)
    scale = system.mass[0] * speeds(system.vel).mean() * system.N
    assert np.linalg.norm(pf - p0) / scale < 1e-10

    ke = sim.calculate("kinetic_energy")
    assert (ke.max() - ke.min()) / ke.mean() < 1e-9


def test_relaxation_to_maxwell_boltzmann():
    # start every particle at the same speed; collisions should broaden it into
    # the 2D Maxwell-Boltzmann (Rayleigh) distribution.
    system = bc.box(N=400, size=(45, 45), temperature=300,
                    velocity_init="uniform_speed", seed=3)
    sp0 = speeds(system.vel)
    assert abs(np.mean(sp0 ** 2) / sp0.mean() ** 2 - 1.0) < 0.02  # ~ delta

    sim = system.run(steps=20000, animate=False)
    spf = speeds(sim.vel[-1])
    ratio = np.mean(spf ** 2) / spf.mean() ** 2
    assert abs(ratio - 4.0 / np.pi) < 0.08  # Rayleigh signature

    # mean speed should match the analytic Maxwell-Boltzmann value
    assert abs(spf.mean() - mean_speed(300, system.mass[0])) / spf.mean() < 0.05


def test_periodic_wraps_positions():
    system = bc.box(N=200, size=(30, 30), temperature=300,
                    boundary="periodic", seed=1)
    sim = system.run(steps=5000, animate=False)
    assert sim.pos.min() >= 0.0
    assert sim.pos[..., 0].max() < system.Lx
    assert sim.pos[..., 1].max() < system.Ly
