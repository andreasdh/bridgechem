"""API-surface tests: construction, options, and the analysis dispatch."""

import numpy as np
import pytest

import bridgechem as bc


def test_box_construction_and_units():
    system = bc.box(N=100, size=(20, 20), temperature=250, seed=0)
    assert system.N == 100
    assert np.isclose(system.Lx, 20e-9)  # nm -> m
    assert np.isclose(system.Ly, 20e-9)
    assert system.vel.shape == (100, 2)
    assert system.pos.shape == (100, 2)
    # velocities sampled at the requested temperature
    T = bc.analysis.temperature(system.vel, system.mass, dim=2)
    assert np.isclose(T, 250, rtol=1e-9)


def test_gas_presets_change_mass():
    ar = bc.box(N=10, size=(20, 20), gas="argon")
    he = bc.box(N=10, size=(20, 20), gas="helium")
    assert he.mass[0] < ar.mass[0]


def test_custom_mass_and_radius_override():
    system = bc.box(N=10, size=(20, 20), mass=10.0, radius=0.1)
    assert np.isclose(system.mass[0], 10.0 * bc.constants.AMU)
    assert np.isclose(system.radius[0], 0.1e-9)


def test_run_returns_simulation_with_expected_shapes():
    system = bc.box(N=50, size=(20, 20), seed=0)
    sim = system.run(steps=2000, sample_every=100, animate=False)
    assert isinstance(sim, bc.Simulation)
    assert sim.n_particles == 50
    assert sim.pos.shape[1:] == (50, 2)
    assert sim.vel.shape == sim.pos.shape
    assert sim.times.shape[0] == sim.n_frames


def test_t_alias_for_steps():
    system = bc.box(N=20, size=(20, 20), seed=0)
    sim = system.run(t=1000, sample_every=100, animate=False)
    assert sim.total_time > 0


def test_calculate_dispatch():
    system = bc.box(N=40, size=(20, 20), seed=0)
    sim = system.run(steps=2000, sample_every=200, animate=False)
    assert bc.analysis.speeds(sim.vel).shape == sim.vel.shape[:-1]
    assert np.isscalar(sim.calculate("pressure")) or np.ndim(sim.calculate("pressure")) == 0
    assert sim.calculate("mean_speed") > 0
    assert sim.calculate("temperature").shape[0] == sim.n_frames
    with pytest.raises(ValueError):
        sim.calculate("nonsense")


def test_auto_radius_targets_packing():
    # with no explicit radius, particles are sized to ~`packing` of the box
    system = bc.box(N=200, size=(30, 30), packing=0.15)
    covered = system.N * np.pi * system.radius[0] ** 2 / system.area
    assert np.isclose(covered, 0.15, rtol=1e-6)


def test_too_many_particles_raises():
    with pytest.raises(ValueError):
        bc.box(N=100000, size=(5, 5), radius=0.05)


def test_advance_steps_live_state():
    system = bc.box(N=30, size=(20, 20), seed=0)
    pos_before = system.pos.copy()
    system.advance(steps=10)
    # positions are ~1e-9 m, so compare against a physically meaningful scale
    assert np.max(np.abs(system.pos - pos_before)) > 1e-12


def test_add_interactions_switches_engine():
    # full functional coverage lives in tests/test_lj.py
    system = bc.box(N=10, size=(20, 20))
    assert not system._has_interactions
    system.add_interactions("LJ")
    assert system._has_interactions


def test_set_mass_by_value_all_particles():
    system = bc.box(N=20, size=(20, 20), gas="argon")
    system.set_mass(50.0)
    assert np.allclose(system.mass / bc.constants.AMU, 50.0)
    assert np.allclose(system.inv_mass, 1.0 / system.mass)


def test_set_mass_by_indices_creates_mixture():
    system = bc.box(N=20, size=(20, 20), gas="argon")
    original_amu = (system.mass / bc.constants.AMU).copy()
    system.set_mass(80.0, indices=slice(0, 5))
    new_amu = system.mass / bc.constants.AMU
    assert np.allclose(new_amu[:5], 80.0)
    assert np.allclose(new_amu[5:], original_amu[5:])


def test_set_mass_by_gas_name():
    system = bc.box(N=10, size=(20, 20), gas="argon")
    system.set_mass(gas="helium")
    assert np.isclose(system.mass[0] / bc.constants.AMU, 4.0026, rtol=1e-3)


def test_set_mass_requires_exactly_one_of_mass_or_gas():
    system = bc.box(N=10, size=(20, 20))
    with pytest.raises(ValueError):
        system.set_mass()
    with pytest.raises(ValueError):
        system.set_mass(mass=10.0, gas="helium")


def test_set_mass_rejects_nonpositive():
    system = bc.box(N=10, size=(20, 20))
    with pytest.raises(ValueError):
        system.set_mass(-5.0)
    with pytest.raises(ValueError):
        system.set_mass(0.0)
