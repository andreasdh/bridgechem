"""Milestone 2: Lennard-Jones interactions, velocity-Verlet, thermostat.

Energy conservation is the gold-standard sanity check for MD -- LJ dynamics
is notoriously easy to get subtly wrong (a bad dt, a sign error in the force,
a missing potential shift), and small drift shows up immediately as growing
total energy. These tests were used to calibrate the automatic time step
(see Box._auto_dt) across a wide temperature range.
"""

import numpy as np
import pytest

import bridgechem as bc


def _reduced_density_box(N=200, rho_star=0.5, sigma_nm=0.34, **kwargs):
    """A box sized so N*sigma^2/Area == rho_star, for reproducible LJ tests."""
    area_nm2 = N * sigma_nm ** 2 / rho_star
    L_nm = float(np.sqrt(area_nm2))
    return bc.box(N=N, size=(L_nm, L_nm), **kwargs)


def test_add_interactions_defaults_from_gas():
    system = bc.box(N=10, size=(10, 10), gas="argon")
    system.add_interactions("LJ")
    assert system._has_interactions
    assert np.isclose(system._interaction["epsilon"] / bc.constants.K_B, 120.0)
    assert np.isclose(system._interaction["sigma"], 0.340e-9)


def test_add_interactions_dispersion_is_lj_alias():
    system = bc.box(N=10, size=(10, 10))
    system.add_interactions("dispersion")
    assert system._interaction["kind"] == "LJ"


def test_add_interactions_custom_epsilon_sigma():
    system = bc.box(N=10, size=(10, 10))
    system.add_interactions("LJ", epsilon=50.0, sigma=0.3)
    assert np.isclose(system._interaction["epsilon"] / bc.constants.K_B, 50.0)
    assert np.isclose(system._interaction["sigma"], 0.3e-9)


def test_add_interactions_invalid_raises():
    system = bc.box(N=10, size=(10, 10))
    with pytest.raises(ValueError):
        system.add_interactions("nonsense")


def test_add_interactions_nonpositive_params_raise():
    system = bc.box(N=10, size=(10, 10))
    with pytest.raises(ValueError):
        system.add_interactions("LJ", epsilon=-1.0)
    with pytest.raises(ValueError):
        system.add_interactions("LJ", sigma=0.0)


def test_run_requires_velocity_verlet_once_interacting():
    system = bc.box(N=10, size=(10, 10))
    system.add_interactions("LJ")
    with pytest.raises(ValueError):
        system.run(steps=100, method="hard-sphere", animate=False)
    sim = system.run(steps=100, method="velocity-verlet", animate=False)
    assert isinstance(sim, bc.Simulation)


@pytest.mark.parametrize("temperature", [30, 300, 1500])
def test_lj_energy_conserved_across_temperatures(temperature):
    # dt is chosen automatically; must stay stable and conservative from
    # cold (tight, curvature-limited) to hot (fast, speed-limited) regimes.
    system = _reduced_density_box(N=150, rho_star=0.5, temperature=temperature,
                                  boundary="periodic", seed=0)
    system.add_interactions("LJ")
    sim = system.run(steps=8000, animate=False)
    E = sim.calculate("total_energy")
    assert (E.max() - E.min()) / abs(E.mean()) < 0.01


def test_lj_energy_conserved_reflective_walls():
    system = _reduced_density_box(N=150, rho_star=0.3, temperature=200,
                                  boundary="reflective", seed=0)
    system.add_interactions("LJ")
    sim = system.run(steps=10000, animate=False)
    E = sim.calculate("total_energy")
    assert (E.max() - E.min()) / abs(E.mean()) < 0.01


def test_lj_potential_energy_nonzero_when_dense():
    system = _reduced_density_box(N=150, rho_star=0.5, temperature=100,
                                  boundary="periodic", seed=0)
    system.add_interactions("LJ")
    sim = system.run(steps=2000, animate=False)
    assert np.any(sim.calculate("potential_energy") < 0.0)


def test_lj_wall_and_virial_pressure_agree():
    system = _reduced_density_box(N=150, rho_star=0.3, temperature=200,
                                  boundary="reflective", seed=0)
    system.add_interactions("LJ")
    sim = system.run(steps=15000, animate=False)
    p_wall = sim.calculate("pressure", method="wall")
    p_virial = sim.calculate("pressure", method="virial")
    assert 0.8 < p_virial / p_wall < 1.2


def test_momentum_conserved_with_interactions_periodic():
    system = _reduced_density_box(N=150, rho_star=0.4, temperature=200,
                                  boundary="periodic", seed=0)
    system.add_interactions("LJ")
    p0 = (system.mass[:, None] * system.vel).sum(axis=0)
    sim = system.run(steps=5000, animate=False)
    pf = (sim.mass[:, None] * sim.vel[-1]).sum(axis=0)
    scale = system.mass[0] * bc.analysis.speeds(system.vel).mean() * system.N
    assert np.linalg.norm(pf - p0) / scale < 1e-6


# -- thermostat / cooling ramp ------------------------------------------------
def test_set_temperature_instant_jump():
    system = bc.box(N=50, size=(10, 10), temperature=300, seed=0)
    system.set_temperature(150)  # no rate -> jump immediately
    sim = system.run(steps=2000, animate=False)
    T = sim.calculate("temperature")
    assert np.isclose(T[0], 300.0, rtol=1e-6)  # pre-thermostat state
    assert np.allclose(T[1:], 150.0, rtol=0.05)


def test_set_temperature_ramp_reaches_target():
    system = bc.box(N=50, size=(10, 10), temperature=300, seed=0)
    system.set_temperature(100, rate=500)  # K/ps
    sim = system.run(steps=10000, animate=False)
    T = sim.calculate("temperature")
    assert T[0] > T[-1]  # cooled down
    assert np.isclose(T[-1], 100.0, rtol=0.05)


def test_set_temperature_applies_only_to_next_run():
    system = bc.box(N=50, size=(10, 10), temperature=300, seed=0)
    system.set_temperature(100)
    system.run(steps=500, animate=False)
    assert system._thermostat is None  # consumed
    sim2 = system.run(steps=500, animate=False)  # no thermostat this time
    T2 = sim2.calculate("temperature")
    assert np.isclose(T2[0], T2[-1], rtol=0.2)  # left alone, no forced ramp


def test_set_temperature_rejects_nonpositive():
    system = bc.box(N=10, size=(10, 10))
    with pytest.raises(ValueError):
        system.set_temperature(-10)
    with pytest.raises(ValueError):
        system.set_temperature(0)


def test_cooling_lj_gas_increases_binding():
    # a real phase-transition-flavoured check: cooling a periodic LJ gas
    # should make the configuration more tightly bound (more negative PE per
    # particle) as it condenses -- not just "temperature went down".
    system = _reduced_density_box(N=150, rho_star=0.4, temperature=400,
                                  boundary="periodic", seed=0)
    system.add_interactions("LJ")
    pe_hot = np.mean(bc.kernels._lj_forces(
        system.pos, system.Lx, system.Ly, system.periodic,
        system._interaction["epsilon"], system._interaction["sigma"] ** 2,
        system._interaction["r_cut2"], system._interaction["u_shift"],
        np.zeros((system.N, 2)),
    )[0]) / system.N

    system.set_temperature(20, rate=100)
    sim = system.run(steps=30000, animate=False)
    pe_cold = sim.calculate("potential_energy")[-1] / system.N
    assert pe_cold < pe_hot  # more negative = more bound
