"""The same physics, run in 2D and in 3D.

Every test here is parametrised over the dimension. That is the point: a
dimension bug in the kernels (summing two velocity components while dividing
by three degrees of freedom, say) passes a 2D-only suite silently, and shows
up here as a factor of dim/2 in the temperature.
"""

import numpy as np
import pytest

import bridgechem as bc
from bridgechem.constants import K_B


SIZES = {2: (40.0, 40.0), 3: (12.0, 12.0, 12.0)}


def dilute_box(dim, N=120, temperature=300.0, **kwargs):
    """A box dilute enough to behave as an ideal gas (packing well under 1%)."""
    kwargs.setdefault("radius", 0.08)
    kwargs.setdefault("seed", 0)
    return bc.box(N=N, size=SIZES[dim], temperature=temperature, **kwargs)


# --- the interface ---------------------------------------------------------
def test_dimension_comes_from_the_number_of_side_lengths():
    assert bc.box(N=10, size=(20, 20)).dim == 2
    assert bc.box(N=10, size=(20, 20, 20)).dim == 3


def test_bad_number_of_side_lengths_is_rejected():
    with pytest.raises(ValueError, match="2 entries"):
        bc.box(N=10, size=(20,))
    with pytest.raises(ValueError, match="2 entries"):
        bc.box(N=10, size=(20, 20, 20, 20))


def test_area_is_2d_only_but_volume_always_works():
    flat, cube = bc.box(N=10, size=(20, 20)), bc.box(N=10, size=(20, 20, 20))
    assert np.isclose(flat.area, flat.volume)
    assert np.isclose(cube.volume, (20e-9) ** 3)
    with pytest.raises(AttributeError, match="volume"):
        cube.area


def test_pressure_unit_follows_the_dimension():
    assert dilute_box(2).run(t=20, animate=False).pressure_unit == "N/m"
    assert dilute_box(3).run(t=20, animate=False).pressure_unit == "Pa"


# --- equipartition and Maxwell-Boltzmann -----------------------------------
@pytest.mark.parametrize("dim", [2, 3])
def test_kinetic_energy_per_particle_is_half_dim_kT(dim):
    """<KE> per particle = (dim/2) k_B T. The trap-catcher."""
    sim = dilute_box(dim, temperature=250.0).run(t=100, animate=False)
    ke = np.mean(sim.calculate("kinetic_energy")) / sim.n_particles
    T = float(np.mean(sim.calculate("temperature")))
    assert np.isclose(ke, 0.5 * dim * K_B * T, rtol=1e-10)
    assert np.isclose(T, 250.0, rtol=0.05)


@pytest.mark.parametrize("dim", [2, 3])
def test_rms_speed_matches_the_analytic_value(dim):
    T, mass_kg = 300.0, bc.constants.gas_properties("argon")["mass_kg"]
    sim = dilute_box(dim, temperature=T).run(t=100, animate=False)
    v_rms = float(np.sqrt(np.mean(sim.calculate("speeds") ** 2)))
    assert np.isclose(v_rms, bc.rms_speed(T, mass_kg, dim=dim), rtol=0.02)


@pytest.mark.parametrize("dim", [2, 3])
def test_mean_speed_matches_the_analytic_value(dim):
    T, mass_kg = 300.0, bc.constants.gas_properties("argon")["mass_kg"]
    sim = dilute_box(dim, temperature=T).run(t=100, animate=False)
    v_mean = float(np.mean(sim.calculate("speeds")))
    assert np.isclose(v_mean, bc.mean_speed(T, mass_kg, dim=dim), rtol=0.02)


@pytest.mark.parametrize("dim", [2, 3])
def test_uniform_speed_relaxes_towards_maxwell_boltzmann(dim):
    """Everyone starts at the same speed; collisions spread them out."""
    system = dilute_box(dim, N=200, radius=0.4, velocity_init="uniform_speed")
    spread_before = float(np.std(bc.analysis.speeds(system.vel)))
    sim = system.run(t=300, animate=False)
    spread_after = float(np.std(sim.calculate("speeds")[-1]))
    assert spread_before < 1e-6          # all identical to begin with
    assert spread_after > 0.2 * float(np.mean(sim.calculate("speeds")[-1]))


# --- pressure --------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 3])
def test_dilute_gas_obeys_the_ideal_gas_law(dim):
    sim = dilute_box(dim).run(t=2000, animate=False)
    assert np.isclose(sim.pressure("virial") / sim.ideal_gas_pressure(),
                      1.0, atol=0.05)


@pytest.mark.parametrize("dim", [2, 3])
def test_wall_and_virial_agree_for_a_dilute_reflective_box(dim):
    sim = dilute_box(dim).run(t=2000, animate=False)
    assert np.isclose(sim.pressure("wall") / sim.pressure("virial"),
                      1.0, atol=0.08)


@pytest.mark.parametrize("dim", [2, 3])
def test_finite_size_raises_the_pressure_above_ideal(dim):
    """Excluded volume: bigger particles push harder. The b of van der Waals."""
    dilute = dilute_box(dim, radius=0.08).run(t=1000, animate=False)
    dense = dilute_box(dim, radius=0.5).run(t=1000, animate=False)
    assert (dense.pressure("virial") / dense.ideal_gas_pressure()
            > dilute.pressure("virial") / dilute.ideal_gas_pressure() + 0.05)


@pytest.mark.parametrize("dim", [2, 3])
def test_per_frame_pressure_averages_to_the_whole_run_value(dim):
    sim = dilute_box(dim).run(t=1000, animate=False)
    series = sim.pressure("wall", per_frame=True)
    assert series.shape == (sim.n_frames,)
    assert np.isclose(np.mean(series[1:]), sim.pressure("wall"), rtol=0.05)


@pytest.mark.parametrize("dim", [2, 3])
def test_wall_collision_record_reproduces_the_wall_pressure(dim):
    """A student can do the sum by hand and get the same number back."""
    sim = dilute_box(dim).run(t=1000, animate=False)
    impulse = sim.wall_collisions()                     # (dim,), kg m/s
    wall_extent = sim.volume / sim.L                    # area (3D) or length (2D)
    by_hand = float(np.mean(impulse / (sim.total_time * 2.0 * wall_extent)))
    assert np.isclose(by_hand, sim.pressure("wall"), rtol=1e-10)


# --- conservation and containment ------------------------------------------
@pytest.mark.parametrize("dim", [2, 3])
def test_hard_spheres_conserve_energy(dim):
    sim = dilute_box(dim, radius=0.4).run(t=500, animate=False)
    ke = sim.calculate("kinetic_energy")
    assert np.allclose(ke, ke[0], rtol=1e-9)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("boundary", ["reflective", "periodic"])
def test_particles_stay_inside_the_box(dim, boundary):
    system = dilute_box(dim, radius=0.4, boundary=boundary)
    sim = system.run(t=500, animate=False)
    for d in range(dim):
        assert sim.pos[..., d].min() >= -1e-18
        assert sim.pos[..., d].max() <= system.L[d] + 1e-18


@pytest.mark.parametrize("dim", [2, 3])
def test_lennard_jones_runs_and_roughly_conserves_energy(dim):
    system = bc.box(N=80, size=SIZES[dim], temperature=200.0,
                    boundary="periodic", seed=0)
    system.add_interactions("LJ")
    sim = system.run(t=20, animate=False)
    total = sim.calculate("total_energy")
    drift = abs(total[-1] - total[0]) / abs(total[0])
    assert drift < 0.02


def _dense_lj_box(dim, N=80, rho_star=0.4, **kwargs):
    """A periodic LJ box at a given reduced density N sigma^dim / V."""
    sigma_nm = bc.constants.GASES["argon"]["sigma_nm"]
    side = (N * sigma_nm ** dim / rho_star) ** (1.0 / dim)
    return bc.box(N=N, size=(side,) * dim, boundary="periodic", **kwargs)


@pytest.mark.parametrize("dim", [2, 3])
def test_cooling_an_interacting_gas_binds_it(dim):
    system = _dense_lj_box(dim, temperature=300.0, seed=0)
    system.add_interactions("LJ")
    warm = system.run(t=10, animate=False).calculate("potential_energy")[-1]
    system.set_temperature(25, rate=50)
    cold = system.run(t=40, animate=False).calculate("potential_energy")[-1]
    assert cold < warm  # more negative = more bound


# --- the physical sanity check ---------------------------------------------
def test_a_realistic_3d_argon_box_reads_a_recognisable_pressure():
    """N particles of argon at 300 K in a box of this size should read the
    pressure the ideal gas law predicts, in pascal, to within a few percent."""
    N, side_nm, T = 150, 18.4, 300.0
    sim = bc.box(N=N, size=(side_nm,) * 3, temperature=T, radius=0.08,
                 seed=0).run(t=4000, animate=False)
    expected_Pa = N * K_B * T / (side_nm * 1e-9) ** 3
    assert np.isclose(sim.pressure("virial"), expected_Pa, rtol=0.05)
    # 150 argon atoms in an 18.4 nm cube at room temperature sit at
    # very nearly one atmosphere -- a number a student can recognise.
    assert np.isclose(expected_Pa, 1.013e5, rtol=0.05)
