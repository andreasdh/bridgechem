"""Physical constants (SI) and convenience unit conversions.

bridgechem works internally in **SI base units** (metre, second, kilogram,
kelvin, joule). For ergonomics the user-facing API accepts a couple of
chemistry-friendly input units that are converted to SI immediately:

* lengths (box ``size``, particle ``radius``) are given in **nanometres**,
* masses are given in **atomic mass units (amu)**.

Everything that comes *out* of a simulation (speeds, temperatures, energies,
pressures) is in SI.
"""

from __future__ import annotations

# --- Fundamental constants (SI, CODATA) -------------------------------------
K_B = 1.380649e-23      # Boltzmann constant, J/K
N_A = 6.02214076e23     # Avogadro constant, 1/mol
AMU = 1.66053906660e-27  # atomic mass unit, kg

# --- Input-unit conversions -------------------------------------------------
NM = 1e-9  # nanometre in metres

# --- Reference gases --------------------------------------------------------
# Radii are effective hard-sphere radii (~ sigma_LJ / 2) in metres; masses in kg.
# epsilon_over_kB (K) and sigma (nm) are standard tabulated Lennard-Jones
# parameters (e.g. Hirschfelder, Curtiss & Bird), used once interactions are
# switched on via Box.add_interactions("LJ"). These are didactic ball-park
# values, good enough to reproduce sensible behaviour for dilute noble gases.
GASES = {
    "argon":   {"mass_amu": 39.948, "radius_nm": 0.171, "epsilon_over_kB": 120.0, "sigma_nm": 0.340},
    "helium":  {"mass_amu": 4.0026, "radius_nm": 0.140, "epsilon_over_kB": 10.2, "sigma_nm": 0.258},
    "neon":    {"mass_amu": 20.180, "radius_nm": 0.154, "epsilon_over_kB": 36.2, "sigma_nm": 0.275},
    "krypton": {"mass_amu": 83.798, "radius_nm": 0.202, "epsilon_over_kB": 171.0, "sigma_nm": 0.360},
    "xenon":   {"mass_amu": 131.29, "radius_nm": 0.216, "epsilon_over_kB": 221.0, "sigma_nm": 0.410},
}

DEFAULT_GAS = "argon"


def gas_properties(name: str) -> dict:
    """Return ``{"mass_kg", "radius_m", "epsilon_J", "sigma_m"}`` for a named gas."""
    key = name.lower()
    if key not in GASES:
        raise ValueError(
            f"Unknown gas {name!r}. Available: {', '.join(sorted(GASES))}."
        )
    g = GASES[key]
    return {
        "mass_kg": g["mass_amu"] * AMU,
        "radius_m": g["radius_nm"] * NM,
        "epsilon_J": g["epsilon_over_kB"] * K_B,
        "sigma_m": g["sigma_nm"] * NM,
    }
