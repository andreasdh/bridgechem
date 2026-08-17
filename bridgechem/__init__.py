"""bridgechem -- exploratory particle simulations for physical chemistry.

Bridging the microscopic, macroscopic and symbolic levels of chemistry with
small, interactive Jupyter-friendly simulations.

Quick start
-----------
    import bridgechem as bc

    system = bc.box(N=200, size=(60, 60))       # 60 nm x 60 nm box of argon
    system = bc.box(N=200, size=(20, 20, 20))   # or a 20 nm cube, in 3D

    sim = system.run(t=500)                     # 500 picoseconds
    sim.show()                                  # live animation
    sim.histogram("speeds")                     # vs Maxwell-Boltzmann
    print(sim.calculate("pressure"), sim.pressure_unit)   # N/m in 2D, Pa in 3D

The number of side lengths you pass decides the dimension. Everything else
follows: equipartition uses dim/2 k_B T, the speed distribution uses the 2D
or 3D Maxwell-Boltzmann form, and the pressure comes out in the right units.
"""

from __future__ import annotations

from . import analysis, constants
from .box import Box, box
from .simulation import Simulation
from .analysis import maxwell_boltzmann_speed, mean_speed, rms_speed

__version__ = "0.1.0"

__all__ = [
    "box",
    "Box",
    "Simulation",
    "analysis",
    "constants",
    "maxwell_boltzmann_speed",
    "mean_speed",
    "rms_speed",
    "__version__",
]
