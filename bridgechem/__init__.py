"""bridgechem -- exploratory particle simulations for physical chemistry.

Bridging the microscopic, macroscopic and symbolic levels of chemistry with
small, interactive Jupyter-friendly simulations.

Quick start
-----------
    import bridgechem as bc

    system = bc.box(N=1000, size=(60, 60))   # 60 nm x 60 nm box of argon
    sim = system.run(steps=20000)
    sim.show()                               # live animation
    sim.histogram("speeds")                  # vs Maxwell-Boltzmann
    print(sim.calculate("pressure"))         # 2D pressure (N/m)
"""

from __future__ import annotations

from . import analysis, constants
from .box import Box, box
from .simulation import Simulation
from .analysis import maxwell_boltzmann_speed, mean_speed

__version__ = "0.1.0"

__all__ = [
    "box",
    "Box",
    "Simulation",
    "analysis",
    "constants",
    "maxwell_boltzmann_speed",
    "mean_speed",
    "__version__",
]
