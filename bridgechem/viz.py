"""Matplotlib-based visualisation for bridgechem simulations.

The default renderer replays a stored trajectory as an animation, which is far
more robust inside Jupyter than trying to draw while integrating. Circles are
drawn at their true physical size (in data units) so packing looks right.
"""

from __future__ import annotations

import numpy as np


def _nm(x):
    """Metres -> nanometres, for nicer axis labels."""
    return x * 1e9


def animate(traj_pos, radius, Lx, Ly, *, times=None, color_by=None,
            velocities=None, interval=40, figsize=(5.5, 5.5)):
    """Return a matplotlib animation of a trajectory.

    Parameters
    ----------
    traj_pos : (n_frames, N, 2) array, metres
    radius   : (N,) array, metres
    Lx, Ly   : box dimensions, metres
    color_by : None or "speed" -- colour particles by their instantaneous speed
    velocities : (n_frames, N, 2) array, required if ``color_by="speed"``
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import EllipseCollection

    traj_pos = np.asarray(traj_pos)
    n_frames = traj_pos.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, _nm(Lx))
    ax.set_ylim(0, _nm(Ly))
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")

    widths = _nm(2.0 * np.asarray(radius))

    if color_by == "speed":
        if velocities is None:
            raise ValueError("color_by='speed' requires velocities")
        spd = np.sqrt(np.sum(np.asarray(velocities) ** 2, axis=-1))
        vmax = float(spd.max()) if spd.size else 1.0
        colors0 = spd[0]
    else:
        colors0 = None

    coll = EllipseCollection(
        widths, widths, np.zeros_like(widths), units="xy",
        offsets=_nm(traj_pos[0]), offset_transform=ax.transData,
        facecolors="tab:blue" if colors0 is None else None,
        edgecolors="none",
    )
    if colors0 is not None:
        coll.set_array(colors0)
        coll.set_cmap("plasma")
        coll.set_clim(0, vmax)
        cbar = fig.colorbar(coll, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("speed (m/s)")
    ax.add_collection(coll)

    title = ax.set_title("")

    def update(frame):
        coll.set_offsets(_nm(traj_pos[frame]))
        if color_by == "speed":
            coll.set_array(spd[frame])
        if times is not None:
            title.set_text(f"t = {times[frame] * 1e12:.2f} ps")
        return (coll, title)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval,
                         blit=False)
    plt.close(fig)  # avoid double display in notebooks
    return anim


def histogram(speeds_array, *, temperature_K=None, mass_kg=None, dim=2,
              bins=40, ax=None, label="simulation"):
    """Plot a speed histogram, optionally overlaying Maxwell-Boltzmann."""
    import matplotlib.pyplot as plt

    from .analysis import maxwell_boltzmann_speed

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    speeds_array = np.asarray(speeds_array).ravel()
    ax.hist(speeds_array, bins=bins, density=True, alpha=0.6, label=label,
            color="tab:blue")
    if temperature_K is not None and mass_kg is not None:
        v = np.linspace(0, speeds_array.max() * 1.05, 400)
        mb = maxwell_boltzmann_speed(v, temperature_K, mass_kg, dim=dim)
        ax.plot(v, mb, "r-", lw=2,
                label=f"Maxwell-Boltzmann (T={temperature_K:.0f} K)")
    ax.set_xlabel("speed (m/s)")
    ax.set_ylabel("probability density")
    ax.legend()
    return ax
