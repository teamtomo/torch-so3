"""Generates a set of Euler angles that uniformly samples SO(3) using Hopf fibration."""

from typing import Literal

import torch

from torch_so3.base_s2_grid import healpix_base_grid, uniform_base_grid


def get_uniform_euler_angles(
    in_plane_step: float = 1.5,
    out_of_plane_step: float = 2.5,
    phi_min: float = 0.0,
    phi_max: float = 360.0,
    theta_min: float = 0.0,
    theta_max: float = 180.0,
    psi_min: float = 0.0,
    psi_max: float = 360.0,
    base_grid_method: Literal["uniform", "healpix"] = "uniform",
) -> torch.Tensor:
    """Generate sets of uniform Euler angles (ZYZ) using Hopf fibration.

    Parameters
    ----------
    in_plane_step: float, optional
        Angular step for in-plane rotation (phi) in degrees. Default is 1.5 degrees.
    out_of_plane_step: float, optional
        Angular step for out-of-plane rotation (theta) in degrees. Default is 2.5
        degrees.
    phi_min: float, optional
        Minimum value for phi in degrees. Default is 0.0.
    phi_max: float, optional
        Maximum value for phi in degrees. Default is 360.0.
    theta_min: float, optional
        Minimum value for theta in degrees. Default is 0.0.
    theta_max: float, optional
        Maximum value for theta in degrees. Default is 180.0.
    psi_min: float, optional
        Minimum value for psi in degrees. Default is 0.0.
    psi_max: float, optional
        Maximum value for psi in degrees. Default is 360.0.
    base_grid_method: str, optional
        String literal specifying the method to generate the base grid. Default is
        "uniform". Options are "uniform" and "healpix".

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 3) containing Euler angles in degrees, where N is the
        number of angles generated.
    """
    # TODO: Validation of inputs, wrapping between zero and 2*pi, etc.

    # Choose the method to generate the base grid from
    if base_grid_method == "uniform":
        base_grid_mth = uniform_base_grid
    elif base_grid_method == "healpix":
        base_grid_mth = healpix_base_grid
    else:
        raise ValueError(f"Invalid base grid method {base_grid_method}.")

    base_grid = base_grid_mth(
        out_of_plane_step=out_of_plane_step,
        theta_min=theta_min,
        theta_max=theta_max,
        phi_min=phi_min,
        phi_max=phi_max,
    )

    # Mesh-grid-like operation to include the in-plane rotation
    psi_all = torch.arange(psi_min, psi_max, in_plane_step, dtype=torch.float64)

    psi_mesh = psi_all.repeat_interleave(base_grid.size(0))
    base_grid = base_grid.repeat(psi_all.size(0), 1)

    all_angles = torch.cat([base_grid, psi_mesh[:, None]], dim=1)

    return all_angles
