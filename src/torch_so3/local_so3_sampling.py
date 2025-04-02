"""Generate finer angular search around multiple selected Euler angles."""

from typing import Literal

import torch

from torch_so3.uniform_so3_sampling import get_uniform_euler_angles

EPS = 1e-6


def get_local_high_resolution_angles(
    coarse_psi_step: float = 1.5,
    coarse_theta_step: float = 2.5,
    coarse_phi_step: float = 2.5,
    fine_psi_step: float = 0.1,
    fine_theta_step: float = 0.1,
    fine_phi_step: float = 0.1,
    base_grid_method: Literal["uniform", "healpix", "cartesian"] = "uniform",
) -> torch.Tensor:
    """Local orientation refinement from a coarse to fine grid.

    This function is essentially creating a high-resolution grid of Euler angles around
    the pole to sample all points within the coarse grid. It creates a "disk" of angles
    for the S^2 (theta, phi) sphere and also samples the in-plane angle (phi) uniformly.

    Parameters
    ----------
    coarse_phi_step : float
        Coarse step size for phi in degrees.
    coarse_theta_step : float
        Coarse step size for theta in degrees.
    coarse_psi_step : float
        Coarse step size for psi in degrees.
    fine_psi_step : float
        Finer step size for psi in degrees.
    fine_theta_step : float
        Finer step size for theta in degrees.
    fine_phi_step : float
        Finer step size for phi in degrees.
    base_grid_method : Literal["uniform", "healpix", "cartesian"]
        Method to generate the base grid.

    Returns
    -------
    euler_angles : torch.Tensor
        Tensor of shape (N, 3) containing Euler angles in degrees, where N is the
        number of angles generated. Angles exist around pole (0, 0, 0) and define grid
        to search over.
    """
    if base_grid_method == "cartesian":
        euler_angles = get_uniform_euler_angles(
            phi_step=fine_phi_step,
            theta_step=fine_theta_step,
            psi_step=fine_psi_step,
            phi_min=-coarse_phi_step,  # Completely sample for uniform base grid?
            phi_max=coarse_phi_step,
            theta_min=-coarse_theta_step,
            theta_max=coarse_theta_step,
            psi_min=-coarse_psi_step,
            psi_max=coarse_psi_step + EPS,
            base_grid_method=base_grid_method,
        )
    else:
        euler_angles = get_uniform_euler_angles(
            psi_step=fine_psi_step,
            theta_step=fine_theta_step,
            phi_min=-coarse_psi_step,  # Completely sample for uniform base grid?
            phi_max=coarse_psi_step,
            theta_min=-coarse_theta_step,
            theta_max=coarse_theta_step,
            psi_min=-coarse_psi_step,
            psi_max=coarse_psi_step + EPS,
            base_grid_method=base_grid_method,
        )

    return euler_angles
