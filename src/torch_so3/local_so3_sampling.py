"""Generate finer angular search around multiple selected Euler angles."""

import torch

from torch_so3.uniform_so3_sampling import get_uniform_euler_angles

EPS = 1e-6


def get_local_high_resolution_angles(
    coarse_in_plane_step: float = 1.5,
    coarse_out_of_plane_step: float = 2.5,
    fine_in_plane_step: float = 0.1,
    fine_out_of_plane_step: float = 0.1,
) -> torch.Tensor:
    """Local orientation refinement from a coarse to fine grid.

    This function is essentially creating a high-resolution grid of Euler angles around
    the pole to sample all points within the coarse grid. It creates a "disk" of angles
    for the S^2 (theta, phi) sphere and also samples the in-plane angle (phi) uniformly.

    Parameters
    ----------
    coarse_in_plane_step : float
        Coarse step size for in-plane angle (psi) in degrees.
    coarse_out_of_plane_step : float
        Coarse step size for out-of-plane angle (theta, phi) in degrees.
    fine_in_plane_step : float
        Finer step size for in-plane angles (psi) in degrees.
    fine_out_of_plane_step : float
        Finer step size for out-of-plane angle (theta, phi) in degrees.

    Returns
    -------
    euler_angles : torch.Tensor
        Tensor of shape (N, 3) containing Euler angles in degrees, where N is the
        number of angles generated. Angles exist around pole (0, 0, 0) and define grid
        to search over.
    """
    euler_angles = get_uniform_euler_angles(
        in_plane_step=fine_in_plane_step,
        out_of_plane_step=fine_out_of_plane_step,
        psi_min=-coarse_in_plane_step,
        psi_max=coarse_in_plane_step + EPS,
        theta_min=0.0,
        theta_max=coarse_out_of_plane_step,
        phi_min=0.0,  # Completely sample around the s2 sphere
        phi_max=360.0,
    )

    return euler_angles
