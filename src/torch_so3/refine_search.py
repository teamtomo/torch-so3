"""Generate finer angular search around multiple selected Euler angles."""

import torch

from torch_so3.uniform_so3_sampling import get_uniform_euler_angles

EPS = 1e-10


def increased_resolution_grid(
    coarse_in_plane_step: float = 1.5,
    coarse_out_of_plane_step: float = 2.5,
    fine_in_plane_step: float = 0.1,
    fine_out_of_plane_step: float = 0.1,
) -> torch.Tensor:
    """Local orientation refinement from a coarse to fine grid.

    Parameters
    ----------
    coarse_in_plane_step : float
        Coarse step size for in-plane angles (phi, psi) in degrees.
    coarse_out_of_plane_step : float
        Coarse step size for out-of-plane angle (theta) in degrees.
    fine_in_plane_step : float
        Finer step size for in-plane angles (phi, psi) in degrees.
    fine_out_of_plane_step : float
        Finer step size for out-of-plane angle (theta) in degrees.

    Returns
    -------
    euler_angles : torch.Tensor
        Tensor of shape (N, 3) containing Euler angles in degrees, where N is the
        number of angles generated. Angles exist around pole (0, 0, 0) and define grid
        to search over.
    """
    fine_theta_range = (
        -coarse_out_of_plane_step + fine_out_of_plane_step,
        coarse_out_of_plane_step - fine_out_of_plane_step,
    )
    fine_psi_range = (
        -coarse_in_plane_step + fine_in_plane_step,
        coarse_in_plane_step - fine_in_plane_step,
    )

    # Shouldn't need to run phi range for refined angles
    phi_range = (0, EPS)

    # Now get angles using Hopf fibration
    euler_angles = get_uniform_euler_angles(
        in_plane_step=fine_in_plane_step,
        out_of_plane_step=fine_out_of_plane_step,
        phi_min=phi_range[0],
        phi_max=phi_range[1],
        theta_min=fine_theta_range[0],
        theta_max=fine_theta_range[1],
        psi_min=fine_psi_range[0],
        psi_max=fine_psi_range[1],
    )

    return euler_angles
