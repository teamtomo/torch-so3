"""Generate multiple sets of uniform Euler angles using Hopf fibration."""

import numpy as np
import torch


def get_uniform_euler_angles(
    in_plane_step: float = 1.5,
    out_of_plane_step: float = 2.5,
    phi_min: float = 0.0,
    phi_max: float = 360.0,
    theta_min: float = 0.0,
    theta_max: float = 180.0,
    psi_min: float = 0.0,
    psi_max: float = 360.0,
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

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 3) containing Euler angles in degrees, where N is the
        number of angles generated.
    """
    # TODO: Validation of inputs, wrapping between zero and 2*pi, etc.

    # Convert input angles from degrees into radians
    in_plane_step_rad = np.deg2rad(in_plane_step)
    out_of_plane_step_rad = np.deg2rad(out_of_plane_step)
    phi_min_rad = np.deg2rad(phi_min)
    phi_max_rad = np.deg2rad(phi_max)
    theta_min_rad = np.deg2rad(theta_min)
    theta_max_rad = np.deg2rad(theta_max)
    psi_min_rad = np.deg2rad(psi_min)
    psi_max_rad = np.deg2rad(psi_max)

    # Initialize the list to store results
    all_euler_angles = []

    # The grid of theta and psi values to search over
    # NOTE: Including the 180.0 degree entry in theta_all since projection of
    # non-symmetric object (e.g. ribosome) is different at 0.0 and 180.0 degrees
    theta_all = torch.arange(
        theta_min_rad,
        theta_max_rad + out_of_plane_step_rad,
        step=out_of_plane_step_rad,
        dtype=torch.float64,
    )
    psi_all = torch.arange(
        psi_min_rad,
        psi_max_rad,
        step=in_plane_step_rad,
        dtype=torch.float64,
    )

    # Phi step increment is modulated by the position on the sphere (sin(theta)), but
    # don't allow it to exceed the maximum step size
    phi_max_step = phi_max_rad - phi_min_rad
    phi_step_all = torch.clamp(
        torch.abs(out_of_plane_step_rad / torch.sin(theta_all)), max=phi_max_step
    )
    phi_step_all = phi_max_step / torch.round(phi_max_step / phi_step_all)

    for j, phi_step in enumerate(phi_step_all):
        phi_array = torch.arange(
            phi_min_rad, phi_max_rad, phi_step, dtype=torch.float64
        )

        grid_phi, grid_theta, grid_psi = torch.meshgrid(
            phi_array,
            theta_all[j],  # indexing only a single value from theta_all
            psi_all,
            indexing="ij",
        )
        euler_angles = torch.stack([grid_phi, grid_theta, grid_psi], dim=-1)
        all_euler_angles.append(euler_angles.reshape(-1, 3))

    all_euler_angles = torch.cat(all_euler_angles, dim=0)
    all_euler_angles = torch.rad2deg(all_euler_angles)

    return all_euler_angles
