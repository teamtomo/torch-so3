"""Generate Euler angles ranges applying symmetry operations."""

import torch


def get_symmetry_ranges(
    symmetry_group: str = "C",
    symmetry_order: int = 1,
) -> torch.Tensor:
    """
    Generate Euler angle ranges based on symmetry group.

    Args:
        symmetry_group: Symmetry group of the particle (C, D, T, O, I).
        symmetry_order: Order of the symmetry group.

    Returns
    -------
        torch.Tensor: Tensor of shape (3, 2)
            containing the ranges for phi, theta, and psi.
    """
    # Convert to upper case
    symmetry_group = symmetry_group.upper()
    phi_max = 180.0
    theta_max = 180.0
    psi_max = 180.0
    if symmetry_group == "C":
        phi_max = 180 / float(symmetry_order)
    elif symmetry_group == "D":
        phi_max = 180 / float(symmetry_order)
        theta_max = 90.0
    elif symmetry_group == "T":
        phi_max = 90.0
        theta_max = 54.7356
    elif symmetry_group == "O":
        phi_max = 45.0
        theta_max = 54.7356
    elif symmetry_group == "I":
        phi_max = 90.0
        theta_max = 31.7
    else:
        raise ValueError("Symmetry group not recognized")

    return torch.tensor(
        [[-phi_max, phi_max], [0, theta_max], [-psi_max, psi_max]], dtype=torch.float64
    )
