"""Generate multiple sets of uniform Euler angles using Hopf fibration."""

import torch


def get_uniform_euler_angles(
    in_plane_step: float = 1.5,
    out_of_plane_step: float = 2.5,
    phi_ranges: torch.Tensor = None,
    theta_ranges: torch.Tensor = None,
    psi_ranges: torch.Tensor = None,
) -> list:
    """Generate sets of uniform Euler angles (ZYZ) using Hopf fibration.

    Args:
        in_plane_step: Angular step for in-plane rotation (phi, psi) in degrees.
        out_of_plane_step: Angular step for out-of-plane rotation (theta) in degrees.
        phi_ranges: Tensor of shape (n, 2) specifying ranges for phi in degrees.
        theta_ranges: Tensor of shape (n, 2) specifying ranges for theta in degrees.
        psi_ranges: Tensor of shape (n, 2) specifying ranges for psi in degrees.

    Returns
    -------
        list of torch.Tensor of shape (N, 3),
            where list length is the number of range sets,
            and N is the number of Euler angles in each set.
    """
    # Set default ranges if not provided
    if phi_ranges is None:
        phi_ranges = torch.tensor([[-180, 180]], dtype=torch.float64)
    if theta_ranges is None:
        theta_ranges = torch.tensor([[0, 180]], dtype=torch.float64)
    if psi_ranges is None:
        psi_ranges = torch.tensor([[-180, 180]], dtype=torch.float64)
    # Convert step sizes to radians
    in_plane_step = torch.deg2rad(torch.tensor(in_plane_step, dtype=torch.float64))
    out_of_plane_step = torch.deg2rad(
        torch.tensor(out_of_plane_step, dtype=torch.float64)
    )

    # Initialize the list to store results
    all_euler_angles = []

    # Iterate over the range sets
    for i in range(phi_ranges.shape[0]):
        # Extract and convert the current ranges to radians
        phi_range = torch.deg2rad(phi_ranges[i])
        theta_range = torch.deg2rad(theta_ranges[i])
        psi_range = torch.deg2rad(psi_ranges[i])

        # Calculate the number of samples for each angle
        n_theta = int(torch.ceil((theta_range[1] - theta_range[0]) / out_of_plane_step))
        n_psi = int(torch.ceil((psi_range[1] - psi_range[0]) / in_plane_step))

        # Generate theta values (out-of-plane angle)
        theta_all = torch.linspace(
            theta_range[0], theta_range[1], n_theta, dtype=torch.float64
        )
        psi_all = torch.linspace(psi_range[0], psi_range[1], n_psi, dtype=torch.float64)
        # Wrap theta values around to the range of 0 to pi
        theta_all = torch.where(theta_all < 0, theta_all * -1, theta_all)
        theta_all = torch.where(
            theta_all > torch.pi, 2 * torch.pi - theta_all, theta_all
        )
        # Wrap psi values around to the range of -pi to pi
        psi_all = torch.where(psi_all < -torch.pi, psi_all + 2 * torch.pi, psi_all)
        psi_all = torch.where(psi_all > torch.pi, psi_all - 2 * torch.pi, psi_all)

        # Generate phi values using Hopf fibration
        phi_max_step = phi_range[1] - phi_range[0]
        euler_angles = []
        phi_step_all = torch.clamp(
            torch.abs(out_of_plane_step / torch.sin(theta_all)), max=phi_max_step
        )
        phi_step_all = phi_max_step / torch.round(phi_max_step / phi_step_all)

        for j, phi_step in enumerate(phi_step_all):
            this_theta = theta_all[j]
            phi_array = torch.arange(
                phi_range[0], phi_range[1], phi_step, dtype=torch.float64
            )  # eps effectively makes endpoint inclusive
            # Wrap phi values around to the range of -pi to pi
            phi_array = torch.where(
                phi_array < -torch.pi, phi_array + 2 * torch.pi, phi_array
            )
            phi_array = torch.where(
                phi_array > torch.pi, phi_array - 2 * torch.pi, phi_array
            )
            # Create all combinations using meshgrid
            grid1, grid2, grid3 = torch.meshgrid(
                phi_array, this_theta, psi_all, indexing="ij"
            )
            # Stack and reshape to get the desired shape
            combinations = torch.stack([grid1, grid2, grid3], dim=-1).reshape(-1, 3)
            # Append combinations to the tensor
            euler_angles.append(combinations)

        # Concatenate all generated angles for the current set of ranges
        euler_angles = torch.cat(euler_angles, dim=0)
        euler_angles = torch.rad2deg(euler_angles)

        # Append to the list of all sets
        all_euler_angles.append(euler_angles)

    # Stack all sets to create a tensor of shape (n, N, 3)
    # all_euler_angles = torch.stack(all_euler_angles, dim=0)

    return all_euler_angles  # return list of tensors
