"""Functions to convert between rotation matrices and euler angles."""

import einops
import torch
from eulerangles import euler2matrix, matrix2euler


# Should ultimately convert euler2matrix to torch
def euler_to_rotation_matrix(
    euler_angles: torch.Tensor,
) -> torch.Tensor:
    """
    Convert a tensor of euler angles to a tensor of rotation matrices.

    Args:
        euler_angles: Tensor of euler angles, shape (n, 3)

    Returns
    -------
        Tensor of rotation matrices, shape (n, 3, 3)
    """
    rotation_matrices = []
    for euler_angle_tensor in euler_angles:
        rotation_matrices.append(
            torch.from_numpy(
                euler2matrix(
                    euler_angle_tensor.detach().numpy(),
                    axes="zyz",
                    intrinsic=True,
                    right_handed_rotation=False,
                )
            ).float()
        )
    return rotation_matrices


def rotation_matrix_to_euler(
    rotation_matrices: torch.Tensor,
) -> torch.Tensor:
    """
    Convert a tensor of rotation matrices to a tensor of euler angles.

    Args:
        rotation_matrices: Tensor of rotation matrices, shape (n, 3, 3)

    Returns
    -------
        Tensor of euler angles, shape (n, 3)
    """
    euler_angles = []
    for rotation_matrix in rotation_matrices:
        euler_angles.append(
            torch.from_numpy(
                matrix2euler(
                    rotation_matrix.detach().numpy(),
                    axes="zyz",
                    intrinsic=True,
                    right_handed_rotation=False,
                )
            ).float()
        )
    return euler_angles


def multiply_rotation_matrices(
    base_matrices: torch.Tensor,  # Shape (n, 3, 3)
    rotation_matrices: torch.Tensor,  # Shape (m, 3, 3)
) -> torch.Tensor:
    """
    Multiply each base matrix by each rotation matrix.

    Args:
        base_matrices: Tensor of n rotation matrices, shape (n, 3, 3)
        rotation_matrices: Tensor of m rotation matrices, shape (m, 3, 3)

    Returns
    -------
        Tensor of shape (m, n, 3, 3) containing all combinations of rotated matrices
    """
    # Rearrange tensors for broadcasting using einops
    base_matrices = einops.rearrange(
        base_matrices, "n h w -> 1 n h w"
    )  # Shape: (1, n, 3, 3)
    rotation_matrices = einops.rearrange(
        rotation_matrices, "m h w -> m 1 h w"
    )  # Shape: (m, 1, 3, 3)

    # Broadcast and perform matrix multiplication
    # Result shape will be (m, n, 3, 3)
    result = torch.matmul(rotation_matrices, base_matrices)

    return result
