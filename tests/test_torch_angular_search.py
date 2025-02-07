"""Tests for `torch_angular_search` package."""

from torch_so3.hopf_angles import get_uniform_euler_angles
from torch_so3.refine_search import increased_resolution_grid

# TODO: Check actual values of returned tensors


def test_angle_generator():
    # Test the angle generator
    angles = get_uniform_euler_angles()
    assert angles.shape == (1584960, 3)


def test_increase_resolution():
    angles_increased = increased_resolution_grid()
    assert angles_increased.shape == (1372, 3)
