"""Tests for `torch_angular_search` package."""

import platform

import pytest

from torch_so3.refine_search import increased_resolution_grid
from torch_so3.uniform_so3_sampling import get_uniform_euler_angles

# TODO: Check actual values of returned tensors


def test_get_uniform_euler_angles():
    # Test the angle generator
    angles = get_uniform_euler_angles(base_grid_method="uniform")
    assert angles.shape == (1584480, 3)


@pytest.mark.skipif(
    platform.system() == "Windows", reason="healpy is not supported on Windows"
)
def test_get_uniform_euler_angles_healpix():
    angles = get_uniform_euler_angles(base_grid_method="healpix")
    assert angles.shape == (1658880, 3)


def test_increase_resolution():
    angles_increased = increased_resolution_grid()
    assert angles_increased.shape == (1372, 3)
