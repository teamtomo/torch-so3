"""Tests for `torch_angular_search` package."""

from torch_angular_search.hopf_angles import get_uniform_euler_angles
from torch_angular_search.refine_search import increase_resolution


def test_angle_generator():
    # Test the angle generator
    angles = get_uniform_euler_angles()
    assert angles[0].shape[0] == 1562400


def test_increase_resolution():
    angles_increased = increase_resolution()
    print(f"shape pole increase res: {angles_increased[0].shape}")
    assert angles_increased[0].shape[0] == 1344


"""
#deprecated
def test_refine_angles():
    best_angles = torch.tensor([[87, 87, 87], [45, 45, 45]])
    angles_refined = refine_euler_angles(best_angles)
    assert angles_refined[0].shape[0] == 37632
    assert angles_refined[1].shape[0] == 39168
"""
