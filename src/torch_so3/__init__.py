"""Generate uniform 3D euler angles (ZYZ)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-so3")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Josh Dickerson"
__email__ = "jdickerson@berkeley.edu"

from .local_so3_sampling import get_local_high_resolution_angles
from .uniform_so3_sampling import get_uniform_euler_angles

__all__ = ["get_uniform_euler_angles", "get_local_high_resolution_angles"]
