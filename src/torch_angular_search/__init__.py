"""Generate unfirform 3D euler angles (ZYZ)"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-angular-search")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Josh Dickerson"
__email__ = "jdickerson@berkeley.edu"
