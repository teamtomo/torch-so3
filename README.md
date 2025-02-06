# torch-angular-search

[![License](https://img.shields.io/pypi/l/torch-angular-search.svg?color=green)](https://github.com/jdickerson95/torch-angular-search/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-angular-search.svg?color=green)](https://pypi.org/project/torch-angular-search)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-angular-search.svg?color=green)](https://python.org)
[![CI](https://github.com/jdickerson95/torch-angular-search/actions/workflows/ci.yml/badge.svg)](https://github.com/jdickerson95/torch-angular-search/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jdickerson95/torch-angular-search/branch/main/graph/badge.svg)](https://codecov.io/gh/jdickerson95/torch-angular-search)

Generate uniform 3D euler angles (ZYZ)

## Installation

Install via pip
```zsh
pip install torch-angular-search
```

Install via source by first cloning the repository then running.
```zsh
git clone https://github.com/jdickerson95/torch-angular-search.git
cd torch-angular-search
pip install -e .
```
And for development and testing use
```zsh
pip install -e ".[dev,test]"
```

For those contributing make sure to run tests before, and to adhere to the pre-commit hooks.
```zsh
python -m pytest
pre-commit run
```

## Usage

A basic example of generating uniform Euler angles in 4.0 and 6.0 degree increments across the entire SO(3) group is shown below.

```python
from torch_angular_search.hopf_angles import get_uniform_euler_angles

angles = get_uniform_euler_angles(
    in_plane_step=4.0,  # units of degrees
    out_of_plane_step=6.0,
)
angles.shape  # (103500, 3)
```
