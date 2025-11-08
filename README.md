<h1 align="center">caput</h1>

<div align="center">

[![Zenodo Badge](https://zenodo.org/badge/802446381.svg)](https://doi.org/10.5281/zenodo.5846374)
[![CI](https://github.com/radiocosmology/caput/actions/workflows/tests.yaml/badge.svg?branch=master)](https://github.com/radiocosmology/caput/actions/workflows/tests.yaml)
[![doc-status](https://app.readthedocs.org/projects/caput/badge/?version=latest)](https://caput.readthedocs.io/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

**caput** provides a variety of utilities for dealing with large datasets
on computer clusters with applications to radio astronomy in mind. For more information,
see the docs: https://caput.readthedocs.io/.

Includes:
- An MPI-distributed `np.ndarray` class
- In-memory mock-ups of `h5py` objects, with support for reading from and writing to `zarr` files
- Lightweight infrastructure for running data pipelines, scaling from a laptop to large, distributed 
clusters
- A collection of fast, optimized algorithms written using `cython`
- A collection of astronomy-oriented coordinate, ephemeris, and time utilities

## Installation
**caput** can be installed with pip in the usual way:
```
$ pip install git+https://github.com/radiocosmology/caput.git
```
