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

For complete functionality, it is recommended to do a `complete` install:
```
$ pip install caput[complete]@git+https://github.com/radiocosmology/caput.git
```

## Contributing
Developers are recommended to do an editable `dev` install:
```
$ pip install caput[dev]@git+https://github.com/radiocosmology/caput.git
```

### Ruff
All new code must be formatted and checked using
```
$ ruff format
$ ruff check
```

### Typing
At minimum, all new code must include type hints for:
- Public function signatures
- Public class attributes, properties, and methods
- Global variables

Additional type hints are encouraged but not required.

### Tests
Tests which should only be run with MPI should be marked using `pytest.mark.mpi`.
Tests that are expected to xfail when run with MPI should be marked with
`pytest.mark.mpi_xfail`.  Finally, tests that should be skipped when running
with MPI should use `pytest.mark.mpi_skip`. When running test with MPI, the
invocation should include the flag `--with-mpi` in order for these marks to be
set correctly. See [pytest-mpi](https://pytest-mpi.readthedocs.io/en/latest/markers.html)
for more information.
