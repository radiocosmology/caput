import types
from pathlib import Path
from typing import overload

import h5py
from mpi4py import MPI

@overload
def open_h5py_mpi(
    f: str | Path | h5py.File,
    mode: str,
    use_mpi: bool = True,
    comm: MPI.Comm | None = None,
) -> h5py.File: ...
@overload
def open_h5py_mpi(
    f: h5py.Group, mode: str, use_mpi: bool = True, comm: MPI.Comm | None = None
) -> h5py.Group: ...

class lock_file:
    filename: str
    rank0: bool
    preserve: bool
    def __init__(
        self, filename: str, preserve: bool = False, comm: MPI.Comm | None = None
    ) -> None: ...
    def __enter__(self) -> str: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> bool: ...
    @property
    def tmpfile(self) -> str: ...
    @property
    def lockfile(self) -> str: ...
