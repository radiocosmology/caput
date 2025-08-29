"""A MPI-distribued numpy array.

This module provides a distributed array object which is a subclass of
:py:class:`numpy.ndarray`. The goal of this module is to enable seamless
use of :py:mod:`numpy` arrays in a distributed memory context, while remaining
as functionally similar to :py:class:`np.ndarray` as possible.

Examples
--------
This example performs a transfrom from time-freq to lag-m space. This involves
Fourier transforming each of these two axes of the distributed array::

    import numpy as np
    from mpi4py import MPI

    from caput.mpiarray import MPIArray

    nfreq = 32
    nprod = 2
    ntime = 32

    Initialise array with (nfreq, nprod, ntime) global shape
    darr1 = MPIArray((nfreq, nprod, ntime), dtype=np.float64)

    # Load in data into parallel array
    for lfi, fi in darr1.enumerate(axis=0):
        darr1[lfi] = load_freq_data(gfi)

    # Perform m-transform (i.e. FFT)
    darr2 = MPIArray.wrap(np.fft.fft(darr1, axis=1), axis=0)

    # Redistribute to get all frequencies onto each process, this performs the
    # global transpose using MPI to make axis=1 the distributed axis, and make
    # axis=0 completely local.
    darr3 = darr2.redistribute(axis=1)

    # Perform the lag transform on the frequency direction.
    darr4 = MPIArray.wrap(np.fft.irfft(darr3, axis=0), axis=1)

Note: If a user wishes to create an MPIArray from an ndarray, they should use
:py:meth:`.MPIArray.wrap`. They should not use :py:func:`np.ndarray.view(MPIArray)`.
Attributes will not be set correctly if they do.

Global Slicing
==============

The :py:class:`.MPIArray` also supports slicing with the global index using the
:py:attr:`.MPIArray.global_slice` property. This can be used for both fetching
and assignment with global indices, supporting the basic slicing notation of
`numpy`.

Its behaviour changes depending on the exact slice it gets:

- A full slice (`:`) along the parallel axis returns a :py:class:`.MPIArray` on
  fetching, and accepts an :py:class:`.MPIArray` on assignment.
- A partial slice (`:`) returns and accepts a numpy array on the rank holding
  the data, and :obj:`None` on other ranks.

It's important to note that it never communicates data between ranks. It only
ever operates on data held on the current rank.

Global Slicing Examples
-----------------------

Here is an example of this in action. Create and set an MPI array:

>>> import numpy as np
>>> from caput.mpiarray import MPIArray
>>> from caput.util import mpitools
>>>
>>> arr = MPIArray((mpitools.size, 3), dtype=np.float64)
>>> arr[:] = 0.0
>>> for ri in range(mpitools.size):
...     if ri == mpitools.rank:
...         print(ri, arr)
...     mpitools.barrier()
0 [[0. 0. 0.]]

Use a global index to assign to the array

>>> arr.global_slice[3] = 17

Fetch a view of the whole array with a full slice

>>> arr2 = arr.global_slice[:, 2]

Print the third column of the array on all ranks

>>> for ri in range(mpitools.size):
...     if ri == mpitools.rank:
...         print(ri, arr2)
...     mpitools.barrier()
0 [0.]

Fetch a view of the whole array with a partial slice. The final two ranks should be None

>>> arr3 = arr.global_slice[:2, 2]
>>> for ri in range(mpitools.size):
...     if ri == mpitools.rank:
...         print(ri, arr3)
...     mpitools.barrier()
0 [0.]

Direct Slicing
==============

:py:class:`.MPIArray` supports direct slicing using ``[...]`` (implemented via
:py:meth:`.MPIArray.__getitem__`). This can be used for both fetching and assignment. It is
recommended to only index into the non-parallel axis or to do a full slice ``[:]``.

Direct Slicing Behaviour
------------------------

- A full slice ``[:]`` will return a :py:class:`~.MPIArray` on fetching, with
  identical properties to the original array.
- Any indexing or slicing into the non-parallel axis, will also return a
  :py:class:`~.MPIArray`. The number associated with the parallel axis,
  will be adjusted if a slice results in an axis reduction.
- Any indexing into the parallel axis is discouraged. This behaviour is
  deprecated. For now, it will result into a local index on each rank,
  returning a regular :py:class:`np.ndarray`, along with a warning.
  In the future, it is encouraged to index into the local array
  :py:attr:`.MPIArray.local_array`, if you wish to locally index into
  the parallel axis

Direct Slicing Examples
-----------------------

.. deprecated:: 21.04
    Direct indexing into parallel axis is DEPRECATED. For now, it will return a numpy
    array equal to local array indexing, along with a warning. This behaviour will be
    removed in the future.

>>> darr = MPIArray((mpitools.size,), axis=0)
>>> (darr[0] == darr.local_array[0]).all()
np.True_
>>> not hasattr(darr[0], "axis")
True

If you wish to index into local portion of a distributed array along its parallel
axis, you need to index into the :py:attr:`MPIArray.local_array`.

>>> darr[:] = 1.0
>>> float(darr.local_array[0])
1.0

indexing into non-parallel axes returns an MPIArray with appropriate attributes
Slicing could result in a reduction of axis, and a lower parallel axis number

>>> darr = MPIArray((4, mpitools.size), axis=1)
>>> darr[:] = mpitools.rank
>>> (darr[0] == mpitools.rank).all()
array([ True])
>>> darr[0].axis == 0
True

ufunc
=====

In NumPy, universal functions (or ufuncs) are functions that operate on ndarrays
in an element-by-element fashion. :py:class:`.MPIArray` supports all ufunc calculations,
except along the parallel axis.

ufunc Requirements
------------------

- Every input :py:class:`.MPIArray` *must* be distributed along the same axis.
- If you pass a kwarg `axis` to the ufunc, it must not be the parallel axis.

ufunc Behaviour
---------------

- If no output are provided, the results are converted back to MPIArrays. The new
  array will either be parallel over the same axis as the input MPIArrays, or possibly
  one axis down if the `ufunc` is applied via a `reduce` method (i.e. the shape of the
  array is reduced by one axis).
- For operations that normally reduce to a scalar, the scalars will be wrapped into a 1D
  array distributed across axis 0.
- shape related attributes will be re-calculated.

ufunc Examples
--------------

Create an array

>>> dist_arr = MPIArray((mpitools.size, 4), axis=0)
>>> dist_arr[:] = mpitools.rank

Element wise summation and `.all()` reduction

>>> (dist_arr + dist_arr == 2 * mpitools.rank).all()
array([ True])

Element wise multiplication and reduction

>>> (dist_arr * 2 == 2 * mpitools.rank).all()
array([ True])

The distributed axis is unchanged during an elementwise operation

>>> (dist_arr + dist_arr).axis == 0
True

An operation on multiple arrays with different parallel axes is not possible and will
result in an exception

>>> (
...     MPIArray((mpitools.size, 4), axis=0) - MPIArray((mpitools.size, 4), axis=1)
... )  # doctest: +NORMALIZE_WHITESPACE
Traceback (most recent call last):
    ...
caput.mpiarray.AxisException: Input argument 1 has an incompatible distributed axis.

Summation across a non-parallel axis

>>> (dist_arr.sum(axis=1) == 4 * mpitools.rank).all()
array([ True])

A sum reducing across all axes will reduce across each local array and give a new
distributed array with a single element on each rank.

>>> (dist_arr.sum() == 4 * 3 * mpitools.rank).all()
array([ True])
>>> (dist_arr.sum().local_shape) == (1,)
True
>>> (dist_arr.sum().global_shape) == (mpitools.size,)
True

Reduction methods might result in a decrease in the distributed axis number

>>> dist_arr = MPIArray((mpitools.size, 4, 3), axis=1)
>>> dist_arr.sum(axis=0).axis == 0
True

MPI.Comm
========

:py:obj:`mpi4py.MPI.Comm` provides a wide variety of functions for `communication across nodes`_
They provide an upper-case and lower-case variant of many functions. With MPIArrays,
please use the uppercase variant of the function. The lower-case variants involve an
intermediate pickling process, which can lead to malformed arrays.

.. _`communication across nodes`: https://mpi4py.readthedocs.io/en/stable/overview.html?highlight=allreduce#collective-communications
"""

# Required to type annotate classes defined later
# in the same module
from __future__ import annotations

import logging
import warnings
from itertools import pairwise
from types import EllipsisType
from typing import TYPE_CHECKING

import numpy as np

from .memdata import fileformats
from .util import mpitools

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import Any, Literal

    # TODO: this should all get moved to `fileformats` module
    import h5py
    import numpy.typing as npt
    import zarr
    from mpi4py import MPI


__all__ = ["AxisException", "MPIArray", "UnsupportedOperation", "ones", "zeros"]


logger = logging.getLogger(__name__)


# Custom type definitions
# SelectionTupleLike = TypeVar("SelectionTupleLike", bound=tuple[slice | int | EllipsisType])
SelectionTupleLike = tuple[slice | int | EllipsisType]
SelectionLike = SelectionTupleLike | slice | int | EllipsisType


class _global_resolver:
    """Private class implementing global sampling for MPIArray."""

    def __init__(self, array: MPIArray) -> None:
        self.array: MPIArray = array
        self.axis: int = array.axis
        self.offset: int = array.local_offset[self.axis]
        self.length: int = array.global_shape[self.axis]

    def _resolve_slice(
        self, slobj: SelectionLike
    ) -> tuple[tuple[slice | int, ...], bool]:
        """Transforms a basic slice into a slice on the local array.

        Converts a basic numpy slice on the global array into a fully
        fleshed out slice tuple referencing the positions in the local arrays.
        If a single integer index is specified for the distributed axis, then
        either the local index is returned, or None if it doesn't exist on the
        current rank.
        """
        ndim = self.array.ndim
        local_length = self.array.shape[self.axis]

        # Expand a single integer or slice index
        if isinstance(slobj, int | slice):
            slobj: SelectionTupleLike = (slobj, Ellipsis)

        # Add an ellipsis if length of slice object is too short
        if isinstance(slobj, tuple) and len(slobj) < ndim:
            # This is a workaround for the more straightforward
            # `Ellipsis not in slobj`. If one of the axes is indexed
            # by a numpy array, the simpler logic will check each element
            # of the array and will fail. Comparing each object directly
            # gets around this.
            if not any(obj is Ellipsis for obj in slobj):
                slobj: SelectionTupleLike = (*slobj, Ellipsis)

        # Expand an ellipsis
        slice_list = []
        for sl in slobj:
            if sl is Ellipsis:
                for _ in range(ndim - len(slobj) + 1):
                    slice_list.append(slice(None, None))
            else:
                slice_list.append(sl)

        fullslice = True

        # Process the parallel axis. Calculate the correct index for the
        # containing rank, and set None on all others.
        if isinstance(slice_list[self.axis], int):
            index = slice_list[self.axis] - self.offset
            slice_list[self.axis] = (
                None if (index < 0 or index >= local_length) else index
            )
            fullslice = False

        # If it's a slice, then resolve any offsets
        # If any of start or stop is defined then mark that this is not a complete slice
        # Also mark if there is any actual data on this rank
        elif isinstance(slice_list[self.axis], slice):
            s = slice_list[self.axis]
            start = s.start
            stop = s.stop
            step = s.step

            # Check if start is defined, and modify slice
            if start is not None:
                start = (
                    start if start >= 0 else start + self.length
                )  # Resolve negative indices
                fullslice = False
                start = start - self.offset
            else:
                start = 0

            # Check if stop is defined and modify slice
            if stop is not None:
                stop = (
                    stop if stop >= 0 else stop + self.length
                )  # Resolve negative indices
                fullslice = False
                stop = stop - self.offset
            else:
                stop = local_length

            # If step is defined we don't need to adjust this, but it's no longer a
            # complete slice
            if step is not None:
                fullslice = False

            # If there is no data on this node place None on the parallel axis
            if start >= local_length or stop < 0:
                slice_list[self.axis] = None
            else:
                # Normalise the indices and create slice
                start = max(min(start, local_length), 0)
                stop = max(min(stop, local_length), 0)
                slice_list[self.axis] = slice(start, stop, step)

        return tuple(slice_list), fullslice

    def __getitem__(self, slobj: SelectionLike) -> np.ndarray | None:
        # Resolve the slice object
        slobj, is_fullslice = self._resolve_slice(slobj)

        # If not a full slice, return a numpy array (or None)
        if not is_fullslice:
            # If the distributed axis has a None, that means there is no data at that
            # index on this rank
            if slobj[self.axis] is None:
                return None

            return self.array.local_array[slobj]

        # Fix up slobj for axes where there is no data
        slobj: SelectionTupleLike = tuple(
            slice(None, None, None) if sl is None else sl for sl in slobj
        )

        return self.array[slobj]

    def __setitem__(self, slobj: SelectionLike, value: np.generic) -> None:
        slobj, _ = self._resolve_slice(slobj)

        # If the distributed axis has a None, that means that index is not available on
        # this rank
        if slobj[self.axis] is None:
            return

        self.array[slobj] = value


class MPIArray(np.ndarray):
    r"""A numpy-like array distributed across multiple processes.

    Parameters
    ----------
    global_shape : Sequence[int]
        The global shape of the array.
    axis : int, optional
        The axis over which the array is distributed. Default is 0.
    comm : MPI.Comm | None, optional
        The communicator over which the array is distributed.
        Default is None, which will use `MPI.COMM_WORLD`.
    args : Any, optional
        Additional positional arguments to pass to the :py:class:`~np.ndarray` constructor.
    \**kwargs : Any, optional
        Additional keyword arguments to pass to the :py:class:`~np.ndarray` constructor.
    """

    def __getitem__(self, slobj: SelectionLike) -> np.ndarray:
        """Returned a slice view of this array."""
        # ensure slobj is a tuple, with one entry for every axis
        if not isinstance(slobj, tuple):
            slobj: SelectionTupleLike = (slobj,)

        slobj, axis_map, final_map = sanitize_slice(slobj, self.ndim)

        # __getitem__ should not be receiving sub-slices or direct indexes on the
        # distributed axis. global_slice should be used for both
        dist_axis = axis_map[self.axis]
        dist_axis_index = slobj[dist_axis]
        if (dist_axis_index != slice(None, None, None)) and (
            dist_axis_index != slice(0, self.local_array.shape[self.axis], None)
        ):
            if isinstance(dist_axis_index, int):
                warnings.warn(
                    "You are indexing directly into the distributed axis "
                    "returning a view into the local array. "
                    "Please use global_slice, or .local_array before indexing instead.",
                    stacklevel=2,
                )

                return self.local_array.__getitem__(slobj)
            warnings.warn(
                "You are directly sub-slicing the distributed axis "
                "returning a view into the local array. "
                "Please use global_slice, or .local_array before indexing.",
                stacklevel=2,
            )
            return self.local_array.__getitem__(slobj)

        # Calculate the final position of the distributed axis
        final_dist_axis = final_map[axis_map[self.axis]]

        if final_dist_axis is None:
            # Should not get here
            raise IndexError("Distributed axis does not appear in final output.")

        if final_dist_axis == self.axis:
            return super().__getitem__(slobj)

        # the MPIArray array_finalize assumes that the output distributed axis is the
        # same as the source since the number for the distributed axes has changed, we
        # will need a fresh MPIArray object
        # First we do the actual slice on the numpy arrays
        arr_sliced = self.local_array.__getitem__(slobj)

        # determine the shape of the new array
        # grab the length of the distributed axes from the original
        # instead of performing an mpi.allreduce
        new_global_shape = list(arr_sliced.shape)

        # if a single value, not an array, just return
        if not new_global_shape:
            return arr_sliced

        # Create a view of the numpy sliced array as an MPIArray.
        return self._view_from_data_and_params(
            arr_sliced,
            final_dist_axis,
            self.global_shape[self.axis],
            self.local_offset[self.axis],
            self.comm,
        )

    def __setitem__(self, slobj: SelectionLike, value: np.generic) -> None:
        self.local_array.__setitem__(slobj, value)

    def __repr__(self) -> str:
        return self.local_array.__repr__()

    def __str__(self) -> str:
        return self.local_array.__str__()

    @property
    def global_shape(self) -> tuple[int, ...]:
        """Global array shape.

        Returns
        -------
        global_shape : tuple[int, ...]
        """
        return self._global_shape

    @property
    def axis(self) -> int:
        """Distributed axis.

        Returns
        -------
        distributed_axis : int
        """
        return self._axis

    @property
    def local_shape(self) -> tuple[int, ...]:
        """Shape of local section.

        Returns
        -------
        rank_local_shape : int
        """
        return self._local_shape

    @property
    def local_offset(self) -> tuple[int, ...]:
        """Offset into global array.

        This is equivalent to the global-index of
        the [0, 0, ...] element of the local section.

        Returns
        -------
        rank_local_offset : tuple[int, ...]
        """
        return self._local_offset

    @property
    def local_array(self) -> np.ndarray:
        """rank-local :py:class:`numpy.ndarray` view of the array.

        Returns
        -------
        rank_local_array : ndarray
        """
        return self.view(np.ndarray)

    @property
    def local_bounds(self) -> slice:
        """Global bounds of the local array along the distributed axis.

        Returns
        -------
        rank_local_bounds : slice
        """
        return slice(
            self.local_offset[self.axis],
            self.local_offset[self.axis] + self.local_shape[self.axis],
        )

    @property
    def comm(self) -> MPI.Comm:
        """The communicator over which the array is distributed.

        Returns
        -------
        comm : MPI.Comm
        """
        return self._comm

    @comm.setter
    def comm(self, var: MPI.Comm) -> None:
        """Set the communicator over which the array is distributed.

        Parameters
        ----------
        var : MPI.Comm
            New communicator
        """
        self._comm = var

    def __new__(
        cls,
        global_shape: tuple[int, ...],
        axis: int = 0,
        comm: MPI.Comm | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> MPIArray:
        """Create a new MPIArray instance."""
        if comm is None:
            comm = mpitools.world

        # Determine local section of distributed axis
        local_num, local_start, _ = mpitools.split_local(global_shape[axis], comm=comm)

        # Figure out the local shape and offset
        lshape = list(global_shape)
        lshape[axis] = local_num

        loffset = [0] * len(global_shape)
        loffset[axis] = local_start

        # Create array
        arr = np.ndarray.__new__(cls, lshape, *args, **kwargs)

        # Set attributes of class
        arr._global_shape = tuple(global_shape)
        arr._axis = axis
        arr._local_shape = tuple(lshape)
        arr._local_offset = tuple(loffset)
        arr._comm = comm

        return arr

    @property
    def global_slice(self) -> _global_resolver:
        """Return an objects that presents a view of the array with global slicing.

        Returns
        -------
        global_slice_resolver : _global_resolver
            :py:class:`._global_resolver` instance.
        """
        return _global_resolver(self)

    @classmethod
    def _view_from_data_and_params(
        cls,
        array: npt.NDArray,
        axis: int,
        global_length: int,
        local_start: int,
        comm: MPI.Comm,
    ) -> MPIArray:
        """Create an MPIArray view of a numpy array with relevant distribution.

        Parameters
        ----------
        array : ndarray
            :py:class:`~np.ndarray` to wrap.
        axis : int
            New distributed axis.
        global_length : int
            Global length of the array along the distributed axis.
        local_start : int
            Global index of the first element of the local array.
        comm : MPI.Comm
            :py:class:`mpi4py.MPI.Comm` communicator.

        Returns
        -------
        array_view : MPIArray
            :py:class:`.MPIArray` view of the input array.
        """
        # Set shape and offset
        lshape = array.shape
        global_shape = list(lshape)
        global_shape[axis] = global_length

        loffset = [0] * len(lshape)
        loffset[axis] = local_start

        # Setup attributes of class
        dist_arr = array.view(cls)
        dist_arr._global_shape = tuple(global_shape)
        dist_arr._axis = axis
        dist_arr._local_shape = tuple(lshape)
        dist_arr._local_offset = tuple(loffset)
        dist_arr._comm = comm

        return dist_arr

    @classmethod
    def wrap(
        cls, array: npt.NDArray, axis: int, comm: MPI.Comm | None = None
    ) -> MPIArray:
        """Turn :py:class:`~np.ndarray`s spread across ranks into a distributed :py:class:`.MPIArray` object.

        Parameters
        ----------
        array : ndarray
            Array to wrap.
        axis : int
            Axis over which the array is distributed. The lengths are checked
            to try and ensure this is correct.
        comm : MPI.Comm | None, optional
            The communicator over which the array is distributed. If `None`
            (default), use `MPI.COMM_WORLD`.

        Returns
        -------
        wrapped_array : MPIArray
            :py:class:`.MPIArray` view of the input :py:class:`~np.ndarray`.

        Raises
        ------
        AxisException
            If the distributed axis is invalid.
        """
        if comm is None:
            comm = mpitools.world

        # Get axis length, both locally, and globally
        try:
            axlen = array.shape[axis]
        except IndexError as e:
            raise AxisException(
                f"Distributed axis {axis} does not exist in global shape {array.shape}"
            ) from e

        totallen = mpitools.allreduce(axlen, comm=comm)

        # Figure out what the distributed layout should be
        local_num, local_start, _ = mpitools.split_local(totallen, comm=comm)

        # Check the local layout is consistent with what we expect, and send
        # result to all ranks.
        layout_issue = mpitools.allreduce(
            axlen != local_num, op=mpitools.MAX, comm=comm
        )

        if layout_issue:
            raise AxisException(
                "Cannot wrap, distributed axis local length is incorrect."
            )

        return cls._view_from_data_and_params(array, axis, totallen, local_start, comm)

    def redistribute(self, axis: int) -> MPIArray:
        """Change the rank distribution axis of the array.

        Guarantees that the result is c-contiguous.

        Parameters
        ----------
        axis : int
            Axis to distribute over.

        Returns
        -------
        redistributed_array : MPIArray
            :py:class:`.MPIArray` instance of the original array, distributed
            over the new axis. If the new axis is the same as the old axis,
            the original array is returned.

        Raises
        ------
        MemoryError
            If there is not enough memory to perform the redistribution.
        """
        # Check to see if this is the current distributed axis
        if self.axis == axis or self.comm is None:
            return self

        # Avoid repeat mpi property calls
        csize = self.comm.size
        crank = self.comm.rank

        if csize == 1:
            return MPIArray.wrap(self.local_array, axis, self.comm)

        # Check to make sure there is enough memory to perform the redistribution.
        # Must be able to allocate the target array and 2 buffers. We allocate
        # slightly more space than needed to be safe
        min_req_mem = int((1 + 4 / self.comm.size) * self.local_array.nbytes)

        if not mpitools.can_allocate(min_req_mem, scope="process", comm=self.comm):
            raise MemoryError(
                f"Cannot allocate {min_req_mem} bytes required for redistribute."
            )

        # Make a new distributed array
        dist_arr = MPIArray(
            self.global_shape, axis=axis, dtype=self.dtype, comm=self.comm
        )

        # Get views into local and target arrays
        arr = self.local_array
        target_arr = dist_arr.local_array

        # Get the start and end of each subrange of interest
        _, sac, eac = mpitools.split_all(self.global_shape[axis], self.comm)
        _, sar, ear = mpitools.split_all(self.global_shape[self.axis], self.comm)
        # Split the soruce array into properly sized blocks for sending
        blocks = np.array_split(arr, np.insert(eac, 0, sac[0]), axis)[1:]
        # Create fixed-size contiguous buffers for sending and receiving
        buffer_shape = list(target_arr.shape)
        buffer_shape[self.axis] = max(ear - sar)
        buffer_shape[axis] = max(eac - sac)
        # Pre-allocate buffers and buffer type
        recv_buffer = np.empty(buffer_shape, dtype=self.dtype)
        send_buffer = np.empty_like(recv_buffer)
        buf_type = self._prep_buf(send_buffer)[1]

        # Empty slices for target, send buf, recv buf
        targetsl = [slice(None)] * len(buffer_shape)
        sendsl = [slice(None)] * len(buffer_shape)
        recvsl = [slice(None)] * len(buffer_shape)
        # Send and recv buf have some fixed axis slices per rank
        sendsl[self.axis] = slice(ear[crank] - sar[crank])
        recvsl[axis] = slice(eac[crank] - sac[crank])

        mpistatus = mpitools.MPI.Status()

        # Cyclically pass and receive array chunks across ranks
        for i in range(csize):
            send_to = (crank + i) % csize
            recv_from = (crank - i) % csize

            # Write send data into send buffer location
            sendsl[axis] = slice(eac[send_to] - sac[send_to])
            send_buffer[tuple(sendsl)] = blocks[send_to]

            self.comm.Sendrecv(
                sendbuf=(send_buffer, buf_type),
                dest=send_to,
                sendtag=(csize * crank + send_to),
                recvbuf=(recv_buffer, buf_type),
                source=recv_from,
                recvtag=(csize * recv_from + crank),
                status=mpistatus,
            )

            if mpistatus.error != mpitools.MPI.SUCCESS:
                raise RuntimeError(
                    f"**** ERROR in MPI SEND/RECV "
                    f"(rank={crank}, "
                    f"target={send_to}, "
                    f"receive={recv_from}) ****"
                )

            # Write buffer into target location
            targetsl[self.axis] = slice(sar[recv_from], ear[recv_from])
            recvsl[self.axis] = slice(ear[recv_from] - sar[recv_from])

            target_arr[tuple(targetsl)] = recv_buffer[tuple(recvsl)]

        return dist_arr

    def allreduce(self, op: MPI.Op | None = None) -> np.number:
        """Perform MPI reduction `op` within memory buffer.

        Usage is restricted to arrays with a single element per rank.
        Returns same scalar final result to all ranks.

        E.g. usage: mpi_array.sum().allreduce()
        to every rank.

        Parameters
        ----------
        op : MPI.Op | None
            Reduction operation to perform. Default: :py:func:`~mpi4py.MPI.SUM`.

        Returns
        -------
        value : number
            Result of reduction. This is a single value which is the
            same on all ranks.
        """
        if self.local_shape != (1,):
            raise ValueError(
                "MPIArray.allreduce()'s is limited to arrays with a single element per "
                f"rank, but rank={self.comm.rank} has shape-{self.local_shape}."
            )

        result_arr = np.zeros((1,), dtype=self.dtype)
        self.comm.Allreduce(self, result_arr, mpitools.SUM if op is None else op)

        return result_arr[0]

    def enumerate(self, axis: int) -> Iterator[tuple[int, int]]:
        """Helper for enumerating over a given axis.

        Parameters
        ----------
        axis : int
            Axis to enumerate over.

        Returns
        -------
        enumerator : enumerate
            An enumerator which returns the local index into the array *and*
            the global index it corresponds to.
        """
        start = self.local_offset[axis]
        end = start + self.local_shape[axis]

        return enumerate(range(start, end))

    @classmethod
    def from_hdf5(
        cls,
        f: str | h5py.File,
        dataset: str,
        comm: MPI.Comm | None = None,
        axis: int = 0,
        sel: SelectionTupleLike | None = None,
    ) -> MPIArray:
        """Read MPIArray from an HDF5 dataset in parallel.

        Parameters
        ----------
        f : str | h5py.File
            File to read dataset from.
        dataset : str
            Name of dataset to read from. Must exist.
        comm : MPI.Comm | None, optional
            MPI communicator to distribute over. If `None` optional, use
            `MPI.COMM_WORLD`.
        axis : int, optional
            Axis over which the read should be distributed. This can be used
            to select the most efficient axis for the reading. Default is 0.
        sel : SelectionTupleLike | None, optional
            A tuple of slice objects used to make a selection from the array
            *before* reading. The output will be this selection from the dataset
            distributed over the given axis.

        Returns
        -------
        array : MPIArray
            :py:class:`.MPIArray` read from the dataset.
        """
        return cls.from_file(f, dataset, comm, axis, sel, file_format=fileformats.HDF5)

    @classmethod
    def from_file(
        cls,
        f: str | h5py.File | h5py.Group | zarr.Group,
        dataset: str,
        comm: MPI.Comm | None = None,
        axis: int = 0,
        sel: SelectionTupleLike | None = None,
        file_format: fileformats.FileFormat = fileformats.HDF5,
    ) -> MPIArray:
        """Read MPIArray from an HDF5 dataset or Zarr array on disk in parallel.

        Parameters
        ----------
        f : str | h5py.File | h5py.Group | zarr.Group
            File to read dataset from.
        dataset : str
            Name of dataset to read from. Must exist.
        comm : MPI.Comm | None, optional
            MPI communicator to distribute over. If `None` optional, use
            :py:obj:`~mpi4py.MPI.COMM_WORLD`.
        axis : int, optional
            Axis over which the read should be distributed. This can be used
            to select the most efficient axis for the reading.
        sel : SelectionTupleLike | None, optional
            A tuple of slice objects used to make a selection from the array
            *before* reading. The output will be this selection from the dataset
            distributed over the given axis.
        file_format : fileformat.FileFormat, optional
            File format to use. Default :py:class:`~caput.memdata.fileformats.HDF5`.

        Returns
        -------
        array : MPIArray
            :py:class:`MPIArray` read from the dataset.

        Raises
        ------
        RuntimeError
            If the file format is not supported.
        ValueError
            If the dataset argument type is invalid.
        AxisException
            If the distributed axis is invalid.
        """
        if file_format == fileformats.HDF5:
            from .memdata._io import open_h5py_mpi

            # Don't bother using MPI where the axis is not zero. It's probably just slower.
            # TODO: with tuning this might not be true. Keep an eye on this.
            use_mpi = axis > 0

            # Read the file. Opening with MPI if requested, and we can
            fh = open_h5py_mpi(f, "r", use_mpi=use_mpi, comm=comm)
        elif file_format == fileformats.Zarr:
            try:
                import numcodecs  # noqa: F401
            except ImportError:
                raise RuntimeError("Install numcodecs to read from zarr files.")

            # NOTE: blosc may share incorrect global state amongst processes causing programs to hang.
            # See https://zarr.readthedocs.io/en/stable/tutorial.html#parallel-computing-and-synchronization
            # I think this has now been resolved
            # (https://github.com/zarr-developers/numcodecs/pull/42), so I'm
            # reenabling threads, but we should be careful
            # numcodecs.blosc.use_threads = False

            if isinstance(f, str):
                fh = file_format.open(f, "r")
            elif isinstance(f, file_format.module.Group):
                fh = f
            else:
                raise ValueError(
                    f"Can't write to {f} (Expected a {file_format.module.__name__}.Group or str filename)."
                )

        dset = fh[dataset]
        dshape = dset.shape  # Shape of the underlying dataset
        naxis = len(dshape)
        dtype = np.dtype(dset.dtype)

        # Check that the axis is valid and wrap to an actual position
        if axis < -naxis or axis >= naxis:
            raise AxisException(
                f"Distributed axis {axis} not in range ({-naxis}, {naxis - 1})"
            )
        axis = naxis + axis if axis < 0 else axis

        # Ensure sel is defined to cover all axes
        sel: list = _expand_sel(sel, naxis)

        # Figure out the final array size and create it
        gshape = []
        for l, sl in zip(dshape, sel):
            gshape.append(_len_slice(sl, l))
        dist_arr = cls(gshape, axis=axis, comm=comm, dtype=dtype)

        # Get the local start and end indices
        lstart = dist_arr.local_offset[axis]
        lend = lstart + dist_arr.local_shape[axis]

        # Create the slice object into the dataset by resolving the rank's slice on the
        # sel
        sel[axis] = _reslice(sel[axis], dshape[axis], slice(lstart, lend))
        sel: tuple = tuple(sel)

        if file_format == fileformats.HDF5:
            # Split the axis to get the IO size under ~2GB (only if MPI-IO)
            split_axis, partitions = dist_arr._partition_io(skip=(not fh.is_mpi))

            # Check that there are no null slices, otherwise we need to turn off
            # collective IO to work around an h5py issue (#965)
            no_null_slices = dist_arr.global_shape[axis] >= dist_arr.comm.size

            if fh.is_mpi and no_null_slices:
                iocontext = dset.collective
            else:
                from contextlib import nullcontext

                iocontext = nullcontext()

            # Read using collective MPI-IO if specified
            with iocontext:
                # Loop over partitions of the IO and perform them
                for part in partitions:
                    islice, fslice = _partition_sel(
                        sel, split_axis, dshape[split_axis], part
                    )
                    if dtype.byteorder == "=":
                        # Native dtype so use `read_direct` to limit copies. This won't
                        # necessarily always be faster, but it should never be slower
                        dset.read_direct(dist_arr, islice, fslice)
                    else:
                        # There's a bug in `h5py` which causes endianess conversions
                        # to be very slow, so it's faster to just read and let numpy
                        # do the conversion. I haven't quite nailed down the cases where
                        # this happens, so for now just do it most of the time.
                        # https://github.com/h5py/h5py/issues/1851
                        dist_arr[fslice] = dset[islice]

            if fh.opened:
                fh.close()
        else:
            # If using zarr we can directly read the full dataset into the final array
            dset.get_basic_selection(sel, out=dist_arr.local_array)

        return dist_arr

    def to_hdf5(
        self,
        f: str | h5py.File | h5py.Group,
        dataset: str,
        create: bool = False,
        chunks: tuple[int, ...] | None = None,
        compression: str | int | None = None,
        compression_opts: tuple | None = None,
    ) -> None:
        """Parallel write into a contiguous HDF5 dataset.

        Parameters
        ----------
        f : str | h5py.File | h5py.Group
            File to write dataset into.
        dataset : str
            Name of dataset to write into. Should not exist.
        create : bool, optional
            True if a new file should be created (if needed).
        chunks : tuple[int, ...] | None, optional
            Chunk shape.
        compression : str | int | None, optional
            Name or identifier of HDF5 compression filter.
        compression_opts : tuple | None, optional
            See HDF5 documentation for compression filters.
            Compression options for the dataset.

        Raises
        ------
        ValueError
            If :py:mod:`h5py` is built without MPI support and `f` is not a string.
        """
        import h5py

        if not h5py.get_config().mpi:
            if isinstance(f, str):
                self._to_hdf5_serial(f, dataset, create)
                return

            raise ValueError(
                "Argument must be a filename if h5py does not have MPI support"
            )

        from .memdata._io import open_h5py_mpi

        mode = "a" if create else "r+"
        fh = open_h5py_mpi(f, mode, self.comm)

        sel = self._make_selections()

        # Split the axis to get the IO size under ~2GB (only if MPI-IO)
        split_axis, partitions = self._partition_io(skip=(not fh.is_mpi))

        dset = _create_or_get_dset(
            fh,
            dataset,
            shape=self.global_shape,
            dtype=self.dtype,
            chunks=chunks,
            compression=compression,
            compression_opts=compression_opts,
        )

        # Decide whether to use collective IO. There are a few relevant aspects to this
        # one:
        # - we must use collective IO if writing chunked/compressed data. If it's not
        #   available we cannot compress the data
        # - we cannot use collective IO if there are null slices present (h5py bug)
        # TODO: change if h5py bug fixed https://github.com/h5py/h5py/issues/965
        no_null_slices = self.global_shape[self.axis] >= self.comm.size

        if fh.is_mpi and no_null_slices:
            # We can use collective IO
            iocontext = dset.collective
        elif compression is not None:
            # There's a logic case here where `fh.is_mpi` is False but compression
            # is set. In theory, this should never be possible since we checked
            # for MPI support at the start of this function
            raise RuntimeError(
                "Cannot use collective IO, must disable compression. "
                f"MPI enabled: {fh.is_mpi}, null slice present: {not no_null_slices}."
            )
        else:
            # We cannot use collective IO, so use a dummy context manager
            from contextlib import nullcontext

            iocontext = nullcontext()

        # Read using collective MPI-IO if specified
        with iocontext:
            # Loop over partitions of the IO and perform them
            for part in partitions:
                islice, fslice = _partition_sel(
                    sel, split_axis, self.global_shape[split_axis], part
                )
                if np.dtype(dset.dtype).byteorder == "=" and self.flags["C_CONTIGUOUS"]:
                    # Native dtype and C contiguous so use `write_direct` to limit copies.
                    # This won't necessarily always be faster, but it should never be slower
                    dset.write_direct(self, fslice, islice)
                else:
                    # There's a bug in `h5py` which causes endianess conversions
                    # to be very slow, so it's faster to just write and let numpy
                    # do the conversion. I haven't quite nailed down the cases where
                    # this happens, so for now just do it most of the time.
                    # https://github.com/h5py/h5py/issues/1851
                    dset[islice] = self[fslice]

        if fh.opened:
            fh.close()

    def to_zarr(
        self,
        f: str | zarr.Group,
        dataset: str,
        create: bool,
        chunks: tuple[int, ...] | None,
        compression: str | int | None,
        compression_opts: tuple | None = None,
    ) -> None:
        """Parallel write into a contiguous Zarr dataset.

        Parameters
        ----------
        f : str | zarr.Group
            File to write dataset into.
        dataset : str
            Name of dataset to write into. Should not exist.
        create : bool
            True if a new file should be created (if needed)
        chunks : tuple[int, ...] | None, optional
            Chunk shape.
        compression : str | int | None, optional
            Name or identifier of compression filter.
        compression_opts : tuple | None, optional
            See documentation for compression filters.
            Compression options for the dataset.

        Raises
        ------
        RuntimeError
            If :py:mod:`zarr` is not installed.
        ValueError
            If :py:mod:`zarr` group is already initialized without
            a synchronizer, or if file name argument type is invalid.
        """
        try:
            import numcodecs  # noqa: F401
            import zarr
        except ImportError as err:
            raise RuntimeError(
                f"Can't write to zarr file. Please install zarr and numcodecs: {err}"
            )

        # NOTE: blosc may share incorrect global state amongst processes causing programs to hang.
        # See https://zarr.readthedocs.io/en/stable/tutorial.html#parallel-computing-and-synchronization
        # I think this has now been resolved
        # (https://github.com/zarr-developers/numcodecs/pull/42), so I'm
        # reenabling threads, but we should be careful
        # numcodecs.blosc.use_threads = False

        mode = "a" if create else "r+"
        extra_args = fileformats.Zarr.compression_kwargs(
            compression=compression,
            compression_opts=compression_opts,
        )

        lockfile = None

        if isinstance(f, str):
            if self.comm.rank == 0 and create:
                zarr.open(store=f, mode=mode)
            lockfile = f".{f}.sync"
            self.comm.Barrier()
            group = zarr.open_group(
                store=f,
                mode="r+",
                synchronizer=zarr.ProcessSynchronizer(lockfile),
            )
        elif isinstance(f, zarr.Group):
            if f.synchronizer is None:
                raise ValueError(
                    "Got zarr.Group without synchronizer, can't perform parallel write."
                )
            group = f
        else:
            raise ValueError(
                f"Can't write to {f} (Expected a zarr.Group or str filename)."
            )

        sel = self._make_selections()

        # Split the axis
        split_axis, partitions = self._partition_io(skip=True)

        if self.comm.rank == 0:
            _create_or_get_dset(
                group,
                dataset,
                shape=self.global_shape,
                dtype=self.dtype,
                chunks=chunks,
                **extra_args,
            )
        self.comm.Barrier()

        for part in partitions:
            islice, fslice = _partition_sel(
                sel, split_axis, self.global_shape[split_axis], part
            )
            group[dataset][islice] = self.local_array[fslice]
        self.comm.Barrier()
        if self.comm.rank == 0 and lockfile is not None:
            fileformats.remove_file_or_dir(lockfile)

    def to_file(
        self,
        f: str | h5py.File | h5py.Group | zarr.Group,
        dataset: str,
        create: bool = False,
        chunks: tuple[int, ...] | None = None,
        compression: str | int | None = None,
        compression_opts: tuple | None = None,
        file_format: fileformats.FileFormat = fileformats.HDF5,
    ) -> None:
        """Parallel write into a contiguous HDF5/Zarr dataset.

        Parameters
        ----------
        f : str | h5py.File | h5py.Group | zarr.Group
            File to write dataset into.
        dataset :  str
            Name of dataset to write into. Should not exist.
        create : bool
            True if a new file should be created (if needed)
        chunks : tuple[int, ...] | None, optional
            Chunk shape
        compression : str | int | None, optional
            Name or identifier of HDF5 compression filter.
        compression_opts : tuple | None, optional
            See HDF5 documentation for compression filters.
            Compression options for the dataset.
        file_format : fileformats.FileFormat
            Instance of :py:class:`~caput.fileformats.FileFormat` specifying
            the file format to use. Default :py:class:`~caput.fileformats.HDF5`.
        """
        if chunks is None and hasattr(self, "chunks"):
            logger.debug(f"getting chunking opts from mpiarray: {self.chunks}")
            chunks = self.chunks

        if compression is None and hasattr(self, "compression"):
            logger.debug(f"getting compression opts from mpiarray: {self.compression}")
            compression = self.compression

        if compression_opts is None and hasattr(self, "compression_opts"):
            logger.debug(
                f"getting compression_opts opts from mpiarray: {self.compression_opts}"
            )
            compression_opts = self.compression_opts

        if file_format == fileformats.HDF5:
            self.to_hdf5(f, dataset, create, chunks, compression, compression_opts)
        elif file_format == fileformats.Zarr:
            self.to_zarr(f, dataset, create, chunks, compression, compression_opts)
        else:
            raise ValueError(f"Unknown file format: {file_format}")

    def _make_selections(self) -> list[slice | int]:
        """Make selections for writing local data to distributed file."""
        # Construct slices for axis
        sel = ([slice(None, None)] * self.axis) + [self.local_bounds]

        return _expand_sel(sel, self.ndim)

    def transpose(self, *axes: Sequence[int] | int | None) -> MPIArray:  # noqa: D417
        r"""Transpose the array axes.

        Parameters
        ----------
        \*axes : Sequence[int] | int | None, optional
            Transpose arguments:

            - None or no argument: reverses the order of the axes.
            - sequence of ints: `i` in the `j`-th place in the tuple means ``a``'s `i`-th axis
              becomes ``a.transpose()``'s `j`-th axis.
            - n ints: same as an n-tuple of the same ints (this form is intended simply
              as a “convenience” alternative to the tuple form)

        Returns
        -------
        transposed_array : MPIArray
            Transposed :py:class:`MPIArray` view of the original data.
        """
        tdata = np.ndarray.transpose(self, *axes)

        if len(axes) == 1 and isinstance(axes[0], tuple | list):
            axes: list | tuple = axes[0]
        elif axes is None or axes == ():
            axes: list = list(range(self.ndim - 1, -1, -1))

        tdata._global_shape = tuple(self.global_shape[ax] for ax in axes)
        tdata._local_shape = tuple(self.local_shape[ax] for ax in axes)
        tdata._local_offset = tuple(self.local_offset[ax] for ax in axes)

        tdata._axis = list(axes).index(self.axis)
        tdata._comm = self._comm

        return tdata

    def reshape(self, *shape: Sequence[int | None], order: str = "C") -> MPIArray:  # noqa: D417
        r"""Reshape the array.

        Must not attempt to reshape the distributed axis. That axis must be
        given an input length `None`.

        Parameters
        ----------
        \*shape : Sequence[int | None]
            Sequence of axis lengths. The distributed must be given `None`.
        order : str, optional
            Read/write index order, ignoring memory layout of underlying array.
            'C' means C-like order, 'F' means Fortran-like order. 'A' uses Fortran
            order _if_ array is Fortran-contiguous in memory, and C order otherwise.

        Returns
        -------
        reshaped_array : MPIArray
            Reshaped :py:class:`MPIArray` as a view of the original data.
        """

        def _check_shapes(
            current_shape: Sequence[int], new_shape: Sequence[int]
        ) -> tuple[int, ...]:
            """Check that we can reshape one array into another.

            Returns the fully fleshed out shape, or raises a ValueError.
            """
            actual_size = 1
            for s in current_shape:
                actual_size *= s

            wildcard_pos = -1

            new_size = 1
            for ii, s in enumerate(new_shape):
                if s == -1 and wildcard_pos < 0:
                    wildcard_pos = ii
                elif s == -1 and wildcard_pos >= 0:
                    raise ValueError(
                        f"Found two wildcards (-1) in new array shape {new_shape}"
                    )
                else:
                    new_size *= s

            if wildcard_pos >= 0 and actual_size % new_size == 0:
                # If there was a wildcard then it then we need to check that it would be
                # a sane value and calculate what it is
                new_shape: list = list(new_shape)
                new_shape[wildcard_pos] = actual_size // new_size
                return tuple(new_shape)

            if wildcard_pos < 0 and actual_size == new_size:
                # If not, the total sizes must be exactly equal
                return tuple(new_shape)
            # Throw an error. Note that this exception gives the full array shapes
            # not just the parts being tested to make more sense for the user.
            raise ValueError(
                f"Cannot reshape MPIArray of shape={self.shape} into new shape={shape}."
            )

        if len(shape) == 1 and isinstance(shape[0], tuple | list):
            shape: tuple = tuple(shape[0])

        # Fill in the missing value
        local_shape = list(shape)
        new_axis = local_shape.index(None)
        local_shape[new_axis] = self.local_shape[self.axis]

        # We will check the reshaping in two steps so we can be sure that we are not
        # messing up the axis distribution. In both cases we will *include* the
        # distributed axis in the test as this takes care of cases where we are adding
        # length-1 axes where there are currently no axes.
        # First, we test any axes after the distributed axis...
        post_shape = self.local_shape[self.axis :]
        new_post_shape: list = local_shape[new_axis:]
        new_post_shape: tuple = _check_shapes(post_shape, new_post_shape)
        # Then, we test any axes before the distributed axis...
        pre_shape = self.local_shape[: (self.axis + 1)]
        new_pre_shape: list = local_shape[: (new_axis + 1)]
        new_pre_shape: tuple = _check_shapes(pre_shape, new_pre_shape)

        # Construct the new fleshed out local array shape
        local_shape = new_pre_shape[:new_axis] + new_post_shape

        # Now we actually try the resize...
        new_data = self.local_array.reshape(local_shape, order=order)

        # ...and then construct the final MPIArray object
        return self.__class__._view_from_data_and_params(
            new_data,
            new_axis,
            self.global_shape[self.axis],
            self._local_offset[self.axis],
            self.comm,
        )

    def _prep_buf(self, x: npt.NDArray) -> tuple[np.ndarray, MPI.Datatype]:
        """Prepare a buffer for sending/recv by mpi4py.

        If this is array is a supported datatype it just returns a simple buffer spec,
        if not, it will create a view of the underlying bytes, and return that alongside
        an MPI.BYTE datatype.

        Parameters
        ----------
        x : ndarray
            Array to prepare.

        Returns
        -------
        buffer : ndarray
            The prepared buffer.
        dtype : MPI.Datatype
            MPI datatype.
        """
        try:
            mpi_dtype = mpitools.typemap(self.dtype)
            return (x, mpi_dtype)
        except KeyError:
            return (
                x.view(np.byte).reshape((*x.shape, self.itemsize)),
                mpitools.MPI.BYTE,
            )

    def gather(self, rank: int = 0) -> np.ndarray | None:
        """Gather a full copy onto a specific rank.

        Parameters
        ----------
        rank : int
            Rank to gather onto. Default is 0.

        Returns
        -------
        gathered_array : ndarray | None
            The full global array on the specified rank. Returns
            :py:obj:`None` on all other ranks.
        """
        if self.comm.rank == rank:
            arr = np.ndarray(self.global_shape, dtype=self.dtype)
        else:
            arr = None

        splits = mpitools.split_all(self.global_shape[self.axis], self.comm)

        for ri, (n, s, e) in enumerate(zip(*splits)):
            if self.comm.rank == rank:
                # Construct a temporary array for the data to be received into
                tshape = list(self.global_shape)
                tshape[self.axis] = n
                tbuf = np.empty(tshape, dtype=self.dtype)

                # Set up the non-blocking receive request
                request = self.comm.Irecv(self._prep_buf(tbuf), source=ri)

            # Send the data
            if self.comm.rank == ri:
                request = self.comm.Isend(self._prep_buf(self.local_array), dest=rank)

                # Wait until the data has been sent
                stat = mpitools.MPI.Status()
                request.Wait(status=stat)

                if stat.error != mpitools.MPI.SUCCESS:
                    raise RuntimeError(
                        f"**** ERROR in MPI SEND (source: {ri},  dest rank: {rank}) *****"
                    )

            if self.comm.rank == rank:
                # Wait until the data has arrived
                stat = mpitools.MPI.Status()
                request.Wait(status=stat)

                if stat.error != mpitools.MPI.SUCCESS:
                    raise RuntimeError(
                        f"**** ERROR in MPI RECV (source: {ri},  dest rank: {rank}) *****"
                    )

                # Put the data into the correct location
                dest_slice = [slice(None)] * len(self.shape)
                dest_slice[self.axis] = slice(s, e)
                arr[tuple(dest_slice)] = tbuf

        return arr

    def allgather(self) -> np.ndarray:
        """Gather a full copy onto each rank.

        Returns
        -------
        gathered_array : ndarray
            The full global array gathered into a :py:class:`~np.ndarray`
            on all ranks.
        """
        arr = np.empty(self.global_shape, dtype=self.dtype)

        splits = mpitools.split_all(self.global_shape[self.axis], self.comm)

        for ri, (n, s, e) in enumerate(zip(*splits)):
            # Construct a temporary array for the data to be received into
            tshape = list(self.global_shape)
            tshape[self.axis] = n
            tbuf = np.ndarray(tshape, dtype=self.dtype)

            if self.comm.rank == ri:
                tbuf[:] = self.local_array

            self.comm.Bcast(self._prep_buf(tbuf), root=ri)

            # Copy the array into the correct place
            dest_slice = [slice(None)] * len(self.shape)
            dest_slice[self.axis] = slice(s, e)
            arr[tuple(dest_slice)] = tbuf

        return arr

    def copy(self, *args: Any, **kwargs: Any) -> MPIArray:
        """Return a copy of the MPIArray.

        Arguments are passed along to :py:meth:`~np.ndarray.copy`.

        Returns
        -------
        array_copy : MPIArray
            Copy of the array based on args passed to :py:meth:`~np.ndarray.copy`.
        """
        return MPIArray.wrap(
            self.view(np.ndarray).copy(*args, **kwargs), axis=self.axis, comm=self.comm
        )

    def ravel(self, *args, **kwargs) -> None:
        """Method is explicitly not implemented.

        This method would return a flattened view of the entire
        array, which is not supported across the distributed axis.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "Method 'ravel' is not implemented for distributed arrays. "
            "Try using '.local_array'."
        )

    def _to_hdf5_serial(
        self, filename: str, dataset: str, create: bool = False
    ) -> None:
        """Write into an HDF5 dataset.

        This explicitly serialises the IO so that it works when h5py does not
        support MPI-IO.

        .. warning::
            This method is intended as a fallback to get around certain bugs
            in ``HDF5``. Don't use it unless you **really** need to.

        Parameters
        ----------
        filename : str
            File to write dataset into.
        dataset : str
            Name of dataset to write into.
        create : bool, optional
            True if a new file should be created (if needed).
            Default is False.
        """
        import os
        import time

        from .memdata._io import open_h5py_mpi

        if self.comm is None or self.comm.rank == 0:
            with open_h5py_mpi(filename, "a" if create else "r+", use_mpi=False) as fh:
                dset = _create_or_get_dset(
                    fh,
                    dataset,
                    self.global_shape,
                    dtype=self.dtype,
                )
                dset[:] = np.array(0.0).astype(self.dtype)

        # wait until all processes see the created file
        while not os.path.exists(filename):
            time.sleep(1)

        self.comm.Barrier()

        if self.axis == 0:
            dist_arr = self
        else:
            dist_arr = self.redistribute(axis=0)

        size = 1 if self.comm is None else self.comm.size
        for ri in range(size):
            rank = 0 if self.comm is None else self.comm.rank
            if ri == rank:
                with open_h5py_mpi(filename, "r+", use_mpi=False) as fh:
                    start = dist_arr.local_offset[0]
                    end = start + dist_arr.local_shape[0]

                    fh[dataset][start:end] = dist_arr

            dist_arr.comm.Barrier()

    def _partition_io(
        self, skip: bool = False, threshold: float = 1.99
    ) -> tuple[int, list[slice]]:
        """Split IO of this array into local sections under `threshold`.

        Parameters
        ----------
        skip : bool, optional
            Don't partition, just find and return a full axis.
            Default is False.
        threshold : float, optional
            Maximum size of IO (in GB). Default is 1.99.

        Returns
        -------
        split_axis : int
            Which axis are we going to split along.
        partitions : list[slice]
            List of partition slices.
        """
        threshold_bytes = threshold * 2**30
        largest_size = self.comm.allreduce(self.nbytes, op=mpitools.MAX)
        min_axis_size = int(np.ceil(largest_size / threshold_bytes))

        # Return early if we can
        if skip or min_axis_size == 1:
            return 0, [slice(0, self.local_shape[0])]

        if self.ndim == 1:
            raise RuntimeError("To parition an array we must have multiple axes.")

        # Try and find the axis to split over
        # TODO: we should be able to be clever and split over the distributed axis
        for split_axis in range(self.ndim):
            if (
                split_axis != self.axis
                and self.global_shape[split_axis] >= min_axis_size
            ):
                break
        else:
            raise RuntimeError(
                f"Can't identify an IO partition less than {threshold:.2f} GB in size: "
                f"shape={self.global_shape!s}, distributed axis={self.axis}"
            )

        axis_length = self.global_shape[split_axis]
        slice_length = axis_length // min_axis_size
        boundaries = list(range(0, axis_length + 1, slice_length))

        # Add the axis length at the end (if required) so we can can construct the start
        # end pairs
        if boundaries[-1] != axis_length:
            boundaries.append(axis_length)

        # Construct the set of slices to apply
        slices = [slice(s, e) for s, e in pairwise(boundaries)]

        logger.debug(f"Splitting along axis {split_axis}, {len(slices)} ways")

        return split_axis, slices

    # array_ufunc is a special general function
    # which facilitates the use of a diverse set of ufuncs
    # some which return nothing, and some which return something

    def __array_ufunc__(  # noqa: D417
        self,
        ufunc: np.ufunc,
        method: Literal["__call__", "reduce", "accumulate"],
        *inputs: tuple[npt.NDArray],
        **kwargs: Any,
    ) -> MPIArray | tuple[MPIArray, ...] | None:
        r"""Handles ufunc operations for MPIArray.

        In NumPy, ufuncs are the various fundamental operations applied to
        ndarrays in an element-by-element fashion, such as add() and divide().
        https://numpy.org/doc/stable/reference/ufuncs.html

        ndarray has lots of built-in ufuncs. In order to use them, the MPIArrays
        need to be converted into ndarrays, otherwise NumPy reports a
        NotImplemented error.

        Not all ufunc methods make sense when operating over distributed inputs. The
        "__call__", "reduce", and "accumulate" types are supported, but the "outer",
        "at" and "reduceat" types have more complex behaviour that can't be easily
        generalised to a distributed case so they are not supported.

        Each of the supported methods has slightly different restrictions:

        - "__call__": for single array operations there are no restrictions (e.g.
          `np.exp`), for binary operations (e.g. `A + B`) both operands must share
          broadcast against each other and have the same shape distributed axis, which
          must be at a position consistent with the broadcasting.
        - "accumulate": the accumulation axis must not be the distributed axis,
          otherwise there are no restrictions (e.g. `np.cumsum`).
        - "reduce": the reduction cannot take place along the distributed axis (i.e.
          `axis` must not include the distributed axis), *except* for a total reduction
          over all axes (`axis = None`) which will return an array with a single element
          on each rank.

        For all of these method types, `out` keyword arguments, used for directly placing
        the result in an existing array, are supported. However `where` arguments, used
        for selecting which elements to apply the function to, are not.

        If the operation that you want to do is not supported then you should use
        numpy arrays directly to achieve what you want, first using
        :py:attr:`~.MPIArray.local_array` to get the local numpy data, then applying the ufunc, and
        finally using an :py:meth:`~.MPIArray.wrap` call to construct a distributed array from the
        output.

        Parameters
        ----------
        ufunc : ufunc
            ufunc object to call
        method : {"__call__", "accumulate", "reduce"}
            indicates which ufunc method was called. Only the methods "__call__",
            "accumulate" and "reduce" are supported. The other defined methods "outer",
            "reduceat" and "at" will cause a ValueError to be thrown.
        inputs : tuple[npt.NDArray]
            Sequence of the input arguments to the ufunc.  At least one of the inputs is an
            MPIArray.
        \**kwargs : Any
            Optional input arguments of the ufunc.  Important kwargs considered
            here are ``out`` and ``axis``.

        Returns
        -------
        result : MPIArray | tuple[MPIArray, ...] | None
            Result of the ufunc operation.

        Raises
        ------
        UnsupportedOperation

            - If the ufunc method is not supported for MPIArrays. Some more complicated
              methods are not possible without custom implementations.
            - If the `where` argument is provided.

        TypeError
            If at least one output is not an MPIArray.
        ValueError
            If the distribution of at least one output does not match expectation.
        """
        # Each ufunc application method must have a corresponding function that both
        # validates the inputs and arguments are appropriate (same distributed axis
        # etc.), and infers and returns the parameters of the output distributed axis
        # (i.e. it's position, length and the offset on this rank)

        _validation_fn = {
            "__call__": self._array_ufunc_call,
            "accumulate": self._array_ufunc_accumulate,
            "reduce": self._array_ufunc_reduce,
        }.get(method, None)

        if _validation_fn is None:
            raise UnsupportedOperation(
                f"ufunc method type '{method}' is not supported for MPIArrays. Try "
                "using `.local_array` to convert to a pure numpy array first, and the "
                "use `MPIArray.wrap` to reconstruct an MPIArray at the end."
            )
        # Where arguments are not supported, except where the value is a simple `True`,
        # which is used as a default
        if "where" in kwargs and not (
            isinstance(kwargs["where"], bool) and kwargs["where"]
        ):
            raise UnsupportedOperation(
                "MPIArray ufunc calls do not support 'where' arguments, but "
                f"where={kwargs['where']} was found. Try using numpy arrays directly "
                "if you really need to use a 'where' argument."
            )

        comm = _get_common_comm(inputs)

        if comm is None:
            raise RuntimeError(f"Could not get common MPI Comm for inputs {inputs}.")

        new_dist_axis, global_length, offset = _validation_fn(ufunc, *inputs, **kwargs)

        # Check that all out arguments are valid MPIArrays, and then convert into numpy
        # arrays
        if (out_args := kwargs.get("out")) is not None:
            if not all(isinstance(x, MPIArray) for x in out_args):
                raise TypeError(
                    "At least one output is not an MPIArray. This can happen if a ufunc "
                    "is trying to modify a np.ndarray in-place using values from a MPIArray. "
                    "Try using .local_array or cast the np.ndarray to MPIArray."
                )
            kwargs["out"] = _mpi_to_ndarray(out_args, only_mpiarray=True)

            # Check the distribution of the output arguments makes sense
            for ii, array in enumerate(out_args):
                try:
                    _check_dist_axis(array, new_dist_axis, global_length, offset, comm)
                except (ValueError, AxisException) as e:
                    raise ValueError(
                        f"Output argument at position {ii} does not match expectation."
                    ) from e
        else:
            out_args = [None] * ufunc.nout

        results = _ensure_list(
            super().__array_ufunc__(ufunc, method, *_mpi_to_ndarray(inputs), **kwargs)
        )

        # Convert any outputs back into valid MPIArrays if required
        ret = []
        for res, out in zip(results, out_args):
            # If out is None a new *numpy* array was initialised for the results. We
            # need to turn this into an MPIArray using the inferrred distribution
            if out is None:
                # Result is a scalar, so we want to return this as a 1D distributed
                # vector
                if not res.shape:
                    res = res.reshape(1)

                # Convert into an an MPIArray
                out = MPIArray._view_from_data_and_params(
                    res, new_dist_axis, global_length, offset, comm
                )

            ret.append(out)

        return ret[0] if len(ret) == 1 else tuple(ret)

    def _array_ufunc_call(
        self, ufunc: np.ufunc, *inputs: tuple[npt.ArrayLike, ...], **kwargs: Any
    ) -> tuple[int, int, int]:
        """Validate and infer the final distributed axis and shape for a standard ufunc call."""
        # For a 'call' the highest dimensional array should determine the dimensionality
        # of the output, so we find that
        max_dim_input = inputs[
            np.argmax(
                [inp.ndim if isinstance(inp, np.ndarray) else -1 for inp in inputs]
            )
        ]
        max_dim = max_dim_input.ndim

        # Find the first MPIArray in the argument list and use this to determine the
        # distributed axis location
        try:
            first_mpi_array = next(inp for inp in inputs if isinstance(inp, MPIArray))
        except StopIteration as exc:
            raise TypeError(
                "MPIArray ufunc didn't get any MPIArray inputs. This probably "
                "means that the ufunc was called with the `out` argument assigned "
                " to a MPIArray."
            ) from exc

        axis = max_dim + first_mpi_array.axis - first_mpi_array.ndim
        length = first_mpi_array.global_shape[first_mpi_array.axis]
        offset = first_mpi_array.local_offset[first_mpi_array.axis]
        local_length = first_mpi_array.local_shape[first_mpi_array.axis]

        # Check that all input arguments have a consistently located distributed axis,
        # while accounting for the fact that numpy will left-pad with length-1
        # dimensions when broadcasting
        for ii, inp in enumerate(inputs):
            if not isinstance(inp, np.ndarray):
                continue

            # Where should the distributed axis lie in this array (even if it's not an
            # MPIArray)
            cur_dist_axis = axis - max_dim + inp.ndim

            if isinstance(inp, MPIArray):
                try:
                    _check_dist_axis(inp, cur_dist_axis, length, offset)
                except (ValueError, AxisException) as e:
                    raise AxisException(
                        f"Input argument {ii} has an incompatible distributed axis."
                    ) from e
            elif cur_dist_axis >= 0:
                cur_axis_length = inp.shape[cur_dist_axis]

                if cur_axis_length == 1:
                    # Can broadcast. Great!
                    continue

                if cur_axis_length == local_length:
                    warnings.warn(
                        "A ufunc is combining an MPIArray with a numpy array that "
                        "matches exactly the local part of the distributed axis. This "
                        "is very fragile and may fail on other ranks. "
                        f"Numpy array shape: {inp.shape}; dist axis: {cur_dist_axis}.",
                        stacklevel=3,
                    )
                elif cur_axis_length != 1:
                    raise AxisException(
                        f"Input argument {ii} is a numpy array, but has shape != 1 on "
                        f"the distributed axis (length={cur_axis_length})."
                    )

        return axis, length, offset

    def _array_ufunc_accumulate(
        self, ufunc: np.ufunc, input_: npt.NDArray, *, axis: int, **kwargs: dict
    ) -> tuple[int, 3]:
        """Validate and infer the final distributed axis for a ufunc accumulation."""
        # An accumulation should give an array the same shape as the input.
        if axis == input_.axis:
            raise AxisException(
                f"Can not accumulate over the distributed axis (axis={input_.axis})"
            )

        return (
            input_.axis,
            input_.global_shape[input_.axis],
            input_.local_offset[input_.axis],
        )

    def _array_ufunc_reduce(
        self,
        ufunc: np.ufunc,
        input_: npt.ArrayLike,
        *,
        axis: int,
        keepdims: bool = False,
        **kwargs: dict,
    ) -> tuple[int, int, int]:
        """Validate and infer the final distributed axis for a ufunc reduction."""
        # If we are doing a complete reduction (indicated by axis=None), the output will
        # be a vector with one element per rank. Directly return parameters to represent
        # that.
        if axis is None:
            return (0, input_.comm.size, input_.comm.rank)

        # Get the normalised set of axes to reduce over
        axis: set = {ax if ax >= 0 else input_.ndim + ax for ax in _ensure_list(axis)}

        if input_.axis in axis:
            raise AxisException(
                f"Can not reduce over the distributed axis (axis={input_.axis})"
            )

        # Calculate the final position of the distributed axis
        final_dist_axis = input_.axis
        for ii, s in enumerate(input_.shape):
            if ii in axis and ii < input_.axis and not keepdims:
                final_dist_axis -= 1

        return (
            final_dist_axis,
            input_.global_shape[input_.axis],
            input_.local_offset[input_.axis],
        )

    def __array_finalize__(self: MPIArray, obj: MPIArray | None) -> None:
        """Finalizes the creation of the MPIArray, when viewed.

        Note: If you wish to create an MPIArray from an ndarray, please use wrap().
        Do not use ndarray.view(MPIArray).

        In NumPy, ndarrays only go through the `__new__` when being instantiated.
        For views and broadcast, they go through __array_finalize__.
        https://numpy.org/doc/stable/user/basics.subclassing.html#the-role-of-array-finalize

        Parameters
        ----------
        self : npt.NDArray
            The array which will be created
        obj : npt.NDArray | None
            The original array being viewed or broadcast.
            When in the middle of a constructor, obj is set to None.
        """
        if obj is None:
            # we are in the middle of a constructor, and the attributes
            # will be set when we return to it
            return

        if not isinstance(obj, MPIArray):
            # in the middle of an np.ndarray.view() in the wrap()
            return

        # we are in a slice, rebuild the attributes from the original MPIArray
        comm = getattr(obj, "comm", mpitools.world)

        axis = obj.axis

        # Get local shape
        lshape = self.shape
        global_shape = list(lshape)

        # Obtaining length of distributed axis, without using an mpi.allreduce
        try:
            axlen = obj.global_shape[axis]
        except IndexError as e:
            raise AxisException(
                f"Distributed axis {axis} does not exist in global shape {global_shape}"
            ) from e

        global_shape[axis] = axlen

        # Get offset
        _, local_start, _ = mpitools.split_local(axlen, comm=comm)

        loffset = [0] * len(lshape)
        loffset[axis] = local_start

        # Setup attributes
        self._global_shape = tuple(global_shape)
        self._axis = axis
        self._local_shape = tuple(lshape)
        self._local_offset = tuple(loffset)
        self._comm = comm


def zeros(*args: Any, **kwargs: Any) -> MPIArray:
    """Generate an MPIArray filled with zeros.

    Parameters
    ----------
    args, kwargs : Any
        Arguments passed straight through to the :py:class:`.MPIArray` constructor.

    Returns
    -------
    array_of_zeros : MPIArray
        :py:class:`.MPIArray` instance filled with zeros.
    """
    arr = MPIArray(*args, **kwargs)
    arr[:] = 0

    return arr


def ones(*args: Any, **kwargs: Any) -> MPIArray:
    """Generate an MPIArray filled with ones.

    Parameters
    ----------
    args, kwargs : Any
        Arguments passed straight through to the :py:class:`.MPIArray` constructor.

    Returns
    -------
    array_of_ones : MPIArray
        :py:class:`.MPIArray` instance filled with ones.
    """
    arr = MPIArray(*args, **kwargs)
    arr[:] = 1

    return arr


def _partition_sel(
    sel: tuple[slice, ...],
    split_axis: int,
    n: int,
    slice_: slice,
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    """Re-slice a selection along a new axis.

    Take a selection (a tuple of slices) and re-slice along the split_axis (which has
    length n).

    Parameters
    ----------
    sel : tuple[slice, ...]
        Selection
    split_axis : int
        New split axis
    n : int
        Length of split axis
    slice_ : slice
        New slice along new axis

    Returns
    -------
    initial_selection : list
        Pre-selection selection
    post_selection : list
        Post-selection selection
    """
    # Reconstruct the slice for the split axis
    slice_init = _reslice(sel[split_axis], n, slice_)

    # Construct the final selection
    sel_final = [slice(None)] * len(sel)
    sel_final[split_axis] = slice_

    # Construct the initial selection
    sel_initial = list(sel)
    sel_initial[split_axis] = slice_init

    return tuple(sel_initial), tuple(sel_final)


def _len_slice(slice_: slice, n: int) -> int:
    """Calculate the output length of a slice applied to an axis of length n."""
    if isinstance(slice_, slice):
        start, stop, step = slice_.indices(n)
        return 1 + (stop - start - 1) // step

    return len(slice_)


def _ensure_list(x: Sequence[Any] | Any) -> list[Any]:
    """Guarantee the output is a list.

    Turns any scalars into a single-element list and turns sets or tuples into a list.
    """
    if isinstance(x, list | tuple | set):
        return list(x)

    return [x]


def _reslice(
    slice_: slice | tuple | list, n: int, subslice: slice
) -> slice | list | tuple | int:
    """Find a single slice with the same affect as two successive slices.

    For a slice along an axis of length n, return the slice that would select the
    slice(start, end) elements of the final array.
    """
    if subslice.step is not None and subslice.step > 1:
        raise ValueError(f"stride > 1 not supported. subslice: {subslice}")

    if isinstance(slice_, slice):
        dstart, dstop, dstep = slice_.indices(n)

        return slice(
            dstart + subslice.start * dstep,
            min(dstart + subslice.stop * dstep, dstop),
            dstep,
        )

    return slice_[subslice]


def _expand_sel(sel: SelectionLike | None, naxis: int) -> list[slice | int]:
    """Expand the selection to the full dimensions."""
    if sel is None:
        sel = [slice(None)] * naxis
    if len(sel) < naxis:
        sel = list(sel) + [slice(None)] * (naxis - len(sel))
    return list(sel)


def _apply_sel(
    arr: npt.ArrayLike,
    sel: slice | tuple[int, ...] | list[int],
    ax: int,
) -> npt.ArrayLike:
    """Apply a selection to a single axis of an array."""
    if type(sel) is slice:
        sel = (slice(None),) * ax + (sel,)
        return arr[sel]

    if type(sel) in {list, tuple}:
        return np.take(arr, sel, axis=ax)

    raise ValueError(
        "Invalid selection type. Selection must be one of (slice, tuple, list). "
        f"Got {type(sel)}."
    )


def _check_dist_axis(
    array: MPIArray,
    dist_axis: int,
    global_length: int,
    offset: int,
    comm: MPI.Comm | None = None,
) -> None:
    if comm and array.comm != comm:
        raise ValueError(
            "MPIArray not distributed over expected communicator. "
            f"Expected {comm}, got {array.comm}."
        )

    if array.axis != dist_axis:
        raise AxisException(
            "provided MPIArray's distributed axis is not consistent "
            f"with the expected distributed axis. Expected {dist_axis}; "
            f"Actual {array.axis}"
        )

    arr_length = array.global_shape[dist_axis]
    if arr_length != global_length:
        raise AxisException(
            "output argument distributed axis length does not match expected. "
            f"Expected {global_length}, actual {arr_length}."
        )

    arr_offset = array.local_offset[dist_axis]
    if arr_offset != offset:
        raise AxisException(
            "output argument distributed axis offset does not match expected. "
            f"Expected {offset}, actual {arr_offset}."
        )


def _get_common_comm(inputs: Sequence[npt.ArrayLike | None]) -> MPI.Comm | None:
    """Get a common MPI communicator from a set of arguments.

    Parameters
    ----------
    inputs : Sequence[array_like | None]
        A mixture of MPIArrays, numpy arrays, scalars and None types.

    Returns
    -------
    common_comm : MPI.Comm
        The communicator shared by all MPIArray arguments. Throws an exception if there
        is no common communicator, returns None if no MPIArrays are present.
    """
    comm = None
    for array in inputs:
        if isinstance(array, MPIArray):
            if comm is None:
                comm = array.comm
            else:
                if comm != array.comm:
                    raise ValueError(
                        "The communicator should be the same for all MPIArrays."
                    )
    return comm


def _mpi_to_ndarray(
    inputs: Sequence[npt.ArrayLike | None],
    only_mpiarray: bool = False,
) -> tuple[np.ndarray]:
    """Ensure a list with mixed MPIArrays and ndarrays are all ndarrays.

    Additionally, ensure that all of the MPIArrays are distributed along the same axis.

    Parameters
    ----------
    inputs : Sequence[array_like | None]
        All MPIArrays should be distributed along the same axis.
    only_mpiarray : bool, optional
        Throw an error if we received anything other than an MPIArray or None.
        Default is False

    Returns
    -------
    arrays : ndarray
        The ndarrays built from the local view of inputed MPIArrays.
    """
    args = []

    for array in inputs:
        if isinstance(array, MPIArray):
            if not hasattr(array, "axis"):
                raise AxisException(
                    "An input to a ufunc has an MPIArray, which is missing its axis "
                    "property. If using a lower-case MPI.Comm function, please use "
                    "its upper-case alternative. Pickling does not preserve the axis "
                    "property. Otherwise, please file an issue with a stacktrace."
                )
            args.append(array.local_array)
        elif array is None or not only_mpiarray:
            args.append(array)
        else:
            raise ValueError("Only MPIArrays or None values are allowed.")

    return tuple(args)


def _create_or_get_dset(
    group: h5py.Group | zarr.Group,
    name: str,
    shape: tuple[int, ...] | list[int],
    dtype: npt.DTypeLike,
    **kwargs: Any,
) -> h5py.Dataset | zarr.Array:
    """Create a dataset if it doesn't exist, or test the existing one for compatibility."""
    if name in group:
        dset = group[name]
        if dset.shape != shape:
            raise RuntimeError(
                "Dataset exists already but with incompatible shape."
                f"Requested shape={shape}, but on disk shape={dset.shape}."
            )
        if dset.dtype != dtype:
            raise RuntimeError(
                "Dataset exists already but with incompatible dtype. "
                f"Requested dtype={dtype}, on disk dtype={dset.dtype}."
            )
    else:
        dset = group.create_dataset(
            name,
            shape=shape,
            dtype=dtype,
            **kwargs,
        )
    return dset


def sanitize_slice(
    sl: SelectionTupleLike, naxis: int
) -> tuple[tuple[slice | int], tuple[int, ...], tuple[int, ...]]:
    """Sanitize and extract information from the arguments to an array indexing.

    Parameters
    ----------
    sl : SelectionTupleLike
        A tuple representing the arguments to array indexing.
    naxis : int
        The total number of axes in the array being indexed.

    Returns
    -------
    new_slice
        An equivalent slice that has been sanitized, i.e. ellipsis expanded, all indices
        cast to ints etc.
    axis_map
        For each axis in the input array, give the position of that axis in the
        sanitized slice.
    final_map
        For each position in the `sl`, give the axis position it corresponds to in the
        final array.

    Raises
    ------
    IndexError
        For incompatible slices, such as too many axes.
    """
    orig_sl = sl

    # Add an Ellipsis at the very end if it isn't present elsewhere
    num_ellipsis = sum([s is Ellipsis for s in sl])
    if num_ellipsis > 1:
        raise IndexError(f"Found more than one Ellipsis in slice {sl}")

    if num_ellipsis == 0:
        sl += (Ellipsis,)

    # Convert all np.int types (which are valid arguments) to ints
    sl = tuple(int(s) if isinstance(s, np.integer) else s for s in sl)

    num_added = 0
    num_removed = 0
    ell_ind = None

    for ii, s in enumerate(sl):
        if s is np.newaxis:
            num_added += 1
        elif isinstance(s, int):
            num_removed += 1
        elif s is Ellipsis:
            ell_ind = ii

    # Calculate the number of slices to add instead of the ellipsis
    ell_length = naxis - len(sl) + num_added + 1

    if ell_length < 0:
        raise IndexError(
            f"slice={orig_sl} passed into MPIArray with {naxis} dims is too long."
        )

    # Expand Ellipsis
    sl = sl[:ell_ind] + tuple([slice(None)] * ell_length) + sl[(ell_ind + 1) :]

    axis_map = []

    final_positions = []
    output_ind = 0

    # Extract the axis mappings
    for ii, s in enumerate(sl):
        # Any slice entry that's not a newaxis should map to an axis in the original
        # array
        if s is not np.newaxis:
            axis_map.append(ii)

        # Exact integer indices will remove an axis from the final output
        if isinstance(s, int):
            final_positions.append(None)
        else:
            final_positions.append(output_ind)
            output_ind += 1

    if len(axis_map) > naxis:
        # Shouldn't actually hit this issue
        raise IndexError(
            f"slice={orig_sl} passed into MPIArray with {naxis} dims is too long."
        )

    return sl, tuple(axis_map), tuple(final_positions)


class AxisException(ValueError):
    """Exception for distributed axes related errors with MPIArrays."""

    def __init__(self, message: str) -> None:
        self.message: str = message
        super().__init__(self.message)


class UnsupportedOperation(ValueError):
    """Exception for when an operation cannot be performed with an MPIArray."""

    def __init__(self, message: str) -> None:
        self.message: str = message
        super().__init__(self.message)
