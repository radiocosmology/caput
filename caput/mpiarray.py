"""
An array class for containing MPI distributed array.

.. currentmodule:: caput.mpiarray

Classes
=======

.. autosummary::
    :toctree: generated/

    MPIArray

Examples
========

This example performs a transfrom from time-freq to lag-m space. This involves
Fourier transforming each of these two axes of the distributed array::

    import numpy as np
    from mpi4py import MPI

    from caput.mpiarray import MPIArray

    nfreq = 32
    nprod = 2
    ntime = 32

    # Initalise array with (nfreq, nprod, ntime) global shape
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

Global Slicing
==============

The :class:`MPIArray` also supports slicing with the global index using the
:py:attr:`~MPIArray.global_slice` property. This can be used for both fetching
and assignment with global indices, supporting the basic slicing notation of
`numpy`.

Its behaviour changes depending on the exact slice it gets:

- A full slice (`:`) along the parallel axis returns an :class:`MPIArray` on
  fetching, and accepts an :class:`MPIArray` on assignment.
- A partial slice (`:`) returns and accepts a numpy array on the rank holding
  the data, and :obj:`None` on other ranks.

It's important to note that it never communicates data between ranks. It only
ever operates on data held on the current rank.

Example
-------

Here is an example of this in action::

    import numpy as np
    from caput import mpiarray, mpiutil

    arr = mpiarray.MPIArray((mpiutil.size, 3), dtype=np.float64)
    arr[:] = 0.0

    for ri in range(mpiutil.size):
        if ri == mpiutil.rank:
            print(ri, arr)
        mpiutil.barrier()

    # Use a global index to assign to the array
    arr.global_slice[3] = 17

    # Fetch a view of the whole array with a full slice
    arr2 = arr.global_slice[:, 2]

    # This should be the third column of the array
    for ri in range(mpiutil.size):
        if ri == mpiutil.rank:
            print(ri, arr2)
        mpiutil.barrier()

    # Fetch a view of the whole array with a partial slice
    arr3 = arr.global_slice[:2, 2]

    # The final two ranks should be None
    for ri in range(mpiutil.size):
        if ri == mpiutil.rank:
            print(ri, arr3)
        mpiutil.barrier()

"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

from past.builtins import basestring
import os
import time
import logging

import numpy as np

from caput import mpiutil, misc


logger = logging.getLogger(__name__)


class _global_resolver(object):
    # Private class implementing the global sampling for MPIArray

    def __init__(self, array):

        self.array = array
        self.axis = array.axis
        self.offset = array.local_offset[self.axis]
        self.length = array.global_shape[self.axis]

    def _resolve_slice(self, slobj):
        # Transforms a numpy basic slice on the global arrays into a fully
        # fleshed out slice tuple referencing the positions in the local arrays.
        # If a single integer index is specified for the distributed axis, then
        # either the local index is returned, or None if it doesn't exist on the
        # current rank.

        ndim = self.array.ndim
        local_length = self.array.shape[self.axis]

        # Expand a single integer or slice index
        if isinstance(slobj, int) or isinstance(slobj, slice):
            slobj = (slobj, Ellipsis)

        # Add an ellipsis if length of slice object is too short
        if isinstance(slobj, tuple) and len(slobj) < ndim and Ellipsis not in slobj:
            slobj = slobj + (Ellipsis,)

        # Expand an ellipsis
        slice_list = []
        for sl in slobj:
            if sl is Ellipsis:
                for i in range(ndim - len(slobj) + 1):
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

            # If step is defined we don't need to adjust this, but it's no longer a complete slice
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

    def __getitem__(self, slobj):

        # Resolve the slice object
        slobj, is_fullslice = self._resolve_slice(slobj)

        # If not a full slice, return a numpy array (or None)
        if not is_fullslice:

            # If the parallel axis has a None, that means there is no data on this rank
            if slobj[self.axis] is None:
                return None
            else:
                return self.array[slobj].view(np.ndarray)

        else:

            # Fix up slobj for axes where there is no data
            slobj = tuple(slice(None, None, None) if sl is None else sl for sl in slobj)

            # Return an MPIArray view
            arr = self.array[slobj]

            # Figure out which is the distributed axis after the slicing, by
            # removing slice axes which are just ints from the mapping
            dist_axis = [
                index for index, sl in enumerate(slobj) if not isinstance(sl, int)
            ].index(self.axis)

            return MPIArray.wrap(arr, axis=dist_axis, comm=self.array._comm)

    def __setitem__(self, slobj, value):

        slobj, is_fullslice = self._resolve_slice(slobj)

        if slobj[self.axis] is None:
            return
        self.array[slobj] = value


class MPIArray(np.ndarray):
    """A numpy array like object which is distributed across multiple processes.

    Parameters
    ----------
    global_shape : tuple
        The global array shape. The returned array will be distributed across
        the specified index.
    axis : integer, optional
        The dimension to distribute the array across.

    Attributes
    ----------
    global_shape : tuple
        Global array shape.
    local_shape : tuple
        Shape of local section.
    axis : integer
        Axis we are distributed over.
    local_offset : tuple
        Offset into global array. This is equivalent to the global-index of
        the [0, 0, ...] element of the local section.
    local_array : np.ndarray
        The view of the local numpy array.
    global_slice : object
        Return an objects that presents a view of the array with global slicing.

    Methods
    -------
    wrap
    redistribute
    enumerate
    from_hdf5
    to_hdf5
    transpose
    reshape
    """

    @property
    def global_shape(self):
        return self._global_shape

    @property
    def axis(self):
        return self._axis

    @property
    def local_shape(self):
        return self._local_shape

    @property
    def local_offset(self):
        return self._local_offset

    @property
    def local_array(self):
        return self.view(np.ndarray)

    @property
    def comm(self):
        return self._comm

    def __new__(cls, global_shape, axis=0, comm=None, *args, **kwargs):

        # if mpiutil.world is None:
        #     raise RuntimeError('There is no mpi4py installation. Aborting.')

        if comm is None:
            comm = mpiutil.world

        # Determine local section of distributed axis
        local_num, local_start, local_end = mpiutil.split_local(
            global_shape[axis], comm=comm
        )

        # Figure out the local shape and offset
        lshape = list(global_shape)
        lshape[axis] = local_num

        loffset = [0] * len(global_shape)
        loffset[axis] = local_start

        # Create array
        arr = np.ndarray.__new__(cls, lshape, *args, **kwargs)

        # Set attributes of class
        arr._global_shape = global_shape
        arr._axis = axis
        arr._local_shape = tuple(lshape)
        arr._local_offset = tuple(loffset)
        arr._comm = comm

        return arr

    @property
    def global_slice(self):
        return _global_resolver(self)

    @classmethod
    def wrap(cls, array, axis, comm=None):
        """Turn a set of numpy arrays into a distributed MPIArray object.

        This is needed for functions such as `np.fft.fft` which always return
        an `np.ndarray`.

        Parameters
        ----------
        array : np.ndarray
            Array to wrap.
        axis : integer
            Axis over which the array is distributed. The lengths are checked
            to try and ensure this is correct.
        comm : MPI.Comm, optional
            The communicator over which the array is distributed. If `None`
            (default), use `MPI.COMM_WORLD`.

        Returns
        -------
        dist_array : MPIArray
            An MPIArray view of the input.
        """

        # from mpi4py import MPI

        if comm is None:
            comm = mpiutil.world

        # Get axis length, both locally, and globally
        axlen = array.shape[axis]
        totallen = mpiutil.allreduce(axlen, comm=comm)

        # Figure out what the distributed layout should be
        local_num, local_start, local_end = mpiutil.split_local(totallen, comm=comm)

        # Check the local layout is consistent with what we expect, and send
        # result to all ranks
        layout_issue = mpiutil.allreduce(axlen != local_num, op=mpiutil.MAX, comm=comm)

        if layout_issue:
            raise Exception("Cannot wrap, distributed axis local length is incorrect.")

        # Set shape and offset
        lshape = array.shape
        global_shape = list(lshape)
        global_shape[axis] = totallen

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

    def redistribute(self, axis):
        """Change the axis that the array is distributed over.

        Parameters
        ----------
        axis : integer
            Axis to distribute over.

        Returns
        -------
        array : MPIArray
            A new copy of the array distributed over the specified axis. Note
            that the local section will have changed.
        """

        # Check to see if this is the current distributed axis
        if self.axis == axis or self.comm is None:
            return self

        # Test to see if the datatype is one understood by MPI, this can
        # probably be fixed up at somepoint by creating a datatype of the right
        # number of bytes
        try:
            mpiutil.typemap(self.dtype)
        except KeyError:
            if self.comm.rank == 0:
                import warnings

                warnings.warn(
                    "Cannot redistribute array of compound datatypes." " Sorry!!"
                )
            return self

        # Get a view of the array
        arr = self.view(np.ndarray)

        if self.comm.size == 1:
            # only one process
            if arr.shape[self.axis] == self.global_shape[self.axis]:
                # We are working on a single node and being asked to do
                # a trivial transpose.
                trans_arr = arr.copy()

            else:
                raise ValueError(
                    "Global shape %s is incompatible with local arrays shape %s"
                    % (self.global_shape, self.shape)
                )
        else:
            pc, sc, ec = mpiutil.split_local(arr.shape[axis], comm=self.comm)
            par, sar, ear = mpiutil.split_all(
                self.global_shape[self.axis], comm=self.comm
            )
            pac, sac, eac = mpiutil.split_all(arr.shape[axis], comm=self.comm)

            new_shape = np.asarray(self.global_shape)
            new_shape[axis] = pc

            requests_send = []
            requests_recv = []

            trans_arr = np.empty(new_shape, dtype=arr.dtype)
            mpitype = mpiutil.typemap(arr.dtype)
            buffers = list()

            # Cut out the right blocks of the local array to send around
            blocks = np.array_split(arr, np.insert(eac, 0, sac[0]), axis)[1:]

            # Iterate over all processes row wise
            for ir in range(self.comm.size):

                # Iterate over all processes column wise
                for ic in range(self.comm.size):

                    # Construct a unique tag
                    tag = ir * self.comm.size + ic

                    # Send and receive the messages as non-blocking passes
                    if self.comm.rank == ir:
                        # Send the message
                        request = self.comm.Isend(
                            [blocks[ic].flatten(), mpitype], dest=ic, tag=tag
                        )

                        requests_send.append([ir, ic, request])

                    if self.comm.rank == ic:
                        buffer_shape = np.asarray(new_shape)
                        buffer_shape[axis] = eac[ic] - sac[ic]
                        buffer_shape[self.axis] = ear[ir] - sar[ir]
                        buffers.append(np.ndarray(buffer_shape, dtype=arr.dtype))

                        request = self.comm.Irecv(
                            [buffers[ir], mpitype], source=ir, tag=tag
                        )
                        requests_recv.append([ir, ic, request])

            # Wait for all processes to have started their messages
            self.comm.Barrier()

            # For each node iterate over all sends and wait until completion
            for ir, ic, request in requests_send:

                stat = mpiutil.MPI.Status()

                request.Wait(status=stat)

                if stat.error != mpiutil.MPI.SUCCESS:
                    logger.error(
                        "**** ERROR in MPI SEND (r: %i c: %i rank: %i) *****".format(
                            ir, ic, self.comm.rank
                        )
                    )

            self.comm.Barrier()

            # For each frequency iterate over all receives and wait until
            # completion
            for ir, ic, request in requests_recv:

                stat = mpiutil.MPI.Status()

                request.Wait(status=stat)

                if stat.error != mpiutil.MPI.SUCCESS:
                    logger.error(
                        "**** ERROR in MPI RECV (r: %i c: %i rank: %i) *****".format(
                            ir, ir, self.comm.rank
                        )
                    )

            # Put together the blocks we received
            np.concatenate(buffers, self.axis, trans_arr)

        # Create a new MPIArray object out of the data
        dist_arr = MPIArray(
            self.global_shape, axis=axis, dtype=self.dtype, comm=self.comm
        )
        dist_arr[:] = trans_arr

        return dist_arr

    def enumerate(self, axis):
        """Helper for enumerating over a given axis.

        Parameters
        ----------
        axis : integer
            Which access to enumerate over.

        Returns
        -------
        iterator : (local_index, global_index)
            An enumerator which returns the local index into the array *and*
            the global index it corresponds to.
        """
        start = self.local_offset[axis]
        end = start + self.local_shape[axis]

        return enumerate(range(start, end))

    @classmethod
    def from_hdf5(cls, f, dataset, comm=None, axis=0, sel=None):
        """Read MPIArray from an HDF5 dataset in parallel.

        Parameters
        ----------
        f : filename, or `h5py.File` object
            File to read dataset from.
        dataset : string
            Name of dataset to read from. Must exist.
        comm : MPI.Comm, optional
            MPI communicator to distribute over. If `None` optional, use
            `MPI.COMM_WORLD`.
        axis : int, optional
            Axis over which the read should be distributed. This can be used
            to select the most efficient axis for the reading.
        sel : tuple, optional
            A tuple of slice objects used to make a selection from the array
            *before* reading. The output will be this selection from the dataset
            distributed over the given axis.

        Returns
        -------
        array : MPIArray
        """
        # Don't both using MPI where the axis is not zero. It's probably just slower.
        # TODO: with tuning this might not be true. Keep an eye on this.
        use_mpi = axis > 0

        # Read the file. Opening with MPI if requested, and we can
        fh = misc.open_h5py_mpi(f, "r", use_mpi=use_mpi, comm=comm)

        dset = fh[dataset]
        dshape = dset.shape  # Shape of the underlying dataset
        naxis = len(dshape)
        dtype = dset.dtype

        # Check that the axis is valid and wrap to an actual position
        if axis < -naxis or axis >= naxis:
            raise ValueError(
                "Distributed axis %i not in range (%i, %i)" % (axis, -naxis, naxis - 1)
            )
        axis = naxis + axis if axis < 0 else axis

        # Ensure sel is defined to cover all axes
        sel = _expand_sel(sel, naxis)

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
        sel = tuple(sel)

        # Split the axis to get the IO size under ~2GB (only if MPI-IO)
        split_axis, partitions = dist_arr._partition_io(skip=(not fh.is_mpi))

        # Check that there are no null slices, otherwise we need to turn off
        # collective IO to work around an h5py issue (#965)
        no_null_slices = dist_arr.global_shape[axis] >= dist_arr.comm.size

        # Only use collective IO if:
        # - there are no null slices (h5py bug)
        # - we are not distributed over axis=0 as there is no advantage for
        #   collective IO which is usually slow
        # TODO: change if h5py bug fixed
        # TODO: better would be a test on contiguous IO size
        # TODO: do we need collective IO to read chunked data?
        use_collective = fh.is_mpi and no_null_slices and axis > 0

        # Read using collective MPI-IO if specified
        with dset.collective if use_collective else DummyContext():

            # Loop over partitions of the IO and perform them
            for part in partitions:
                islice, fslice = _partition_sel(
                    sel, split_axis, dshape[split_axis], part
                )
                dist_arr[fslice] = dset[islice]

        if fh.opened:
            fh.close()

        return dist_arr

    def to_hdf5(
        self,
        f,
        dataset,
        create=False,
        chunks=None,
        compression=None,
        compression_opts=None,
    ):
        """Parallel write into a contiguous HDF5 dataset.

        Parameters
        ----------
        filename : str, h5py.File or h5py.Group
            File to write dataset into.
        dataset : string
            Name of dataset to write into. Should not exist.
        """

        import h5py

        if not h5py.get_config().mpi:
            if isinstance(f, basestring):
                self._to_hdf5_serial(f, dataset, create)
                return
            else:
                raise ValueError(
                    "Argument must be a filename if h5py does not have MPI support"
                )

        mode = "a" if create else "r+"

        fh = misc.open_h5py_mpi(f, mode, self.comm)

        start = self.local_offset[self.axis]
        end = start + self.local_shape[self.axis]

        # Construct slices for axis
        sel = ([slice(None, None)] * self.axis) + [slice(start, end)]
        sel = _expand_sel(sel, self.ndim)

        # Check that there are no null slices, otherwise we need to turn off
        # collective IO to work around an h5py issue (#965)
        no_null_slices = self.global_shape[self.axis] >= self.comm.size

        # Split the axis to get the IO size under ~2GB (only if MPI-IO)
        split_axis, partitions = self._partition_io(skip=(not fh.is_mpi))

        # Only use collective IO if:
        # - there are no null slices (h5py bug)
        # - we are not distributed over axis=0 as there is no advantage for
        #   collective IO which is usually slow
        # - unless we want to use compression/chunking
        # TODO: change if h5py bug fixed
        # TODO: better would be a test on contiguous IO size
        use_collective = (
            fh.is_mpi and no_null_slices and (self.axis > 0 or compression is not None)
        )

        if fh.is_mpi and not use_collective:
            # Need to disable compression if we can't use collective IO
            chunks, compression, compression_opts = None, None, None

        dset = fh.create_dataset(
            dataset,
            shape=self.global_shape,
            dtype=self.dtype,
            chunks=chunks,
            compression=compression,
            compression_opts=compression_opts,
        )

        # Read using collective MPI-IO if specified
        with dset.collective if use_collective else DummyContext():

            # Loop over partitions of the IO and perform them
            for part in partitions:
                islice, fslice = _partition_sel(
                    sel, split_axis, self.global_shape[split_axis], part
                )
                dset[islice] = self[fslice]

        if fh.opened:
            fh.close()

    def transpose(self, axes):
        """Transpose the array axes.

        Parameters
        ----------
        axes : tuple
            Tuple of axes permutations.

        Returns
        -------
        array : MPIArray
            Transposed MPIArray as a view of the original data.
        """

        tdata = np.ndarray.transpose(self, axes)

        tdata._global_shape = tuple([self.global_shape[ax] for ax in axes])
        tdata._local_shape = tuple([self.local_shape[ax] for ax in axes])
        tdata._local_offset = tuple([self.local_offset[ax] for ax in axes])

        tdata._axis = list(axes).index(self.axis)
        tdata._comm = self._comm

        return tdata

    def reshape(self, *shape):
        """Reshape the array.

        Must not attempt to reshape the distributed axis. That axis must be
        given an input length `None`.

        Parameters
        ----------
        shape : tuple
            Tuple of axis lengths. The distributed must be given `None`.

        Returns
        -------
        array : MPIArray
            Reshaped MPIArray as a view of the original data.
        """

        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        # Find which axis is distributed
        list_shape = list(shape)
        new_axis = list_shape.index(None)

        # Fill in the missing value
        local_shape = list_shape[:]
        global_shape = list_shape[:]
        local_offset = [0] * len(list_shape)
        local_shape[new_axis] = self.local_shape[self.axis]
        global_shape[new_axis] = self.global_shape[self.axis]
        local_offset[new_axis] = self.local_offset[self.axis]

        # Check that the array sizes are compatible
        if np.prod(local_shape) != np.prod(self.local_shape):
            raise Exception("Dataset shapes incompatible.")

        rdata = np.ndarray.reshape(self, local_shape)

        rdata._axis = new_axis
        rdata._comm = self._comm
        rdata._local_shape = tuple(local_shape)
        rdata._global_shape = tuple(global_shape)
        rdata._local_offset = tuple(local_offset)

        return rdata

    def copy(self):
        """Return a copy of the MPIArray.

        Returns
        -------
        arr_copy : MPIArray
        """
        return MPIArray.wrap(
            self.view(np.ndarray).copy(), axis=self.axis, comm=self.comm
        )

    def gather(self, rank=0):
        """Gather a full copy onto a specific rank.

        Parameters
        ----------
        rank : int, optional
            Rank to gather onto. Default is rank=0

        Returns
        -------
        arr : np.ndarray, or None
            The full global array on the specified rank.
        """
        if self.comm.rank == rank:
            arr = np.ndarray(self.global_shape, dtype=self.dtype)
        else:
            arr = None

        splits = mpiutil.split_all(self.global_shape[self.axis], self.comm)

        for ri, (n, s, e) in enumerate(zip(*splits)):

            if self.comm.rank == rank:

                # Construct a temporary array for the data to be received into
                tshape = list(self.global_shape)
                tshape[self.axis] = n
                tbuf = np.ndarray(tshape, dtype=self.dtype)

                # Set up the non-blocking receive request
                request = self.comm.Irecv(tbuf, source=ri)

            # Send the data
            if self.comm.rank == ri:
                self.comm.Isend(self.view(np.ndarray), dest=rank)

            if self.comm.rank == rank:

                # Wait until the data has arrived
                stat = mpiutil.MPI.Status()
                request.Wait(status=stat)

                if stat.error != mpiutil.MPI.SUCCESS:
                    logger.error(
                        "**** ERROR in MPI RECV (source: %i,  dest rank: %i) *****"
                        % (ri, rank)
                    )

                # Put the data into the correct location
                dest_slice = [slice(None)] * len(self.shape)
                dest_slice[self.axis] = slice(s, e)
                arr[tuple(dest_slice)] = tbuf

        return arr

    def allgather(self):
        """Gather a full copy onto each rank.

        Returns
        -------
        arr : np.ndarray
            The full global array.
        """
        arr = np.ndarray(self.global_shape, dtype=self.dtype)

        splits = mpiutil.split_all(self.global_shape[self.axis], self.comm)

        for ri, (n, s, e) in enumerate(zip(*splits)):

            # Construct a temporary array for the data to be received into
            tshape = list(self.global_shape)
            tshape[self.axis] = n
            tbuf = np.ndarray(tshape, dtype=self.dtype)

            if self.comm.rank == ri:
                tbuf[:] = self

            self.comm.Bcast(tbuf, root=ri)

            # Copy the array into the correct place
            dest_slice = [slice(None)] * len(self.shape)
            dest_slice[self.axis] = slice(s, e)
            arr[tuple(dest_slice)] = tbuf

        return arr

    def _to_hdf5_serial(self, filename, dataset, create=False):
        """Write into an HDF5 dataset.

        This explicitly serialises the IO so that it works when h5py does not
        support MPI-IO.

        Parameters
        ----------
        filename : str
            File to write dataset into.
        dataset : string
            Name of dataset to write into. Should not exist.
        """

        ## Naive non-parallel implementation to start

        import h5py

        if h5py.get_config().mpi:
            import warnings

            warnings.warn(
                "h5py has parallel support. "
                "Use the parallel `.to_hdf5` routine instead."
            )

        if self.comm is None or self.comm.rank == 0:

            with h5py.File(filename, "a" if create else "r+") as fh:
                if dataset in fh:
                    raise Exception("Dataset should not exist.")

                fh.create_dataset(dataset, self.global_shape, dtype=self.dtype)
                fh[dataset][:] = np.array(0.0).astype(self.dtype)

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
                with h5py.File(filename, "r+") as fh:

                    start = dist_arr.local_offset[0]
                    end = start + dist_arr.local_shape[0]

                    fh[dataset][start:end] = dist_arr

            dist_arr.comm.Barrier()

    def _partition_io(self, skip=False, threshold=1.99):
        """Split IO of this array into local sections under `threshold`.

        Parameters
        ----------
        skip : bool, optional
            Don't partition, just find and return a full axis.
        threshold : float, optional
            Maximum size of IO (in GB).

        Returns
        -------
        split_axis : int
            Which axis are we going to split along.
        partitions : list of slice objects
            List of slices.
        """
        from mpi4py import MPI

        threshold_bytes = threshold * 2 ** 30
        largest_size = self.comm.allreduce(self.nbytes, op=MPI.MAX)
        num_split = int(np.ceil(largest_size / threshold_bytes))

        # Return early if we can
        if skip or num_split == 1:
            return 0, [slice(0, self.local_shape[0])]

        if self.ndim == 1:
            raise RuntimeError("To parition an array we must have multiple axes.")

        # Try and find the axis to split over
        for split_axis in range(self.ndim):
            if split_axis != self.axis and self.global_shape[split_axis] >= num_split:
                break
        else:
            raise RuntimeError(
                "Can't identify an IO partition less than %.2f GB in size: "
                "shape=%s, distributed axis=%i"
                % (threshold, self.global_shape, self.axis)
            )

        logger.debug("Splitting along axis %i, %i ways", split_axis, num_split)

        # Figure out the start and end of the splits and return
        nums, starts, ends = mpiutil.split_m(self.global_shape[split_axis], num_split)

        slices = [slice(start, end) for start, end in zip(starts, ends)]
        return split_axis, slices


def _partition_sel(sel, split_axis, n, slice_):
    # Take a selection (a tuple of slices) and re-slice along the split_axis
    # (which has length n)
    #
    # Returns the new selections for the initial (pre-selection) space and the final
    # (post-selection) space.

    l = _len_slice(sel[split_axis], n)

    # Reconstruct the slice for the split axis
    slice_init = _reslice(sel[split_axis], n, slice_)

    # Construct the final selection
    sel_final = [slice(None)] * len(sel)
    sel_final[split_axis] = slice_

    # Construct the initial selection
    sel_initial = list(sel)
    sel_initial[split_axis] = slice_init

    return tuple(sel_initial), tuple(sel_final)


def _len_slice(slice_, n):
    # Calculate the output length of a slice applied to an axis of length n
    start, stop, step = slice_.indices(n)
    return 1 + (stop - start - 1) // step


def _reslice(slice_, n, subslice):
    # For a slice along an axis of length n, return the slice that would select the
    # slice(start, end) elements of the final array.
    #
    # In other words find a single slice that has the same affect as application of two successive
    # slices
    dstart, dstop, dstep = slice_.indices(n)

    if subslice.step is not None and subslice.step > 1:
        raise ValueError("stride > 1 not supported. subslice: %s" % subslice)

    return slice(
        dstart + subslice.start * dstep,
        min(dstart + subslice.stop * dstep, dstop),
        dstep,
    )


def _expand_sel(sel, naxis):
    # Expand the selection to the full dimensions
    if sel is None:
        sel = [slice(None)] * naxis
    if len(sel) < naxis:
        sel = list(sel) + [slice(None)] * (naxis - len(sel))
    return list(sel)


class DummyContext(object):
    """A completely dummy context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
