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
:attribute:`MPIArray.global_slice` property. This can be used for both fetching
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
            print ri, arr
        mpiutil.barrier()

    # Use a global index to assign to the array
    arr.global_slice[3] = 17

    # Fetch a view of the whole array with a full slice
    arr2 = arr.global_slice[:, 2]

    # This should be the third column of the array
    for ri in range(mpiutil.size):
        if ri == mpiutil.rank:
            print ri, arr2
        mpiutil.barrier()

    # Fetch a view of the whole array with a partial slice
    arr3 = arr.global_slice[:2, 2]

    # The final two ranks should be None
    for ri in range(mpiutil.size):
        if ri == mpiutil.rank:
            print ri, arr3
        mpiutil.barrier()

"""
# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility

from past.builtins import basestring
import os
import time
import numpy as np

from caput import mpiutil, misc


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
            slice_list[self.axis] = None if (index < 0 or index >= local_length) else index
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
                start = start if start >= 0 else start + self.length  # Resolve negative indices
                fullslice = False
                start = start - self.offset
            else:
                start = 0

            # Check if stop is defined and modify slice
            if stop is not None:
                stop = stop if stop >= 0 else stop + self.length  # Resolve negative indices
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
            dist_axis = [index for index, sl in enumerate(slobj) if not isinstance(sl, int)].index(self.axis)

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
        local_num, local_start, local_end = mpiutil.split_local(global_shape[axis], comm=comm)

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
            if self.comm is None or self.comm.rank == 0:
                import warnings
                warnings.warn('Cannot redistribute array of compound datatypes. Sorry!!')
            return self

        # Construct the list of the axes to swap around
        axlist_f = list(range(len(self.shape)))

        # Remove the axes we are going to swap around
        axlist_f.remove(self.axis)
        axlist_f.remove(axis)

        # Move the current dist axis to the front, and the new to the end
        axlist_f.insert(0, self.axis)
        axlist_f.append(axis)

        # Perform a local transpose on the array to get the axes in the correct order
        trans_arr = self.view(np.ndarray).transpose(axlist_f).copy()

        # Perform the global transpose
        tmp_gshape = (self.global_shape[self.axis],) + trans_arr.shape[1:]
        trans_arr = mpiutil.transpose_blocks(trans_arr, tmp_gshape, comm=self.comm)

        axlist_b = list(range(len(self.shape)))
        axlist_b.pop(0)
        last = axlist_b.pop(-1)
        if self.axis < axis:  # This has to awkwardly depend on the order of the axes
            axlist_b.insert(self.axis, 0)
            axlist_b.insert(axis, last)
        else:
            axlist_b.insert(axis, last)
            axlist_b.insert(self.axis, 0)

        # Perform the local transpose to get the axes back in the correct order
        trans_arr = trans_arr.transpose(axlist_b)

        # Create a new MPIArray object out of the data
        dist_arr = MPIArray(self.global_shape, axis=axis, dtype=self.dtype, comm=self.comm)
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
    def from_hdf5(cls, f, dataset, comm=None):
        """Read MPIArray from an HDF5 dataset in parallel.

        Parameters
        ----------
        f : filename, or `h5py.File` object
            File to read dataset from.
        dataset : string
            Name of dataset to read from. Must exist.
        comm : MPI.Comm
            MPI communicator to distribute over. If `None` optional, use
            `MPI.COMM_WORLD`.

        Returns
        -------
        array : MPIArray
        """
        fh = misc.open_h5py_mpi(f, 'r', comm)

        dset = fh[dataset]
        gshape = dset.shape
        dtype = dset.dtype
        dist_arr = cls(gshape, axis=0, comm=comm, dtype=dtype)

        start = dist_arr.local_offset[0]
        end = start + dist_arr.local_shape[0]

        # Read using MPI-IO if possible
        if fh.is_mpi:
            with dset.collective:
                print("using mpi-io")
                dist_arr[:] = dset[start:end]
        else:
            dist_arr[:] = dset[start:end]

        if fh.opened:
            fh.close()

        return dist_arr

    def to_hdf5(self, f, dataset, create=False):
        """Parallel write into a contiguous HDF5 dataset.

        Parameters
        ----------
        filename : str, h5py.File or h5py.Group
            File to write dataset into.
        dataset : string
            Name of dataset to write into. Should not exist.
        """

        ## Naive non-parallel implementation to start

        import h5py

        if not h5py.get_config().mpi:
            if isinstance(filename, basestring):
                self._to_hdf5_serial(filename, dataset, create)
            else:
                raise ValueError(
                    "Argument must be a filename if h5py does not have MPI support"
                )

        mode = 'a' if create else 'r+'

        fh = misc.open_h5py_mpi(f, mode, self.comm)

        dset = fh.create_dataset(dataset, shape=self.global_shape, dtype=self.dtype)

        start = self.local_offset[self.axis]
        end = start + self.local_shape[self.axis]

        # Construct slices for axis
        sl = [slice(None, None)] * self.axis
        sl += [slice(start, end)]
        sl = tuple(sl)

        with dset.collective:
            dset[sl] = self[:]

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
        return MPIArray.wrap(self.view(np.ndarray).copy(), axis=self.axis, comm=self.comm)

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
            warnings.warn('h5py has parallel support. '
                          'Use the parallel `.to_hdf5` routine instead.')

        if self.comm is None or self.comm.rank == 0:

            with h5py.File(filename, 'a' if create else 'r+') as fh:
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
                with h5py.File(filename, 'r+') as fh:

                    start = dist_arr.local_offset[0]
                    end = start + dist_arr.local_shape[0]

                    fh[dataset][start:end] = dist_arr

            dist_arr.comm.Barrier()