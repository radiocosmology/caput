"""
=============================================
MPI Distributed Array (:mod:`caput.mpiarray`)
=============================================

An array class for containing MPI distributed data.

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

    from mpidataset import MPIArray

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

"""
import numpy as np

from caput import mpiutil


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
            # Return an MPIArray view
            arr = self.array[slobj]
            return MPIArray.wrap(arr, axis=self.axis, comm=self.array._comm)

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
    def comm(self):
        return self._comm

    def __new__(cls, global_shape, axis=0, comm=None, *args, **kwargs):

        if mpiutil.world is None:
            raise RuntimeError('There is no mpi4py installation. Aborting.')

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

        from mpi4py import MPI

        if comm is None:
            comm = mpiutil.world

        # Get axis length, both locally, and globally
        axlen = array.shape[axis]
        totallen = comm.allreduce(axlen)

        # Figure out what the distributed layout should be
        local_num, local_start, local_end = mpiutil.split_local(totallen, comm=comm)

        # Check the local layout is consistent with what we expect, and send
        # result to all ranks
        layout_issue = comm.allreduce(axlen != local_num, op=MPI.MAX)

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
        if self.axis == axis:
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
        trans_arr = mpiutil.transpose_blocks(trans_arr, tmp_gshape)

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
        dist_arr = MPIArray(self.global_shape, axis=axis, dtype=self.dtype)
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

        import h5py

        if type(f) == str:
            fh = h5py.File(f, 'r')
        elif type(f) == h5py.File:
            fh = f
        else:
            raise Exception("Did not receive a h5py.File or filename")

        dset = fh[dataset]
        gshape = dset.shape
        dtype = dset.dtype
        dist_arr = cls(gshape, axis=0, comm=comm, dtype=dtype)

        start = dist_arr.local_offset[0]
        end = start + dist_arr.local_shape[0]

        dist_arr[:] = dset[start:end]

        return dist_arr

    def to_hdf5(self, filename, dataset):
        """Parallel write into a contiguous HDF5 dataset.

        Parameters
        ----------
        filename : str
            File to write dataset into.
        dataset : string
            Name of dataset to write into. Should not exist.
        """

        ## Naive non-parallel implementation to start

        import h5py

        if self._comm.rank == 0:

            with h5py.File(filename, 'a') as fh:
                if dataset in fh:
                    raise Exception("Dataset should not exist.")

                fh.create_dataset(dataset, self.global_shape, dtype=self.dtype)
                fh[dataset][:] = np.array(0.0).astype(self.dtype)

        self._comm.Barrier()

        if self.axis == 0:
            dist_arr = self
        else:
            dist_arr = self.redistribute(axis=0)

        for ri in range(dist_arr._comm.size):

            if ri == dist_arr._comm.rank:
                with h5py.File(filename, 'r+') as fh:

                    start = dist_arr.local_offset[0]
                    end = start + dist_arr.local_shape[0]

                    fh[dataset][start:end] = dist_arr

            dist_arr._comm.Barrier()

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

    def reshape(self, shape):
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
