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
    MPIDataset

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
import os
import collections

import numpy as np
from mpi4py import MPI

from caput import mpiutil


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

        if comm is None:
            comm = MPI.COMM_WORLD

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

        if comm is None:
            comm = MPI.COMM_WORLD

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


class MPIDataset(collections.Mapping):
    """A container for distributed datasets.

    This can have both distributed and non-distributed parts and can load from
    and save to HDF5 files.

    Attributes
    ----------
    attrs : dict
        Dictionary for small attributes.
    common : read only dict
        Dict of common datasets (i.e. np.ndarray's)
    distributed : read only dict
        Distributed datasets (i.e. MPIArray's)
    comm : read only MPI.Comm
        Communicator we are distributed over.

    Methods
    -------
    from_hdf5
    to_hdf5
    redistribute
    """

    _attrs = {}

    @property
    def attrs(self):
        return self._attrs

    _common = {}

    @property
    def common(self):
        return self._common

    _distributed = {}

    @property
    def distributed(self):
        return self._distributed

    _comm = None

    @property
    def comm(self):
        return self._comm

    def copy(self, deep=False):

        cdt = self.__class__.__new__(self.__class__)
        cdt._attrs = self._attrs.copy()

        for k, v in self._common.items():
            cdt._common[k] = v.copy() if deep and v is not None else v

        for k, v in self._distributed.items():
            cdt._distributed[k] = MPIArray.wrap(v.copy(), axis=v.axis, comm=v.comm) if deep and v is not None else v

        cdt._comm = self._comm

        return cdt

    def __init__(self, comm=None):

        if comm is None:
            comm = MPI.COMM_WORLD

        self._comm = comm

        # Explicitly copy to ensure these are defined in the instance and not
        # the class
        self._attrs = self._attrs.copy()
        self._common = self._common.copy()
        self._distributed = self._distributed.copy()

    @classmethod
    def from_hdf5(cls, filename, comm=None):
        """Load up a dataset from an HDF5 file.

        Parameters
        ----------
        filename : string
            File to load from.
        comm : MPI.Comm
            MPI communicator to distribute across.

        Returns
        -------
        dset : MPIDataset (or subclass)
        """
        import h5py

        if not os.path.exists(filename):
            raise IOError('File does not exist.')

        if comm is None:
            comm = MPI.COMM_WORLD

        # Initialise object
        pdset = cls(comm=comm)

        fh = None
        if pdset._comm.rank == 0:
            fh = h5py.File(filename, 'r')

            if '__mpidataset_class' not in fh.attrs:
                raise Exception('Not in the MPIDataset format.')

            # Won't properly deal with inheritance. Ho hum.
            if fh.attrs['__mpidataset_class'] != cls.__name__:
                raise Exception('Not correct MPIDataset class.')

        # Read in attributes
        attr_dict = None
        if pdset._comm.rank == 0:
            attr_dict = { k: v for k, v in fh.attrs.items() }
        pdset._attrs = pdset._comm.bcast(attr_dict, root=0)

        # Read in common datasets and broadcast to all processes
        for dset in pdset.common.keys():
            cdata = None

            if pdset._comm.rank == 0:

                # Check if the dataset exists, If not, we are going to make it None.
                if dset in fh:
                    cdata = fh[dset][:]

            pdset.common[dset] = pdset._comm.bcast(cdata, root=0)

        # Read in parallel datasets across all nodes
        for dset in pdset.distributed.keys():

            # Check whether the datasets exists, and thus we should read it
            # Do only on rank=0 and bcast to other ranks
            should_read = True
            if pdset._comm.rank == 0:
                should_read = dset in fh
            should_read = pdset._comm.bcast(should_read, root=0)

            # Read in the dataset
            if should_read:
                pdata = MPIArray.from_hdf5(filename, dset, comm=pdset._comm)
            else:
                pdata = None

            pdset.distributed[dset] = pdata

        if pdset._comm.rank == 0:
            fh.close()

        return pdset

    def to_hdf5(self, filename):
        """Save a dataset to an HDF5 file.

        Parameters
        ----------
        filename : string
            File to save to.
        """
        import h5py

        if os.path.exists(filename):
            raise IOError('File %s already exists.' % filename)

        self._comm.Barrier()

        if self._comm.rank == 0:

            with h5py.File(filename, 'w') as fh:

                # Save attributes
                for k, v in self.attrs.items():
                    fh.attrs[k] = v

                # Save common datasets
                for dsetname, dsetdata in self.common.items():

                    if dsetdata is not None:
                        fh.create_dataset(dsetname, data=dsetdata)

        self._comm.Barrier()

        # Save parallel datasets
        for dsetname, dsetdata in self.distributed.items():

            # If a dataset is None, don't write it out.
            if dsetdata is not None:
                dsetdata.to_hdf5(filename, dsetname)

    def redistribute(self, axis):
        """Redistribute the distributed datasets onto a new axis.

        Parameters
        ----------
        axis : integer
            Axis to redistribute over. Assumes that the axis exists on all
            datasets.
        """

        for dsetname, dsetdata in self.distributed.items():
            if dsetdata is not None:
                self.distributed[dsetname] = dsetdata.redistribute(axis)

    def __getitem__(self, key):

        ## Fetch data from either the common or distributed sections
        if key in self.common:
            return self.common[key]
        elif key in self.distributed:
            return self.distributed[key]
        else:
            raise KeyError

    def __len__(self):
        clen = len(self.common) + len(self.distributed)
        return clen

    def __iter__(self):
        import itertools
        return itertools.chain(self.common, self.distributed)
