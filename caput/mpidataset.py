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
import warnings

from mpi4py import MPI

from . import mpiarray


warnings.warn('MPIDataset is deprecated in favour of memh5 and will be removed soon.')


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
            cdt._distributed[k] = mpiarray.MPIArray.wrap(v.copy(), axis=v.axis, comm=v.comm) if deep and v is not None else v

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
        pdset = cls.__new__(cls)
        MPIDataset.__init__(pdset, comm=comm)

        fh = None
        if pdset._comm.rank == 0:
            fh = h5py.File(filename, 'r')
            #
            # if '__mpidataset_class' not in fh.attrs:
            #     raise Exception('Not in the MPIDataset format.')
            #
            # # Won't properly deal with inheritance. Ho hum.
            # if fh.attrs['__mpidataset_class'] != cls.__name__:
            #     raise Exception('Not correct MPIDataset class.')

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
                pdata = mpiarray.MPIArray.from_hdf5(filename, dset, comm=pdset._comm)
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
