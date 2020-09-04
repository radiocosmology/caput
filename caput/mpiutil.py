"""
Utilities for making MPI usage transparent.

This module exposes much of the functionality of :mod:`mpi4py` but will still
run in serial if mpi is not present on the system. It is thus useful for
writing code that can be run in either parallel or serial. Also it exposes all
attributes of the :mod:`mpi4py.MPI` by the :class:`SelfWrapper` class for
convenience. You can just use::

    mpiutil.attr

instead of::

    from mpi4py import MPI

    MPI.attr


**Functions**

.. autosummary::
    :toctree: generated/

        active_comm
        active
        close
        partition_list
        partition_list_mpi
        mpilist
        mpirange
        barrier
        bcast
        parallel_map
        typemap
        split_m
        split_all
        split_local
        gather_local
        transpose_blocks
        allocate_hdf5_dataset
        lock_and_write_buffer
        parallel_rows_write_hdf5
"""

# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import logging
import sys
import time
import warnings
from types import ModuleType
import numpy as np


rank = 0
size = 1
_comm = None
world = None
rank0 = True

logger = logging.getLogger(__name__)

## Try to setup MPI and get the comm, rank and size.
## If not they should end up as rank=0, size=1.
try:
    from mpi4py import MPI

    _comm = MPI.COMM_WORLD
    world = _comm

    rank = _comm.Get_rank()
    size = _comm.Get_size()

    if _comm is not None and size > 1:
        logger.debug("Starting MPI rank=%i [size=%i]", rank, size)

    rank0 = True if rank == 0 else False

    def mpi_excepthook(type, value, traceback):

        # Run the standard exception handler, but try to ensure the output if flushed out before
        # aborting
        sys.__excepthook__(type, value, traceback)
        sys.stdout.flush()
        sys.stderr.flush()
        time.sleep(5)

        MPI.COMM_WORLD.Abort(1)

    sys.excepthook = mpi_excepthook


except ImportError:
    warnings.warn("Warning: mpi4py not installed.", ImportWarning)


class _close_message(object):
    def __repr__(self):
        return "<Close message>"


def active_comm(aprocs):
    """Return a communicator consists of a list of processes in `aprocs`."""
    if _comm is None:
        return None
    else:
        # create a new communicator from active processes
        comm = _comm.Create(_comm.Get_group().Incl(aprocs))
        return comm


def active(aprocs):
    """Make a list of processes in `aprocs` active, while others wait."""
    if _comm is None:
        return None
    else:
        # create a new communicator from active processes
        comm = _comm.Create(_comm.Get_group().Incl(aprocs))
        if rank not in aprocs:
            while True:
                # Event loop.
                # Sit here and await instructions.

                # Blocking receive to wait for instructions.
                task = _comm.recv(source=0, tag=MPI.ANY_TAG)

                # Check if message is special sentinel signaling end.
                # If so, stop.
                if isinstance(task, _close_message):
                    break

        return comm


def close(aprocs):
    """Send a message to the waiting processes to close their waiting."""
    if rank0:
        for i in list(set(range(size)) - set(aprocs)):
            _comm.isend(_close_message(), dest=i)


def partition_list(full_list, i, n, method="con"):
    """Partition a list into `n` pieces. Return the `i` th partition."""

    def _partition(N, n, i):
        # If partiion `N` numbers into `n` pieces,
        # return the start and stop of the `i` th piece
        base = N // n
        rem = N % n
        num_lst = rem * [base + 1] + (n - rem) * [base]
        cum_num_lst = np.cumsum([0] + num_lst)

        return cum_num_lst[i], cum_num_lst[i + 1]

    N = len(full_list)
    start, stop = _partition(N, n, i)

    if method == "con":
        return full_list[start:stop]
    elif method == "alt":
        return full_list[i::n]
    elif method == "rand":
        choices = np.random.permutation(N)[start:stop]
        return [full_list[i] for i in choices]
    else:
        raise ValueError("Unknown partition method %s" % method)


def partition_list_mpi(full_list, method="con", comm=_comm):
    """Return the partition of a list specific to the current MPI process."""
    if comm is not None:
        rank = comm.rank
        size = comm.size

    return partition_list(full_list, rank, size, method=method)


# alias mpilist for partition_list_mpi for convenience
mpilist = partition_list_mpi


def mpirange(*args, **kargs):
    """An MPI aware version of `range`, each process gets its own sub section."""
    full_list = list(range(*args))

    method = kargs.get("method", "con")
    comm = kargs.get("comm", _comm)

    return partition_list_mpi(full_list, method=method, comm=comm)


def barrier(comm=_comm):
    if comm is not None and comm.size > 1:
        comm.Barrier()


def bcast(data, root=0, comm=_comm):
    if comm is not None and comm.size > 1:
        return comm.bcast(data, root=root)
    else:
        return data


def allreduce(sendobj, op=None, comm=_comm):
    if comm is not None and comm.size > 1:
        return comm.allreduce(sendobj, op=(op or MPI.SUM))
    else:
        return sendobj


# def Gatherv(sendbuf, recvbuf, root=0, comm=_comm):
#     if comm is not None and comm.size > 1:
#         comm.Gatherv(sendbuf, recvbuf, root=root)
#     else:
#         # if they are just numpy data buffer
#         recvbuf = sendbuf.copy()
#         # TODO, other cases


# def Allgatherv(sendbuf, recvbuf, comm=_comm):
#     if comm is not None and comm.size > 1:
#         return _comm.Allgatherv(sendbuf, recvbuf)
#     else:
#         # if they are just numpy data buffer
#         recvbuf = sendbuf.copy()
#         # TODO, other cases


def parallel_map(func, glist, root=None, method="con", comm=_comm):
    """Apply a parallel map using MPI.

    Should be called collectively on the same list. All ranks return the full
    set of results.

    Parameters
    ----------
    func : function
        Function to apply.
    glist : list
        List of map over. Must be globally defined.
    root : None or Integer
        Which process should gather the results, all processes will gather the results if None.
    method: str
        How to split `glist` to each process, can be 'con': continuously, 'alt': alternatively, 'rand': randomly. Default is 'con'.
    comm : MPI communicator
        MPI communicator that array is distributed over. Default is the gobal _comm.

    Returns
    -------
    results : list
        Global list of results.
    """

    # Synchronize
    barrier(comm=comm)

    # If we're only on a single node, then just perform without MPI
    if comm is None or comm.size == 1:
        return [func(item) for item in glist]

    # Pair up each list item with its position.
    zlist = list(enumerate(glist))

    # Partition list based on MPI rank
    llist = partition_list_mpi(zlist, method=method, comm=comm)

    # Operate on sublist
    flist = [(ind, func(item)) for ind, item in llist]

    barrier(comm=comm)

    rlist = None
    if root is None:
        # Gather all results onto all ranks
        rlist = comm.allgather(flist)
    else:
        # Gather all results onto the specified rank
        rlist = comm.gather(flist, root=root)

    if rlist is not None:
        # Flatten the list of results
        flatlist = [item for sublist in rlist for item in sublist]

        # Sort into original order
        sortlist = sorted(flatlist, key=(lambda item: item[0]))

        # Synchronize
        # barrier(comm=comm)

        # Extract the return values into a list
        return [item for ind, item in sortlist]
    else:
        return None


def typemap(dtype):
    """Map a numpy dtype into an MPI_Datatype.

    Parameters
    ----------
    dtype : np.dtype
        The numpy datatype.

    Returns
    -------
    mpitype : MPI.Datatype
        The MPI.Datatype.
    """
    # Need to try both as the name of the typedoct changed in mpi4py 2.0
    try:
        return MPI.__TypeDict__[np.dtype(dtype).char]
    except AttributeError:
        return MPI._typedict[np.dtype(dtype).char]


def split_m(n, m):
    """
    Split a range (0, n-1) into m sub-ranges of similar length

    Parameters
    ----------
    n : integer
        Length of range to split.
    m : integer
        Number of subranges to split into.

    Returns
    -------
    num : np.ndarray[m]
        Number in each sub-range
    start : np.ndarray[m]
        Starting of each sub-range.
    end : np.ndarray[m]
        End of each sub-range.

    See Also
    --------
    :func:`split_all`, :func:`split_local`

    """
    base = n // m
    rem = n % m

    part = base * np.ones(m, dtype=np.int) + (np.arange(m) < rem).astype(np.int)

    bound = np.cumsum(np.insert(part, 0, 0))

    return np.array([part, bound[:m], bound[1 : (m + 1)]])


def split_all(n, comm=_comm):
    """
    Split a range (0, n-1) into sub-ranges for each MPI Process.

    Parameters
    ----------
    n : integer
        Length of range to split.
    comm : MPI Communicator, optional
        MPI Communicator to use (default COMM_WORLD).

    Returns
    -------
    num : np.ndarray[m]
        Number for each rank.
    start : np.ndarray[m]
        Starting of each sub-range on a given rank.
    end : np.ndarray[m]
        End of each sub-range.

    See Also
    --------
    :func:`split_m`, :func:`split_local`
    """

    m = size if comm is None else comm.size

    return split_m(n, m)


def split_local(n, comm=_comm):
    """
    Split a range (0, n-1) into sub-ranges for each MPI Process. This returns
    the parameters only for the current rank.

    Parameters
    ----------
    n : integer
        Length of range to split.
    comm : MPI Communicator, optional
        MPI Communicator to use (default COMM_WORLD).

    Returns
    -------
    num : integer
        Number on this rank.
    start : integer
        Starting of the sub-range for this rank.
    end : integer
        End of rank for this rank.

    See Also
    --------
    :func:`split_all`, :func:`split_local`
    """

    pse = split_all(n, comm=comm)
    m = rank if comm is None else comm.rank

    return pse[:, m]


def gather_local(global_array, local_array, local_start, root=0, comm=_comm):
    """
    Gather data array in each process to the global array in `root` process.

    Parameters
    ----------
    global_array : np.ndarray
        The global array which will collect data from `local_array` in each process.
    local_array : np.ndarray
        The local array in each process to be collected to `global_array`.
    local_start : N-tuple
        The starting index of the local array to be placed in `global_array`.
    root : integer
        The process local array gathered to.
    comm : MPI communicator
        MPI communicator that array is distributed over. Default is MPI.COMM_WORLD.

    """

    local_size = local_array.shape

    if comm is None or comm.size == 1:
        # only one process
        slc = [slice(s, s + n) for (s, n) in zip(local_start, local_size)]
        global_array[slc] = local_array.copy()
    else:
        local_sizes = comm.gather(local_size, root=root)
        local_starts = comm.gather(local_start, root=root)
        mpi_type = typemap(local_array.dtype)

        # Each process should send its local sections.
        if np.prod(local_size) > 0:
            # send only when array is non-empty
            sreq = comm.Isend(
                [np.ascontiguousarray(local_array), mpi_type], dest=root, tag=0
            )

        if comm.rank == root:
            # list of processes which have non-empty array
            nonempty_procs = [
                i for i in range(comm.size) if np.prod(local_sizes[i]) > 0
            ]
            # create newtype corresponding to the local array section in the global array
            sub_type = [
                mpi_type.Create_subarray(
                    global_array.shape, local_sizes[i], local_starts[i]
                ).Commit()
                for i in nonempty_procs
            ]  # default order=ORDER_C
            # Post each receive
            reqs = [
                comm.Irecv([global_array, sub_type[si]], source=sr, tag=0)
                for (si, sr) in enumerate(nonempty_procs)
            ]

            # Wait for requests to complete
            MPI.Prequest.Waitall(reqs)

        # Wait on send request. Important, as can get weird synchronisation
        # bugs otherwise as processes exit before completing their send.
        if np.prod(local_size) > 0:
            sreq.Wait()


def transpose_blocks(row_array, shape, comm=_comm):
    """
    Take a 2D matrix which is split between processes row-wise and split it
    column wise between processes.

    Parameters
    ----------
    row_array : np.ndarray
        The local section of the global array (split row wise).
    shape : 2-tuple
        The shape of the global array
    comm : MPI communicator
        MPI communicator that array is distributed over. Default is MPI.COMM_WORLD.

    Returns
    -------
    col_array : np.ndarray
        Local section of the global array (split column wise).
    """

    if comm is None or comm.size == 1:
        # only one process
        if row_array.shape[:-1] == shape[:-1]:
            # We are working on a single node and being asked to do
            # a trivial transpose.
            # Note that to mimic the mpi behaviour we have to allow the
            # last index to be trimmed.
            return row_array[..., : shape[-1]].copy()
        else:
            raise ValueError(
                "Shape %s is incompatible with `row_array`s shape %s"
                % (shape, row_array.shape)
            )

    nr = shape[0]
    nc = shape[-1]
    nm = 1 if len(shape) <= 2 else np.prod(shape[1:-1])

    pr, sr, er = split_local(nr, comm=comm) * nm
    pc, sc, ec = split_local(nc, comm=comm)

    par, sar, ear = split_all(nr, comm=comm) * nm
    pac, sac, eac = split_all(nc, comm=comm)

    row_array = row_array[:nr, ..., :nc].reshape(pr, nc)

    requests_send = []
    requests_recv = []

    recv_buffer = np.empty((nr * nm, pc), dtype=row_array.dtype)

    mpitype = typemap(row_array.dtype)

    # Iterate over all processes row wise
    for ir in range(comm.size):

        # Get the start and end of each set of rows
        sir, eir = sar[ir], ear[ir]

        # Iterate over all processes column wise
        for ic in range(comm.size):

            # Get the start and end of each set of columns
            sic, eic = sac[ic], eac[ic]

            # Construct a unique tag
            tag = ir * comm.size + ic

            # Send and receive the messages as non-blocking passes
            if comm.rank == ir:

                # Construct the block to send by cutting out the correct
                # columns
                block = row_array[:, sic:eic].copy()

                # Send the message
                request = comm.Isend([block, mpitype], dest=ic, tag=tag)
                requests_send.append([ir, ic, request])

            if comm.rank == ic:

                # Receive the message into the correct set of rows of recv_buffer
                request = comm.Irecv(
                    [recv_buffer[sir:eir], mpitype], source=ir, tag=tag
                )
                requests_recv.append([ir, ic, request])

    # Wait for all processes to have started their messages
    comm.Barrier()

    # For each node iterate over all sends and wait until completion
    for ir, ic, request in requests_send:

        stat = MPI.Status()

        request.Wait(status=stat)

        if stat.error != MPI.SUCCESS:
            logger.error(
                "**** ERROR in MPI SEND (r: %i c: %i rank: %i) *****"
                % (ir, ic, comm.rank)
            )

    comm.Barrier()

    # For each frequency iterate over all receives and wait until completion
    for ir, ic, request in requests_recv:

        stat = MPI.Status()

        request.Wait(status=stat)

        if stat.error != MPI.SUCCESS:
            logger.error(
                "**** ERROR in MPI RECV (r: %i c: %i rank: %i) *****"
                % (ir, ir, comm.rank)
            )

    return recv_buffer.reshape(shape[:-1] + (pc,))


def allocate_hdf5_dataset(fname, dsetname, shape, dtype, comm=_comm):
    """Create a hdf5 dataset and return its offset and size.

    The dataset will be created contiguously and immediately allocated,
    however it will not be filled.

    Parameters
    ----------
    fname : string
        Name of the file to write.
    dsetname : string
        Name of the dataset to write (must be at root level).
    shape : tuple
        Shape of the dataset.
    dtype : numpy datatype
        Type of the dataset.
    comm : MPI communicator
        Communicator over which to broadcast results.

    Returns
    -------
    offset : integer
        Offset into the file at which the dataset starts (in bytes).
    size : integer
        Size of the dataset in bytes.

    """

    import h5py

    state = None

    if comm is None or comm.rank == 0:

        # Create/open file
        f = h5py.File(fname, "a")

        # Create dataspace and HDF5 datatype
        sp = h5py.h5s.create_simple(shape, shape)
        tp = h5py.h5t.py_create(dtype)

        # Create a new plist and tell it to allocate the space for dataset
        # immediately, but don't fill the file with zeros.
        plist = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        plist.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
        plist.set_fill_time(h5py.h5d.FILL_TIME_NEVER)

        # Create the dataset
        dset = h5py.h5d.create(f.id, dsetname, tp, sp, plist)

        # Get the offset of the dataset into the file.
        state = dset.get_offset(), dset.get_storage_size()

        f.close()

    # state = comm.bcast(state, root=0)
    state = bcast(state, root=0, comm=comm)

    return state


def lock_and_write_buffer(obj, fname, offset, size):
    """Write the contents of a buffer to disk at a given offset, and explicitly
    lock the region of the file whilst doing so.

    Parameters
    ----------
    obj : buffer
        Data to write to disk.
    fname : string
        Filename to write.
    offset : integer
        Offset into the file to start writing at.
    size : integer
        Size of the region to write to (and lock).
    """
    import os
    import os.fcntl as fcntl

    buf = buffer(obj)

    if len(buf) > size:
        raise Exception("Size doesn't match array length.")

    fd = os.open(fname, os.O_RDRW | os.O_CREAT)

    fcntl.lockf(fd, fcntl.LOCK_EX, size, offset, os.SEEK_SET)

    nb = os.write(fd, buf)

    if nb != len(buf):
        raise Exception("Something funny happened with the reading.")

    fcntl.lockf(fd, fcntl.LOCK_UN)

    os.close(fd)


def parallel_rows_write_hdf5(fname, dsetname, local_data, shape, comm=_comm):
    """Write out array (distributed across processes row wise) into a HDF5 in parallel."""

    offset, size = allocate_hdf5_dataset(
        fname, dsetname, shape, local_data.dtype, comm=comm
    )

    lr, sr, er = split_local(shape[0], comm=comm)

    nc = np.prod(shape[1:])

    lock_and_write_buffer(local_data, fname, offset + sr * nc, lr * nc)


# this is a thin wrapper around THIS module (we patch sys.modules[__name__])
class SelfWrapper(ModuleType):
    def __init__(self, self_module, baked_args={}):
        for attr in [
            "__file__",
            "__hash__",
            "__buildins__",
            "__doc__",
            "__name__",
            "__package__",
        ]:
            setattr(self, attr, getattr(self_module, attr, None))

        self.self_module = self_module

    def __getattr__(self, name):
        if name in globals():
            return globals()[name]
        elif _comm is not None and name in MPI.__dict__:
            return MPI.__dict__[name]
        else:
            raise AttributeError("module 'mpiutil' has no attribute '%s'" % name)

    def __call__(self, **kwargs):
        return SelfWrapper(self.self_module, kwargs)


self = sys.modules[__name__]
sys.modules[__name__] = SelfWrapper(self)
