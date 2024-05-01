"""Utilities for making MPI usage transparent.

This module exposes much of the functionality of :mod:`mpi4py` but will still
run in serial if mpi is not present on the system. It is thus useful for
writing code that can be run in either parallel or serial. Also it exposes all
attributes of the :mod:`mpi4py.MPI` by the :class:`SelfWrapper` class for
convenience. You can just use::

    mpiutil.attr

instead of::

    from mpi4py import MPI

    MPI.attr
"""

import logging
import os
import re
import sys
import time
import warnings
from types import ModuleType
from typing import Optional

import numpy as np
import psutil

rank = 0
size = 1
_comm = None
world = None
rank0 = True

logger = logging.getLogger(__name__)

# Try to setup MPI and get the comm, rank and size.
# If not they should end up as rank=0, size=1.
try:
    from mpi4py import MPI

    _comm = MPI.COMM_WORLD
    world = _comm
    world_scomm = _comm.Split_type(MPI.COMM_TYPE_SHARED)

    rank = _comm.Get_rank()
    size = _comm.Get_size()

    if _comm is not None and size > 1:
        logger.debug("Starting MPI rank=%i [size=%i]", rank, size)

    rank0 = rank == 0

except ImportError:
    warnings.warn("Warning: mpi4py not installed.", ImportWarning)


def enable_mpi_exception_handler():
    """Install an MPI aware exception handler.

    When enabled the whole MPI job will abort if *any* MPI process fails. If it's not
    enabled, an MPI job will continue until it is killed by the scheduler.

    The downside of enabling this is that it can cause slurm job steps and interactive
    allocations to be ended prematurely, which is annoying when debugging.
    """
    logger.debug("Installing MPI aware exception handler.")

    def mpi_excepthook(exc_type, exc_obj, exc_tb):
        # Run the standard exception handler, but try to ensure the output is flushed
        # out before aborting
        sys.__excepthook__(exc_type, exc_obj, exc_tb)
        sys.stdout.flush()
        sys.stderr.flush()
        time.sleep(5)

        # Send an MPI Abort to force other MPI ranks to exit
        MPI.COMM_WORLD.Abort(1)

    # Enable the faulthandler tracebacks as they are useful.
    import faulthandler
    import io

    try:
        faulthandler.enable()
    except io.UnsupportedOperation:
        logger.debug(
            "Could not enable faulthandler. "
            "This often happens when running within a test suite."
        )

    # Replace the standard exception handler
    sys.excepthook = mpi_excepthook


class _close_message:
    def __repr__(self):
        return "<Close message>"


def active_comm(aprocs):
    """Return a communicator consists of a list of processes in `aprocs`."""
    if _comm is None:
        return None

    # create a new communicator from active processes
    return _comm.Create(_comm.Get_group().Incl(aprocs))


def active(aprocs):
    """Make a list of processes in `aprocs` active, while others wait."""
    if _comm is None:
        return None

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


def can_allocate(
    nbytes: int,
    scope: str = "shared",
    comm: "MPI.Intracomm" = _comm,
    scomm: "MPI.Intracomm" = None,
) -> bool:
    """Check if nbytes of memory is available to allocate.

    nbytes can be the number of bytes allocated per process,
    number of bytes allocated in all shared memory, or number of
    bytes allocated globally when there are processes which do
    not share memory.

    Parameters
    ----------
    nbytes
        number of bytes that we want to allocate
    scope
        whether to find available memory on a global, per-node, or
        per-process basis
    comm
        MPI communicator
    scomm
        MPI shared memory communicator

    Returns
    -------
    available
        True if there is enough memory available
    """
    # Shared memory comm is used to communicate only between
    # processes on the same node.
    if scomm is None:
        if comm is world:
            scomm = world_scomm
        else:
            scomm = comm.Split_type(MPI.COMM_TYPE_SHARED)

    try:
        available_memory_node = available_memory_shared()
    except OSError as e:
        # We probably don't want this to fail if it can't get the available
        # memory, but it should definitey warn about it
        warnings.warn(f"Unable to check available memory: {e!r}", RuntimeWarning)
        return True

    if scope == "shared":
        can_alloc = nbytes < available_memory_node

    elif scope == "process":
        # Get all memory desired in the shared space. If only
        # some subset of all processes are trying to allocate
        # memory they will be allowed to do so
        nbytes = scomm.allreduce(np.array(nbytes), op=MPI.SUM)

        can_alloc = nbytes < available_memory_node

    elif scope == "global":
        available = np.array(available_memory_node) // scomm.size

        can_alloc = nbytes < comm.allreduce(available, op=MPI.SUM)

    else:
        raise ValueError(
            f"Scope must be one of (process, shared, global). Got {scope}."
        )

    return can_alloc


def available_memory_shared() -> int:
    """Shared memory in bytes available to a process.

    If the process is controlled by a cgroup, cgroup memory
    allocations are used. Assumes that all processes are part
    of the same cgroup.

    Returns
    -------
    available
        Memory available in bytes

    Raises
    ------
    OSError
    """
    # These files aren't very big so we can just read the whole thing
    try:
        cgrouptxt = open(f"/proc/{os.getpid()}/cgroup").read()
    except FileNotFoundError:
        # No cgroup file means no cgroup
        return psutil.virtual_memory().available
    except OSError as e:
        # This could happen for a number of reasons, but since it
        # wasn't a non-existant file that triggered it it should
        # be handled differently
        raise OSError("Failed to get cgroup file information") from e

    # This should only find anything if the process is in a v1 cgroup
    cgroup = re.findall(r"(?<=memory:).*", cgrouptxt)
    cgroup = cgroup[0].strip() if cgroup else "/"

    if cgroup == "/":
        # No cgroup or a v2 cgroup
        return psutil.virtual_memory().available

    # Look at memory.stat file for the current process.
    # This should only be reached if the process is in a v1 cgroup
    try:
        memorystat = open(f"/sys/fs/cgroup/memory/{cgroup}/memory.stat").read()
        mem_used = open(f"/sys/fs/cgroup/memory/{cgroup}/memory.usage_in_bytes").read()
    except OSError as e:
        # It doesn't really matter why this failed but it should inform
        # about failure. This probably shouldn't happen anyway
        raise OSError(f"Failed to get memory info for cgroup {cgroup}") from e

    # Memory limit imposed on the cgroup
    mem_limit = re.findall(r"(?<=hierarchical_memory_limit).*", memorystat)
    mem_limit = int(mem_limit[0].strip()) if mem_limit else 0
    # Approximate memory used by the cgroup
    mem_used = int(mem_used)

    return mem_limit - mem_used


def partition_list(full_list, i, n, method="con"):
    """Partition a list into `n` pieces. Return the `i` th partition."""

    def _partition(N, n, i):
        # If partiion `N` numbers into `n` pieces,
        # return the start and stop of the `i` th piece
        base = N // n
        rem = N % n
        num_lst = rem * [base + 1] + (n - rem) * [base]
        cum_num_lst = np.cumsum([0, *num_lst])

        return cum_num_lst[i], cum_num_lst[i + 1]

    N = len(full_list)
    start, stop = _partition(N, n, i)

    if method == "con":
        return full_list[start:stop]

    if method == "alt":
        return full_list[i::n]

    if method == "rand":
        choices = np.random.permutation(N)[start:stop]
        return [full_list[i] for i in choices]

    raise ValueError(f"Unknown partition method {method}")


def partition_list_mpi(full_list, method="con", comm=_comm):
    """Return the partition of a list specific to the current MPI process."""
    if comm is not None:
        rank = comm.rank
        size = comm.size

    return partition_list(full_list, rank, size, method=method)


# alias mpilist for partition_list_mpi for convenience
mpilist = partition_list_mpi


def mpirange(*args, **kargs):
    """MPI aware version of `range`, each process gets its own sub section."""
    full_list = list(range(*args))

    method = kargs.get("method", "con")
    comm = kargs.get("comm", _comm)

    return partition_list_mpi(full_list, method=method, comm=comm)


def barrier(comm=_comm):
    """Call comm.Barrier() if MPI is enabled."""
    if comm is not None and comm.size > 1:
        comm.Barrier()


def bcast(data, root=0, comm=_comm):
    """Call comm.bcast if MPI is enabled."""
    if comm is not None and comm.size > 1:
        return comm.bcast(data, root=root)

    return data


def allreduce(sendobj, op=None, comm=_comm):
    """Call comm.allreduce if MPI is enabled."""
    if comm is not None and comm.size > 1:
        return comm.allreduce(sendobj, op=(op or MPI.SUM))

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
    """Split a range (0, n-1) into m sub-ranges of similar length.

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

    part = base * np.ones(m, dtype=int) + (np.arange(m) < rem).astype(int)

    bound = np.cumsum(np.insert(part, 0, 0))

    return np.array([part, bound[:m], bound[1 : (m + 1)]])


def split_all(n, comm=_comm):
    """Split a range (0, n-1) into sub-ranges for each MPI Process.

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
    """Split a range (0, n-1) into sub-ranges for each MPI Process.

    This returns the parameters only for the current rank.

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
    """Gather data array in each process to the global array in `root` process.

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
    """Swap 2D matrix split from row-wise to columnwise.

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

        raise ValueError(
            f"Shape {shape} is incompatible with `row_array`s shape {row_array.shape}"
        )

    nr = shape[0]
    nc = shape[-1]
    nm = 1 if len(shape) <= 2 else np.prod(shape[1:-1])

    pr, _, _ = split_local(nr, comm=comm) * nm
    pc, _, _ = split_local(nc, comm=comm)

    _, sar, ear = split_all(nr, comm=comm) * nm
    _, sac, eac = split_all(nc, comm=comm)

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
                "**** ERROR in MPI SEND (r: %i c: %i rank: %i) *****", ir, ic, comm.rank
            )

    comm.Barrier()

    # For each frequency iterate over all receives and wait until completion
    for ir, ic, request in requests_recv:
        stat = MPI.Status()

        request.Wait(status=stat)

        if stat.error != MPI.SUCCESS:
            logger.error(
                "**** ERROR in MPI RECV (r: %i c: %i rank: %i) *****", ir, ir, comm.rank
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
    return bcast(state, root=0, comm=comm)


def lock_and_write_buffer(obj, fname, offset, size):
    """Write buffer contents to disk at a given locked offset.

    Write the contents of a buffer to disk at a given offset, and explicitly
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
    import fcntl
    import os

    buf = memoryview(obj)

    if len(buf) > size:
        raise Exception("Size doesn't match array length.")

    fd = os.open(fname, os.O_RDWR | os.O_CREAT)

    fcntl.lockf(fd, fcntl.LOCK_EX, size, offset, os.SEEK_SET)

    nb = os.write(fd, buf)

    if nb != len(buf):
        raise Exception("Something funny happened with the reading.")

    fcntl.lockf(fd, fcntl.LOCK_UN)

    os.close(fd)


def parallel_rows_write_hdf5(fname, dsetname, local_data, shape, comm=_comm):
    """Write out array (distributed across processes row wise) into a HDF5 in parallel."""
    offset, _ = allocate_hdf5_dataset(
        fname, dsetname, shape, local_data.dtype, comm=comm
    )

    lr, sr, _ = split_local(shape[0], comm=comm)

    nc = np.prod(shape[1:])

    lock_and_write_buffer(local_data, fname, offset + sr * nc, lr * nc)


# Need to disable the pylint error here as we only need to implement one method
class MPILogFilter(logging.Filter):  # pylint: disable=too-few-public-methods
    """Filter log entries by MPI rank.

    Also this will optionally add MPI rank information, and add an elapsed time
    entry.

    Parameters
    ----------
    add_mpi_info : boolean, optional
        Add MPI rank/size info to log records that don't already have it.
    level_rank0 : int
        Log level for messages from rank=0.
    level_all : int
        Log level for messages from all other ranks.
    """

    def __init__(
        self,
        add_mpi_info: bool = True,
        level_rank0: int = logging.INFO,
        level_all: int = logging.WARN,
        comm: Optional["MPI.Intracomm"] = None,
    ):
        super().__init__()

        self.add_mpi_info = add_mpi_info

        self.level_rank0 = level_rank0
        self.level_all = level_all

        self.rank = comm.rank if comm else rank
        self.size = comm.size if comm else size

    def filter(self, record):
        """Return True if we should filter the record."""
        # Add MPI info if desired
        try:
            record.mpi_rank
        except AttributeError:
            if self.add_mpi_info:
                record.mpi_rank = self.rank
                record.mpi_size = self.size

        # Add a new field with the elapsed time in seconds (as a float)
        record.elapsedTime = record.relativeCreated * 1e-3

        # Return whether we should filter the record or not.
        return (record.mpi_rank == 0 and record.levelno >= self.level_rank0) or (
            record.mpi_rank > 0 and record.levelno >= self.level_all
        )


class SelfWrapper(ModuleType):
    """A thin wrapper around THIS module (`we patch sys.modules[__name__]`)."""

    def __init__(self, self_module, baked_args=None):
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

        if _comm is not None and name in MPI.__dict__:
            return MPI.__dict__[name]

        raise AttributeError(f"module 'mpiutil' has no attribute '{name}'")

    def __call__(self, **kwargs):
        """Call self with set module."""
        return SelfWrapper(self.self_module, kwargs)


thismod = sys.modules[__name__]
sys.modules[__name__] = SelfWrapper(thismod)
