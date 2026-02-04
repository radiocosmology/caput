"""Utilities for making MPI usage transparent.

This module exposes much of the functionality of :mod:`mpi4py` but will still
run in serial if MPI is not present on the system. It is, thus, useful for
writing code that can be run in either parallel or serial. Also it exposes all
attributes of the :mod:`mpi4py.MPI` module by the :class:`_SelfWrapper` class for
convenience. You can just use::

    mpitools.attr

instead of::

    from mpi4py import MPI

    MPI.attr
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
import warnings
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import psutil

from .arraytools import partition_list, split_m

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    import numpy.typing as npt

logger = logging.getLogger(__name__)

# Try to setup MPI and get the comm, rank and size.
# If not they should end up as rank=0, size=1.
try:
    from mpi4py import MPI
except ImportError:
    warnings.warn("Warning: :py:mod:`mpi4py` not installed.", ImportWarning)
    _has_mpi = False
else:
    _has_mpi = True

_comm = MPI.COMM_WORLD if _has_mpi else None

world: MPI.Comm | None = _comm
"""Global MPI communicator, if MPI is enabled."""

world_scomm: MPI.Comm | None = (
    world.Split_type(MPI.COMM_TYPE_SHARED) if world is not None else None
)
"""Global shared-memory MPI communicator, if MPI is enabled."""

rank: int = _comm.Get_rank() if _has_mpi else 0
"""MPI rank of the current process."""

size: int = _comm.Get_size() if _has_mpi else 1
"""Total number of MPI ranks."""

rank0: bool = rank == 0
"""True if this rank is rank 0."""

if _comm is not None and size > 1:
    logger.debug(f"Starting MPI rank={rank} [size={size}]")


class _SelfWrapper(ModuleType):
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

        raise AttributeError(f"module 'mpitools' has no attribute '{name}'")

    def __call__(self, **kwargs):
        """Call self with set module."""
        return _SelfWrapper(self.self_module, kwargs)


# Expose all :mod:`mpi4py.MPI` attributes
__thismod = sys.modules[__name__]
sys.modules[__name__] = _SelfWrapper(__thismod)


# Basic MPI Operations
# ====================


def barrier(comm: MPI.Comm | None = _comm) -> None:
    """Call comm.Barrier() if MPI is enabled."""
    if comm is not None and comm.size > 1:
        comm.Barrier()


def bcast(data: Any, root: int = 0, comm: MPI.Comm | None = _comm) -> Any:
    """Call comm.bcast if MPI is enabled."""
    if comm is not None and comm.size > 1:
        return comm.bcast(data, root=root)

    return data


def allreduce(
    sendobj: Any, op: MPI.Op | None = None, comm: MPI.Comm | None = _comm
) -> Any:
    """Call comm.allreduce if MPI is enabled."""
    if comm is not None and comm.size > 1:
        return comm.allreduce(sendobj, op=(op or MPI.SUM))

    return sendobj


# System Information
# ==================


def cpu_count(
    comm: MPI.Comm | None = world, scomm: MPI.Comm | None = None
) -> int | None:
    """Get the number of CPUs available to each process.

    Parameters
    ----------
    comm : MPI.Comm | None, optional
        MPI communicator.
    scomm : MPI.Comm | None, optional
        MPI shared memory communicator.

    Returns
    -------
    num_cores : int | None
        Number of cpus available to each process. Returns ``None`` if
        :py:func:`os.cpu_count` is unable to determine the total
        number of cores.
    """
    if scomm is None:
        if comm is world:
            scomm = world_scomm
        elif comm is not None:
            scomm = comm.Split_type(MPI.COMM_TYPE_SHARED)

    try:
        nproc_per_node = comm.size // scomm.size
    except AttributeError:
        # This would happend if the default comm is None
        nproc_per_node = 1

    cpu_count = os.cpu_count()

    if cpu_count is not None:
        cpu_count = int(cpu_count // nproc_per_node)

    return cpu_count


def can_allocate(
    nbytes: int,
    scope: str = "shared",
    comm: MPI.Comm | None = world,
    scomm: MPI.Comm | None = None,
) -> bool:
    """Check if nbytes of memory is available to allocate.

    nbytes can be the number of bytes allocated per process,
    number of bytes allocated in all shared memory, or number of
    bytes allocated globally when there are processes which do
    not share memory.

    Parameters
    ----------
    nbytes : int
        Number of bytes that we want to allocate.
    scope : {"shared", "process", "global"}, optional
        Whether to find available memory on a per-node, per-process, or
        global basis. Default is "shared".
    comm : MPI.Comm | None, optional
        MPI communicator.
    scomm : MPI.Comm | None, optional
        MPI shared memory communicator.

    Returns
    -------
    memory_is_available : bool
        True if there is enough memory available.
    """
    # Shared memory comm is used to communicate only between
    # processes on the same node.
    if scomm is None:
        if comm is world:
            scomm = world_scomm
        elif comm is not None:
            scomm = comm.Split_type(MPI.COMM_TYPE_SHARED)
        else:
            scomm = None

    try:
        available_memory_node = available_memory_shared()
    except OSError as e:
        # We probably don't want this to fail if it can't get the available
        # memory, but it should definitey warn about it
        warnings.warn(f"Unable to check available memory: {e!r}", RuntimeWarning)
        return True

    if scope == "shared" or comm is None or scomm is None:
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
    memory_bytes : int
        Memory available in bytes.

    Raises
    ------
    OSError
        If `cgroup`, `memory.stat`, or `memory.usage_in_bytes` files exist
        but cannot be read.
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


# Logging and Error Handling
# ==========================


def enable_mpi_exception_handler():
    """Install an MPI-aware exception handler.

    When enabled, the whole MPI job will abort if *any* MPI process fails. If it's not
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


class MPILogFilter(logging.Filter):
    """Filter log entries by MPI rank.

    Also, this will optionally add MPI rank information, and add an elapsed time
    entry.

    Parameters
    ----------
    add_mpi_info : bool, optional
        Add MPI rank/size info to log records that don't already have it.
    level_rank0 : int, optional
        Log level for messages from rank=0. Default is `INFO`.
    level_all : int, optional
        Log level for messages from all other ranks. Default is `WARN`.
    comm : MPI.Comm | None, optional
        MPI Communicator to use (default :py:obj:`None`).
    """

    def __init__(
        self,
        add_mpi_info: bool = True,
        level_rank0: int = logging.INFO,
        level_all: int = logging.WARN,
        comm: MPI.Comm | None = None,
    ) -> None:
        super().__init__()

        self.add_mpi_info = add_mpi_info

        self.level_rank0 = level_rank0
        self.level_all = level_all

        self.rank = comm.rank if comm else rank
        self.size = comm.size if comm else size

    def filter(self, record: logging.LogRecord) -> bool:
        """Return True if the record should be logged."""
        # Add MPI info if desired
        if not hasattr(record, "mpi_rank") and self.add_mpi_info:
            record.mpi_rank = self.rank
            record.mpi_size = self.size

        # Try to get the rank stored in the record, otherwise
        # assume the rank of the current process. In most cases
        # these should be the same thing
        rank = getattr(record, "mpi_rank", self.rank)

        # Add a new field with the elapsed time in seconds (as a float)
        record.elapsedTime = record.relativeCreated * 1e-3

        if rank == 0:
            return record.levelno >= self.level_rank0

        return record.levelno >= self.level_all


# Comm Management
# ===============


class _close_message:
    def __repr__(self) -> str:
        return "<Close message>"


def active_comm(aprocs: list) -> MPI.Comm | None:
    """Return a communicator consisting of processes listed in `aprocs`."""
    if _comm is None:
        return None

    # create a new communicator from active processes
    return _comm.Create(_comm.Get_group().Incl(aprocs))


def active(aprocs: list) -> MPI.Comm | None:
    """Make processes listed in `aprocs` active, while others wait."""
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


def close(aprocs: list) -> None:
    """Send a message to the waiting processes to close their waiting.

    Waiting process include all processes **NOT** listed in `aprocs`.
    """
    if rank0 and _comm is not None:
        for i in list(set(range(size)) - set(aprocs)):
            _comm.isend(_close_message(), dest=i)


# Distributed Partitioning
# ========================


def partition_list_mpi(
    full_list: list, method: str = "con", comm: MPI.Comm | None = _comm
) -> list:
    """Return the partition of a list specific to the current MPI process.

    Parameters
    ----------
    full_list : list
        The full list to partition.
    method : {"con", "alt", "rand"}, optional
        How to split the list, can be "con": contiguous, "alt": alternating,
        "rand": random. Default is "con".
    comm : MPI.Comm | None, optional
        MPI communicator to use (default :py:obj:`COMM_WORLD`).

    Returns
    -------
    partitioned_list : list
        The sub-list for the current MPI process.
    """
    global rank
    global size

    r = rank
    c = size

    if comm is not None:
        r = comm.rank
        c = comm.size

    return partition_list(full_list, r, c, method=method)


def mpirange(*args: Any, **kargs: Any) -> list:
    """MPI aware version of `range`, each process gets its own sub section."""
    full_list = list(range(*args))

    method = kargs.get("method", "con")
    comm = kargs.get("comm", _comm)

    return partition_list_mpi(full_list, method=method, comm=comm)


def split_all(n: int, comm: MPI.Comm | None = _comm) -> np.ndarray:
    """Split a range of integers ``[0, n)`` into sub-ranges for each MPI Process.

    Parameters
    ----------
    n : int
        Length of range to split.
    comm : MPI.Comm | None, optional
        MPI Communicator to use (default COMM_WORLD).

    Returns
    -------
    split : ndarray
        Array of shape (3, ...) with rows corresponding to:

        - Number for each rank.
        - Starting of each sub-range on a given rank.
        - End of each sub-range.

    See Also
    --------
    :py:func:`~caput.util.arraytools.split_m`, :py:func:`split_local`
    """
    m = size if comm is None else comm.size

    return split_m(n, m)


def split_local(n: int, comm: MPI.Comm | None = _comm) -> np.ndarray:
    """Split a range of integers ``[0, n)`` into sub-ranges for each MPI Process.

    This returns the parameters only for the current rank.

    Parameters
    ----------
    n : int
        Length of range to split.
    comm : MPI.Comm | None
        MPI Communicator to use (default COMM_WORLD).

    Returns
    -------
    split_params: ndarray
        Array of shape (3, ...) with rows corresponding to:

        - Number on this rank.
        - Starting of the sub-range for this rank.
        - End of rank for this rank.

    See Also
    --------
    :py:func:`split_all`, :py:func:`~caput.util.arraytools.split_m`
    """
    pse = split_all(n, comm=comm)
    m = rank if comm is None else comm.rank

    return pse[:, m]


# Distributed Data Manipulation
# =============================


def typemap(dtype: npt.DTypeLike) -> MPI.Datatype:
    """Map a numpy dtype into an MPI_Datatype.

    Parameters
    ----------
    dtype : dtype
        The numpy datatype.

    Returns
    -------
    datatype : MPI.Datatype
        The MPI.Datatype.
    """
    return MPI._typedict[np.dtype(dtype).char]


def gather_local(
    global_array: npt.NDArray,
    local_array: npt.NDArray,
    local_start: Sequence[int],
    root: int = 0,
    comm: MPI.Comm | None = _comm,
) -> None:
    """Gather data array in each process to the global array in `root` process.

    Parameters
    ----------
    global_array : array_like
        The global array which will collect data from `local_array` in each process.
    local_array : array_like
        The local array in each process to be collected to `global_array`.
    local_start : Sequence[int]
        The starting index of the local array to be placed in `global_array`.
    root : int, optional
        The process local array gathered to.
    comm : MPI.Comm | None, optional
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


def parallel_map(
    func: Callable,
    glist: list,
    root: int | None = None,
    method: str = "con",
    comm: MPI.Comm | None = _comm,
) -> list | None:
    """Apply a parallel map using MPI.

    Should be called collectively on the same list. All ranks return the full
    set of results.

    Parameters
    ----------
    func : callable
        Function to apply.
    glist : list
        List of map over. Must be globally defined.
    root : int | None, optional
        Which process should gather the results, all processes will gather the results if None.
    method : {"con", "alt", "rand"}, optional
        How to split `glist` to each process, can be 'con': continuously, 'alt': alternatively, 'rand': randomly. Default is 'con'.
    comm : MPI.Comm | None, optional
        MPI communicator that array is distributed over. Default is the gobal _comm.

    Returns
    -------
    results : list | None
        Global list of results. Returns ``None`` if no partition
        on this rank.
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


def transpose_blocks(
    row_array: npt.NDArray, shape: Sequence[int], comm: MPI.Comm | None = _comm
) -> np.ndarray:
    """Swap 2D matrix split from row-wise to columnwise.

    Take a 2D matrix which is split between processes row-wise and split it
    column wise between processes.

    Parameters
    ----------
    row_array : array_like
        The local section of the global array (split row wise).
    shape : Sequence[int]
        The shape of the global array
    comm : MPI.Comm | None, optional
        MPI communicator that array is distributed over. Default is MPI.COMM_WORLD.

    Returns
    -------
    transposed : ndarray
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

    return recv_buffer.reshape((*shape[:-1], pc))


# IO Operations
# =============


def allocate_hdf5_dataset(
    fname: str,
    dsetname: str,
    shape: tuple[int, ...],
    dtype: npt.DTypeLike,
    comm: MPI.Comm | None = _comm,
) -> tuple[int, int]:
    """Create a hdf5 dataset and return its offset and size.

    The dataset will be created contiguously and immediately allocated,
    however it will not be filled.

    Parameters
    ----------
    fname : str
        Name of the file to write.
    dsetname : str
        Name of the dataset to write (must be at root level).
    shape : tuple[int, ...]
        Shape of the dataset.
    dtype : dtype
        Type of the dataset.
    comm : MPI.Comm | None
        Communicator over which to broadcast results.

    Returns
    -------
    dset_state : tuple
        Two integers:

        - Offset into the file at which the dataset starts (in bytes).
        - Size of the dataset in bytes.
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


def lock_and_write_buffer(obj: object, fname: str, offset: int, size: int) -> None:
    """Write buffer contents to disk at a given locked offset.

    Write the contents of a buffer to disk at a given offset, and explicitly
    lock the region of the file whilst doing so.

    Parameters
    ----------
    obj : object
        Data to write to disk. Must support the buffer protocol (i.e., be
        callable by :py:func:`memoryview`).
    fname : str
        Filename to write.
    offset : int
        Offset into the file to start writing at.
    size : int
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


def parallel_rows_write_hdf5(
    fname: str,
    dsetname: str,
    local_data: npt.ArrayLike,
    shape: tuple[int, ...],
    comm: MPI.Comm | None = _comm,
) -> None:
    """Write out array (distributed across processes row wise) into a HDF5 in parallel."""
    offset, _ = allocate_hdf5_dataset(
        fname, dsetname, shape, local_data.dtype, comm=comm
    )

    lr, sr, _ = split_local(shape[0], comm=comm)

    nc = np.prod(shape[1:])

    lock_and_write_buffer(local_data, fname, offset + sr * nc, lr * nc)
