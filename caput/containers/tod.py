"""Time-ordered data containers and utilities.

This module contains data containers, data formats, and utilities based on
:py:mod:`~caput.memdata`. The data represented must have an axis representing time,
and, in particular, concatenating multiple datasets along a time axis must be a
sensible operation.
"""

from __future__ import annotations

import glob
import inspect
from typing import TYPE_CHECKING

import h5py
import numpy as np

from .. import memdata, mpiarray
from ..memdata import _typeutils, fileformats

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any, ClassVar

    import numpy.typing as npt

    from ..memdata._memh5 import FileLike, GroupLike
    from ..memdata.fileformats import FileFormat
    from ..mpiarray import SelectionTupleLike


__all__ = ["TODReader", "TOData", "concatenate"]


class TOData(memdata.BasicCont):
    """Basic time-ordered data container.

    Inherits from :py:class:`~caput.memdata.BasicCont`. A data container with all
    the functionality of its base class but with the concept of a time axis
    which can be concatenated over. Currently the time axis must be the
    fastest-varying axis.
    """

    time_axes: ClassVar[tuple[str, ...]] = ("time",)

    @property
    def time(self) -> np.ndarray[np.floating]:
        """Representation of the `time` axis.

        By convention, this should always return the floating point UTC UNIX time in
        seconds for the *centre* of each time sample.

        Returns
        -------
        time : ndarray
            Array of shape (Ntime,) giving the time centres for each time sample.
        """
        try:
            time = self.index_map["time"][:]["ctime"]
        except (IndexError, ValueError):
            time = self.index_map["time"][:]

        # This method should always return the time centres, so a shift is applied
        # based on the time index_map entry alignment
        alignment = self.index_attrs["time"].get("alignment", 0)

        if alignment != 0:
            time = time + alignment * (abs(np.median(np.diff(time))) / 2)

        return time

    @classmethod
    def from_mult_files(  # noqa: D417
        cls,
        files: FileLike | Sequence[FileLike],
        data_group: GroupLike | None = None,
        start: int | dict | None = None,
        stop: int | dict | None = None,
        datasets: Sequence[str] | None = None,
        dataset_filter: Callable | None = None,
        **kwargs: Any,
    ) -> TOData:
        r"""Create new data object by concatenating a series of files.

        Accepts any parameter supported by :py:func:`.concatenate` (which controls the
        concatenation) or this class's constructor (which controls the initialization of
        each file). By default, each file is opened with `ondisk=True` and `mode='r'`.

        Parameters
        ----------
        files : FileLike
            These are assumed to be identical in every way except along the axis
            representing time, over which they are concatenated. All other data
            and attributes are simply copied from the first entry of the list.
        data_group : GroupLike
            Underlying hdf5 like container that will store the data for the
            BaseData instance.
        start : int | dict, optional
            In the aggregate datasets at what index to start.  Every thing before
            this index is excluded. If provided as a `dict`, the keys should be
            ``data_list[0].time_axes``.
        stop : int | dict, optional
            In the aggregate datasets at what index to stop.  Every thing after
            this index is excluded. If provided as a `dict`, the keys should be
            ``data_list[0].time_axes``.
        datasets : list[str], optional
            Which datasets to include.  Default is all of them.
        dataset_filter : callable, optional
            Function for preprocessing all datasets.  Useful for changing data
            types etc. Takes a dataset as an argument and should return a
            dataset (either h5py or memdata). Optionally may accept a second
            argument that is slice along the time axis, which the filter should
            apply.
        \**kwargs : Any
            Other keyword arguments are passed on to the class's `from_file` method.

        Returns
        -------
        dataset : TOData
            Concatenated time-ordered data.
        """
        if "mode" not in kwargs:
            kwargs["mode"] = "r"
        if "ondisk" not in kwargs:
            kwargs["ondisk"] = True

        files = ensure_file_list(files)
        files = [cls.from_file(f, **kwargs) for f in files]

        return concatenate(
            files,
            out_group=data_group,
            start=start,
            stop=stop,
            datasets=datasets,
            dataset_filter=dataset_filter,
        )

    @staticmethod
    def convert_time(time: Any) -> Any:
        """Overload to provide support for multiple time formats.

        Method accepts scalar times in supported formats and converts them
        to the same format as ``self.time``.
        """
        return time


class TODReader:
    r"""Provides high-level reading of time ordered data.

    Parses and stores meta-data from file headers allowing for the
    interpretation and selection of the data without reading it all from disk.

    Parameters
    ----------
    files : FileLike
        Files containing data. Filename patterns with wild cards (e.g.
        "foo\*.h5") are supported.
    file_format : fileformats.FileFormat, optional
            File format to use. Default `None` (format will be guessed).
    """

    # Controls the association between Reader classes and data classes.
    # Override with subclass of TOData.
    data_class: ClassVar[type[TOData]] = TOData

    def __init__(
        self,
        files: FileLike | Sequence[FileLike],
        file_format: FileFormat | None = None,
    ) -> None:
        # If files is a filename, or pattern, turn into list of files.
        if isinstance(files, str):
            files: list = sorted(glob.glob(files))

        data_empty = self.data_class.from_mult_files(files, datasets=())
        self._data_empty = data_empty

        # Fetch all meta data.
        time = np.copy(data_empty.time)
        first_file, toclose = memdata._memh5.get_file(files[0], file_format=file_format)

        # HACK: we need to aget the time_axes into _copy_non_time_data, this
        # makes it work, although I'm not sure how it ever worked correctly
        # previously
        first_file.time_axes = data_empty.time_axes
        datasets = _copy_non_time_data(first_file)
        # Zarr arrays are flushed automatically flushed and closed
        if toclose and (file_format == fileformats.HDF5):
            first_file.close()

        # Set the metadata attributes.
        self._files: tuple = tuple(files)
        self._time: npt.ArrayLike = time
        self._datasets: list = datasets
        # Set the default selections of the data.
        self._time_sel: tuple = (0, len(self.time))
        self._dataset_sel: list = datasets

    @property
    def files(self) -> tuple:
        """Data files."""
        return self._files

    @property
    def time(self) -> npt.ArrayLike:
        """Time bin centres in data files."""
        return self._time

    @property
    def datasets(self) -> list[str]:
        """Datasets available in data files."""
        return self._datasets

    @property
    def time_sel(self) -> tuple[int, int]:
        """Start and stop indices to read in the time axis.

        Returns
        -------
        indices : tuple
            Start and stop indices for reading along the time axis.
        """
        return self._time_sel

    @time_sel.setter
    def time_sel(self, value: tuple[int, int]) -> None:
        if len(value) != 2:
            raise ValueError("Time selection must be a pair of integers.")

        self._time_sel = (int(value[0]), int(value[1]))

    @property
    def dataset_sel(self) -> list[str]:
        """Which datasets to read.

        Returns
        -------
        datasets : list
            Names of datasets to read.
        """
        return self._dataset_sel

    @dataset_sel.setter
    def dataset_sel(self, value: Sequence[str]) -> None:
        for dataset_name in value:
            if dataset_name not in self.datasets:
                raise ValueError(f"Dataset {dataset_name} not in data files.")

        self._dataset_sel = tuple(value)

    def select_time_range(
        self, start_time: int | None = None, stop_time: int | None = None
    ) -> None:
        """Sets :py:attr:`~TODReader.time_sel` to include a time range.

        The times from the samples selected will have bin centre timestamps
        that are bracketed by the given *start_time* and *stop_time*.

        Parameter time should be in the same format as :py:attr:`.TOData.time`, and
        mush be comparable through standard comparison operator (``<``, ``>=``,
        etc.). Conversion using :py:meth:`.TOData.convert_time` is attempted.

        Parameters
        ----------
        start_time : int | None, optional
            Affects the first element of :py:attr:`~TODReader.time_sel`. Default
            leaves it unchanged.
        stop_time : int | None, optional
            Affects the second element of :py:attr:`~TODReader.time_sel`. Default
            leaves it unchanged.
        """
        if start_time is not None:
            start_time = self.data_class.convert_time(start_time)
            start = np.where(self.time >= start_time)[0][0]
        else:
            start = self.time_sel[0]
        if stop_time is not None:
            stop_time = self.data_class.convert_time(stop_time)
            stop = np.where(self.time < stop_time)[0][-1] + 1
        else:
            stop = self.time_sel[1]

        self.time_sel = (start, stop)

    def read(self, out_group: GroupLike | None = None) -> TOData:
        """Read the selected data.

        Parameters
        ----------
        out_group : GroupLike | None
            Underlying HDF5-like container that will store the data for the
            :py:class:`.TOData` instance.

        Returns
        -------
        tod : TOData
            Data read from :py:attr:`~TODReader.files` based on the selections made
            by user.
        """
        return self.data_class.from_mult_files(
            self.files,
            data_group=out_group,
            start=self.time_sel[0],
            stop=self.time_sel[1],
            datasets=self.dataset_sel,
        )


def concatenate(
    data_list: Sequence[TOData],
    out_group: GroupLike | None = None,
    start: int | dict | None = None,
    stop: int | dict | None = None,
    datasets: Sequence[str] | None = None,
    dataset_filter: Callable | None = None,
    convert_attribute_strings: bool = True,
    convert_dataset_strings: bool = False,
) -> TOData:
    """Concatenate data along the time axis.

    All :py:class:`TOData` objects to be concatenated are assumed to have the
    same datasets and index_maps with compatible shapes and data types.

    Currently only 'time' axis concatenation is supported, and it must be the
    fastest varying index.

    All attributes, history, and other non-time-dependant information is copied
    from the first item.

    Parameters
    ----------
    data_list : list[TOData]
        Sequence of :py:class:`.TOData`. These are assumed to be identical in
        every way except along the axes representing time, over which they
        are concatenated. All other data and attributes are simply copied
        from the first entry of the list.
    out_group : GroupLike | None, optional
        Underlying hdf5 like container that will store the data for the
        BaseData instance.
    start : int | dict, optional
        In the aggregate datasets at what index to start.  Every thing before
        this index is excluded. If provided as a `dict`, the keys should be
        ``data_list[0].time_axes``.
    stop : int | dict, optional
        In the aggregate datasets at what index to stop.  Every thing after
        this index is excluded. If provided as a `dict`, the keys should be
        ``data_list[0].time_axes``.
    datasets : Sequence[str], optional
        Which datasets to include.  Default is all of them.
    dataset_filter : callable, optional
        Function for preprocessing all datasets.  Useful for changing data
        types etc. Takes a dataset as an argument and should return a
        dataset (either h5py or memdata). Optionally may accept a second
        argument that is slice along the time axis, which the filter should
        apply.
    convert_attribute_strings : bool, optional
        Try and convert attribute string types to unicode. Default is `True`.
    convert_dataset_strings : bool, optional
        Try and convert dataset string types to unicode. Default is `False`.

    Returns
    -------
    dataset : TOData
        Concatenated time-ordered data.
    """
    if dataset_filter is None:

        def dataset_filter(d):
            return d

    filter_time_slice = len(inspect.getfullargspec(dataset_filter).args) == 2

    # Inspect first entry in the list to get constant parts..
    first_data = data_list[0]
    concatenation_axes = first_data.time_axes

    # Ensure *start* and *stop* are mappings.
    if not hasattr(start, "__getitem__"):
        start = dict.fromkeys(concatenation_axes, start)
    if not hasattr(stop, "__getitem__"):
        stop = dict.fromkeys(concatenation_axes, stop)

    # Get the length of all axes for which we are concatenating.
    concat_index_lengths = dict.fromkeys(concatenation_axes, 0)
    for data in data_list:
        for index_name in concatenation_axes:
            if index_name not in data.index_map:
                continue
            concat_index_lengths[index_name] += len(data.index_map[index_name])

    # Get real start and stop indexes.
    for axis in concatenation_axes:
        start[axis], stop[axis] = _start_stop_inds(
            start.get(axis, None), stop.get(axis, None), concat_index_lengths[axis]
        )

    if first_data.distributed and not isinstance(out_group, h5py.Group):
        distributed = True
        comm = first_data.comm
    else:
        distributed = False
        comm = None

    # Choose return class and initialize the object.
    out = first_data.__class__(out_group, distributed=distributed, comm=comm)

    # Resolve the index maps. XXX Shouldn't be necessary after fix to
    # _copy_non_time_data.
    for axis, index_map in first_data.index_map.items():
        if axis in concatenation_axes:
            # Initialize the dataset.
            if convert_dataset_strings:
                dtype = _typeutils.dtype_to_unicode(index_map.dtype)
            else:
                dtype = index_map.dtype
            out.create_index_map(
                axis, np.empty(shape=(stop[axis] - start[axis],), dtype=dtype)
            )
        else:
            # Just copy it.
            out.create_index_map(
                axis,
                (
                    _typeutils.ensure_unicode(index_map)
                    if convert_dataset_strings
                    else index_map
                ),
            )
        memdata.copyattrs(first_data.index_attrs[axis], out.index_attrs[axis])

    # Copy over the reverse maps.
    for axis, reverse_map in first_data.reverse_map.items():
        out.create_reverse_map(axis, reverse_map)

    all_dataset_names = _copy_non_time_data(
        data_list,
        out,
        convert_attribute_strings=convert_attribute_strings,
        convert_dataset_strings=convert_dataset_strings,
    )
    if datasets is None:
        dataset_names = all_dataset_names
    else:
        dataset_names = datasets

    current_concat_index_start = dict.fromkeys(concatenation_axes, 0)
    # Now loop over the list and copy the data.
    for data in data_list:
        # Get the concatenation axis lengths for this BaseData.
        current_concat_index_n = {
            axis: len(data.index_map.get(axis, [])) for axis in concatenation_axes
        }
        # Start with the index_map.
        for axis in concatenation_axes:
            axis_finished = current_concat_index_start[axis] >= stop[axis]
            axis_not_started = (
                current_concat_index_start[axis] + current_concat_index_n[axis]
                <= start[axis]
            )
            if axis_finished or axis_not_started:
                continue
            in_slice, out_slice = _get_in_out_slice(
                start[axis],
                stop[axis],
                current_concat_index_start[axis],
                current_concat_index_n[axis],
            )
            out.index_map[axis][out_slice] = (
                _typeutils.ensure_unicode(data.index_map[axis][in_slice])
                if convert_attribute_strings
                else data.index_map[axis][in_slice]
            )
        # Now copy over the datasets and flags.
        this_dataset_names = _copy_non_time_data(
            data,
            convert_attribute_strings=convert_attribute_strings,
            convert_dataset_strings=convert_dataset_strings,
        )
        for name in this_dataset_names:
            dataset = data[name]
            if name not in dataset_names:
                continue
            attrs = dataset.attrs

            # Figure out which axis we are concatenating over.
            for a in _typeutils.bytes_to_unicode(attrs["axis"]):
                if a in concatenation_axes:
                    axis = a
                    break
            else:
                msg = "Dataset %s does not have a valid concatenation axis."
                raise ValueError(msg % name)
            # Figure out where we are in that axis and how to slice it.
            axis_finished = current_concat_index_start[axis] >= stop[axis]
            axis_not_started = (
                current_concat_index_start[axis] + current_concat_index_n[axis]
                <= start[axis]
            )
            if axis_finished or axis_not_started:
                continue
            axis_rate = 1  # Place holder for eventual implementation.
            in_slice, out_slice = _get_in_out_slice(
                start[axis] * axis_rate,
                stop[axis] * axis_rate,
                current_concat_index_start[axis] * axis_rate,
                current_concat_index_n[axis] * axis_rate,
            )

            # Filter the dataset.
            if filter_time_slice:
                dataset = dataset_filter(dataset, in_slice)
            else:
                dataset = dataset_filter(dataset)
            if hasattr(dataset, "attrs"):
                # Some filters modify the attributes; others return a thing
                # without attributes. So we need to check.
                attrs = dataset.attrs

            # Do this *after* the filter, in case filter changed axis order.
            axis_ind = list(_typeutils.bytes_to_unicode(attrs["axis"])).index(axis)

            # Slice input data if the filter doesn't do it.
            if not filter_time_slice:
                in_slice = (slice(None),) * axis_ind + (in_slice,)
                dataset = dataset[in_slice]

            # The time slice filter above will convert dataset from a MemDataset
            # instance to either an MPIArray or np.ndarray (depending on if
            # it is distributed).  Need to convert back to the appropriate
            # subclass of MemDataset for the initialization of output dataset.
            if not isinstance(dataset, memdata.MemDataset):
                if distributed and isinstance(dataset, mpiarray.MPIArray):
                    dataset = memdata.MemDatasetDistributed.from_mpi_array(dataset)
                else:
                    dataset = memdata.MemDatasetCommon.from_numpy_array(dataset)

            # If this is the first piece of data, initialize the output
            # dataset.
            if name not in out:
                shape = dataset.shape
                dtype = dataset.dtype
                full_shape = shape[:axis_ind]
                full_shape += ((stop[axis] - start[axis]) * axis_rate,)
                full_shape += shape[axis_ind + 1 :]
                if distributed and isinstance(dataset, memdata.MemDatasetDistributed):
                    new_dset = out.create_dataset(
                        name,
                        shape=full_shape,
                        dtype=dtype,
                        distributed=True,
                        distributed_axis=dataset.distributed_axis,
                    )
                else:
                    new_dset = out.create_dataset(name, shape=full_shape, dtype=dtype)
                memdata.copyattrs(
                    attrs, new_dset.attrs, convert_strings=convert_attribute_strings
                )

            out_dset = out[name]
            out_slice = (slice(None),) * axis_ind + (out_slice,)

            # Copy the data in.
            out_dtype = out_dset.dtype
            if (
                out_dtype.kind == "V"
                and not out_dtype.fields
                and out_dtype.shape
                and isinstance(out_dset, h5py.Dataset)
            ):
                # Awkward special case for pure subarray dtypes, which h5py and
                # numpy treat differently.
                # Drop down to low level interface. I think this is only
                # nessisary for pretty old h5py.
                mtype = h5py.h5t.py_create(out_dtype)
                mdata = dataset.copy().flat[:]
                mspace = h5py._hl.selections.SimpleSelection(
                    (mdata.size // out_dtype.itemsize,)
                ).id
                fspace = h5py._hl.selections.select(
                    out_dset.shape, out_slice, out_dset.id
                ).id
                out_dset.id.write(mspace, fspace, mdata, mtype)
            else:
                out_dset[out_slice] = dataset[:]
        # Increment the start indexes for the next item of the list.
        for axis in current_concat_index_start.keys():
            current_concat_index_start[axis] += current_concat_index_n[axis]

    return out


def ensure_file_list(files: FileLike | Sequence[FileLike]) -> list[FileLike]:
    """Tries to interpret the input as a sequence of files.

    Expands filename wildcards ("globs") and converts sequences to a list.

    Raises
    ------
    ValueError
        The input could not be interpreted as a list of files.
    """
    if memdata.is_group(files):
        files = [files]
    elif isinstance(files, str):
        files = sorted(glob.glob(files))
    elif hasattr(files, "__iter__"):
        # Copy the sequence and make sure it's mutable.
        files = list(files)
    else:
        raise ValueError("Input could not be interpreted as a list of files.")

    return files


def _copy_non_time_data(
    data: FileLike | Sequence[FileLike],
    out: GroupLike | None = None,
    to_dataset_names: Sequence[str] | None = None,
    convert_attribute_strings: bool = True,
    convert_dataset_strings: bool = False,
) -> list[str]:
    """Crawl data copying everything but time-ordered datasets to out.

    Return list of all time-order dataset names. Leading '/' is stripped off.

    If *out* is `None` do not copy.

    Parameters
    ----------
    data : FileLike
        Input data to crawl. If a list is provided only the first entry is used.
    out : GroupLike
        Output group to copy data into. If `None` no copying is done.
    to_dataset_names : list[str], optional
        List to append time-ordered dataset names to. If `None` a new list is
        created.
    convert_attribute_strings : bool, optional
        Try and convert attribute string types to unicode. Default is `True`.
    convert_dataset_strings : bool, optional
        Try and convert dataset string types to unicode. Default is `False`.

    Returns
    -------
    dataset_names : list[str]
        List of time-ordered dataset names found in `data`.
    """
    if to_dataset_names is None:
        to_dataset_names = []

    if isinstance(data, list):
        # XXX Do something more sophisticated here when/if we aren't getting
        # all our non-time dependant information from the first entry.
        data = data[0]

    # First do a non-recursive walk over the tree to determine which entries are TO
    # datasets, and which are items we need to copy
    to_copy = []
    stack = [data]

    while stack:
        entry = stack.pop()

        if entry.name in ["index_map", "reverse_map"]:
            # XXX exclude index map and reverse map.
            continue

        # Check if this is a dataset with a time axis
        if _dset_has_axis(entry, data.time_axes):
            to_dataset_names.append(entry.name)
        else:
            # Add children into the stack to walk
            if memdata.is_group(entry):
                stack += [entry[k] for k in sorted(entry, reverse=True)]
            to_copy.append(entry)

    # ... then if we need to copy do a second pass iterating over the list of items to
    # copy.
    if out is not None:
        # to_copy should have been constructed in a breadth first order, so parents will
        # be created before their children
        for entry in to_copy:
            if memdata.is_group(entry):
                target = out if entry.name == "/" else out.require_group(entry.name)
            else:
                arr = (
                    _typeutils.ensure_unicode(entry[:])
                    if convert_dataset_strings
                    else entry[:]
                )
                target = out.create_dataset(
                    entry.name,
                    shape=entry.shape,
                    dtype=entry.dtype,
                    data=arr,
                )
            memdata.copyattrs(
                entry.attrs, target.attrs, convert_strings=convert_attribute_strings
            )

    return [n[1:] if n[0] == "/" else n for n in to_dataset_names]


def _dset_has_axis(entry: Any, axes: tuple[str]) -> bool:
    """Check if `entry` is a dataset with an axis named in `axes`."""
    if memdata.is_group(entry):
        return False

    # Assume is a dataset. We need to ensure the output strings are Unicode as h5py may
    # return them as byte strings if the input is an h5py.Dataset
    dset_axes = _typeutils.bytes_to_unicode(entry.attrs.get("axis", ()))

    return len(set(dset_axes).intersection(axes)) > 0


# XXX andata still calls these.
def _start_stop_inds(
    start: np.number | None, stop: np.number | None, ntime: np.number
) -> tuple[np.number, np.number]:
    if start is None:
        start = 0
    elif start < 0:
        start = ntime + start
    if stop is None:
        stop = ntime
    elif stop < 0:
        stop = ntime + stop
    elif stop > ntime:
        stop = ntime
    return start, stop


def _get_in_out_slice(
    start: np.number, stop: np.number, current: np.number, ntime: np.number
) -> tuple[SelectionTupleLike, SelectionTupleLike]:
    out_slice = np.s_[
        max(0, current - start) : min(stop - start, current - start + ntime)
    ]
    in_slice = np.s_[max(0, start - current) : min(ntime, stop - current)]
    return in_slice, out_slice
