"""
Data formats for Time Ordered Data.

.. currentmodule:: caput.tod

This module contains data containers, data formats, and utilities based on
:mod:`caput.memh5`. The data represented must have an axis representing time,
and, in particular, concatenating multiple datasets along a time axis must be a
sensible operation.

Classes
=======

.. autosummary::
   :toctree: generated/

   TOData
   Reader


Functions
=========

.. autosummary::
   :toctree: generated/

    concatenate

"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

from future.utils import text_type
from past.builtins import basestring
import glob
import inspect

import numpy as np
import h5py

from . import memh5
from . import misc
from . import mpiarray


class TOData(memh5.BasicCont):
    """Time ordered data.

    Inherits from :class:`caput.memh5.BasicCont`. A data container in with all
    the functionality of its base class but with the concept of a time axis
    which can be concatenated over. Currently the time axis must be the fastest
    varying axis is present.

    Attributes
    ----------
    time

    Methods
    -------
    from_mult_files
    convert_time

    """

    time_axes = ("time",)

    @property
    def time(self):
        """Representation of the "time" axis.

        The value of ``self.index_map['time']``.

        """

        return self.index_map["time"][:]

    @classmethod
    def from_mult_files(
        cls,
        files,
        data_group=None,
        start=None,
        stop=None,
        datasets=None,
        dataset_filter=None,
        **kwargs
    ):
        """Create new data object by concatenating a series of objects.

        Parameters
        ----------

        Accepts any parameter for :func:`concatenate` (which controls the
        concatenation) or this class's constructor (which controls the
        initialization of each file). By default, each file is opened with
        `ondisk=True` and `mode='r'`.

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
    def convert_time(time):
        """Overload to provide support for multiple time formats.

        Method accepts scalar times in supported formats and converts them
        to the same format as ``self.time``.

        """

        return time


class Reader(object):
    """Provides high level reading of time ordered data.

    Parses and stores meta-data from file headers allowing for the
    interpretation and selection of the data without reading it all from disk.

    Parameters
    ----------
    files : filename, `h5py.File` or list there-of or filename pattern
        Files containing data. Filename patterns with wild cards (e.g.
        "foo*.h5") are supported.

    Attributes
    ----------
    files
    time_sel
    time

    Methods
    -------
    select_time_range
    read

    """

    # Controls the association between Reader classes and data classes.
    # Override with subclass of TOData.
    data_class = TOData

    def __init__(self, files):

        # If files is a filename, or pattern, turn into list of files.
        if isinstance(files, basestring):
            files = sorted(glob.glob(files))

        data_empty = self.data_class.from_mult_files(files, datasets=())
        self._data_empty = data_empty

        # Fetch all meta data.
        time = np.copy(data_empty.time)
        first_file, toclose = memh5.get_h5py_File(files[0])
        datasets = _copy_non_time_data(first_file)
        if toclose:
            first_file.close()

        # Set the metadata attributes.
        self._files = tuple(files)
        self._time = time
        self._datasets = datasets
        # Set the default selections of the data.
        self._time_sel = (0, len(self.time))
        self._dataset_sel = datasets

    @property
    def files(self):
        """Data files."""
        return self._files

    @property
    def time(self):
        """Time bin centres in data files."""
        return self._time

    @property
    def datasets(self):
        """Datasets available in data files."""
        return self._datasets

    @property
    def time_sel(self):
        """Start and stop indices to read in the frequency axis.

        Returns
        -------
        time_sel : pair of ints
            Start and stop indices for reading along the time axis.

        """

        return self._time_sel

    @time_sel.setter
    def time_sel(self, value):
        if len(value) != 2:
            msg = "Time selection must be a pair of integers."
            raise ValueError(msg)
        self._time_sel = (int(value[0]), int(value[1]))

    @property
    def dataset_sel(self):
        """"Which datasets to read.

        Returns
        -------
        dataset_sel : tuple of strings
            Names of datasets to read.

        """
        return self._dataset_sel

    @dataset_sel.setter
    def dataset_sel(self, value):
        for dataset_name in value:
            if dataset_name not in self.datasets:
                msg = "Dataset %s not in data files." % dataset_name
                raise ValueError(msg)
        self._dataset_sel = tuple(value)

    def select_time_range(self, start_time=None, stop_time=None):
        """Sets :attr:`~Reader.time_sel` to include a time range.

        The times from the samples selected will have bin centre timestamps
        that are bracketed by the given *start_time* and *stop_time*.

        Parameter time should be in the same format as :attr:`TOData.time`, and
        mush be comparable through standard comparison operator (``<``, ``>=``,
        etc.). Conversion using :meth:`TOData.convert_time` is attempted.

        Parameters
        ----------
        start_time : scalar time
            Affects the first element of :attr:`~Reader.time_sel`.  Default
            leaves it unchanged.
        stop_time : scalar time
            Affects the second element of :attr:`~Reader.time_sel`.  Default
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

    def read(self, out_group=None):
        """Read the selected data.

        Parameters
        ----------
        out_group : `h5py.Group`, hdf5 filename or `memh5.Group`
            Underlying hdf5 like container that will store the data for the
            BaseData instance.

        Returns
        -------
        data : :class:`TOData`
            Data read from :attr:`~Reader.files` based on the selections made
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
    data_list,
    out_group=None,
    start=None,
    stop=None,
    datasets=None,
    dataset_filter=None,
    convert_attribute_strings=False,
    convert_dataset_strings=False,
):
    """Concatenate data along the time axis.

    All :class:`TOData` objects to be concatenated are assumed to have the
    same datasets and index_maps with compatible shapes and data types.

    Currently only 'time' axis concatenation is supported, and it must be the
    fastest varying index.

    All attributes, history, and other non-time-dependant information is copied
    from the first item.

    Parameters
    ----------
    data_list : list of :class:`TOData`. These are assumed to be identical in
            every way except along the axes representing time, over which they
            are concatenated. All other data and attributes are simply copied
            from the first entry of the list.
    out_group : `h5py.Group`, hdf5 filename or `memh5.Group`
            Underlying hdf5 like container that will store the data for the
            BaseData instance.
    start : int or dict with keys ``data_list[0].time_axes``
        In the aggregate datasets at what index to start.  Every thing before
        this index is excluded.
    stop : int or dict with keys ``data_list[0].time_axes``
        In the aggregate datasets at what index to stop.  Every thing after
        this index is excluded.
    datasets : sequence of strings
        Which datasets to include.  Default is all of them.
    dataset_filter : callable with one or two arguments
        Function for preprocessing all datasets.  Useful for changing data
        types etc. Takes a dataset as an arguement and should return a
        dataset (either h5py or memh5). Optionally may accept a second
        argument that is slice along the time axis, which the filter should
        apply.
    convert_attribute_strings : bool, optional
        Try and convert attribute string types to unicode. Default is `False`.
    convert_dataset_strings : bool, optional
        Try and convert dataset string types to unicode. Default is `False`.

    Returns
    -------
    data : :class:`TOData`

    """

    if dataset_filter is None:

        def dataset_filter(d):
            return d

    filter_time_slice = len(misc.getfullargspec(dataset_filter).args) == 2

    # Inspect first entry in the list to get constant parts..
    first_data = data_list[0]
    concatenation_axes = first_data.time_axes

    # Ensure *start* and *stop* are mappings.
    if not hasattr(start, "__getitem__"):
        start = {axis: start for axis in concatenation_axes}
    if not hasattr(stop, "__getitem__"):
        stop = {axis: stop for axis in concatenation_axes}

    # Get the length of all axes for which we are concatenating.
    concat_index_lengths = {axis: 0 for axis in concatenation_axes}
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
            dtype = index_map.dtype
            out.create_index_map(
                axis, np.empty(shape=(stop[axis] - start[axis],), dtype=dtype)
            )
        else:
            # Just copy it.
            out.create_index_map(axis, index_map)

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

    current_concat_index_start = {axis: 0 for axis in concatenation_axes}
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
            out.index_map[axis][out_slice] = data.index_map[axis][in_slice]
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
            for a in memh5.bytes_to_unicode(attrs["axis"]):
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
            axis_ind = list(memh5.bytes_to_unicode(attrs["axis"])).index(axis)

            # Slice input data if the filter doesn't do it.
            if not filter_time_slice:
                in_slice = (slice(None),) * axis_ind + (in_slice,)
                dataset = dataset[in_slice]

            # The time slice filter above will convert dataset from a MemDataset
            # instance to either an MPIArray or np.ndarray (depending on if
            # it is distributed).  Need to convert back to the appropriate
            # subclass of MemDataset for the initialization of output dataset.
            if not isinstance(dataset, memh5.MemDataset):
                if distributed and isinstance(dataset, mpiarray.MPIArray):
                    dataset = memh5.MemDatasetDistributed.from_mpi_array(dataset)
                else:
                    dataset = memh5.MemDatasetCommon.from_numpy_array(dataset)

            # If this is the first piece of data, initialize the output
            # dataset.
            if name not in out:
                shape = dataset.shape
                dtype = dataset.dtype
                full_shape = shape[:axis_ind]
                full_shape += ((stop[axis] - start[axis]) * axis_rate,)
                full_shape += shape[axis_ind + 1 :]
                if distributed and isinstance(dataset, memh5.MemDatasetDistributed):
                    new_dset = out.create_dataset(
                        name,
                        shape=full_shape,
                        dtype=dtype,
                        distributed=True,
                        distributed_axis=dataset.distributed_axis,
                    )
                else:
                    new_dset = out.create_dataset(name, shape=full_shape, dtype=dtype)
                memh5.copyattrs(
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
                from h5py import h5t
                from h5py._hl import selections

                mtype = h5t.py_create(out_dtype)
                mdata = dataset.copy().flat[:]
                mspace = selections.SimpleSelection(
                    (mdata.size // out_dtype.itemsize,)
                ).id
                fspace = selections.select(out_dset.shape, out_slice, out_dset.id).id
                out_dset.id.write(mspace, fspace, mdata, mtype)
            else:
                out_dset[out_slice] = dataset[:]
        # Increment the start indexes for the next item of the list.
        for axis in current_concat_index_start.keys():
            current_concat_index_start[axis] += current_concat_index_n[axis]

    return out


def ensure_file_list(files):
    """Tries to interpret the input as a sequence of files

    Expands filename wildcards ("globs") and casts sequeces to a list.

    """

    if memh5.is_group(files):
        files = [files]
    elif isinstance(files, basestring):
        files = sorted(glob.glob(files))
    elif hasattr(files, "__iter__"):
        # Copy the sequence and make sure it's mutable.
        files = list(files)
    else:
        raise ValueError("Input could not be interpreted as a list of files.")
    return files


def _copy_non_time_data(
    data,
    out=None,
    to_dataset_names=None,
    convert_attribute_strings=False,
    convert_dataset_strings=False,
):
    """Crawl data copying everything but time-ordered datasets to out.

    Return list of all time-order dataset names. Leading '/' is stripped off.

    If *out* is `None` do not copy.

    """

    if to_dataset_names is None:
        to_dataset_names = []

    if isinstance(data, list):
        # XXX Do something more sophisticated here when/if we aren't getting
        # all our non-time dependant information from the first entry.
        data = data[0]

    if out is not None:
        memh5.copyattrs(
            data.attrs, out.attrs, convert_strings=convert_attribute_strings
        )

    for key, entry in data.items():
        if key in ["index_map", "reverse_map"]:
            # XXX exclude index map and reverse map.
            continue
        if memh5.is_group(entry):
            if out is not None:
                sub_out = out.require_group(key)
            else:
                sub_out = None
            _copy_non_time_data(
                entry,
                sub_out,
                to_dataset_names,
                convert_attribute_strings,
                convert_dataset_strings,
            )
        else:
            # Check if any axis is a 'time' axis
            if "axis" in entry.attrs and set(data.time_axes).intersection(
                memh5.bytes_to_unicode(entry.attrs["axis"])
            ):
                to_dataset_names.append(entry.name)
            elif out is not None:
                arr = (
                    memh5.ensure_unicode(entry.data)
                    if convert_dataset_strings
                    else entry.data
                )
                out.create_dataset(key, shape=entry.shape, dtype=entry.dtype, data=arr)
                memh5.copyattrs(
                    entry.attrs, out[key].attrs, convert_strings=convert_dataset_strings
                )
    to_dataset_names = [n[1:] if n[0] == "/" else n for n in to_dataset_names]
    return to_dataset_names


# XXX andata still calls these.


def _start_stop_inds(start, stop, ntime):
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


def _get_in_out_slice(start, stop, current, ntime):
    out_slice = np.s_[
        max(0, current - start) : min(stop - start, current - start + ntime)
    ]
    in_slice = np.s_[max(0, start - current) : min(ntime, stop - current)]
    return in_slice, out_slice
