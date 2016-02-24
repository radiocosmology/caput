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

import glob

import numpy as np

from . import memh5


class TOData(memh5.BasicCont):

    time_axes = ('time',)

    @classmethod
    def from_files(cls, files, data_group=None, start=None, stop=None,
                   datasets=None, dataset_filter=None, **kwargs):
        """Create new data object by concatenating a series of objects.

        Parameters
        ----------

        Accepts any parameter for :func:`concatenate` (which controls the
        concatenation) or this class's constructor (which controls the
        initialization of each file). By default, each file is opened with
        `ondisk=True` and `mode='r'`.

        """

        if not 'mode' in kwargs:
            kwargs['mode'] = 'r'
        if not 'ondisk' in kwargs:
            kwargs['ondisk'] = True

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


class Reader(object):
    """Provides high level reading of CHIME data.

    You do not want to use this class, but rather one of its inherited classes
    (:class:`CorrReader`, :class:`HKReader`, :class:`WeatherReader`).

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

    # Controls the assotiation between Reader classes and data classes.
    # Override with subclass of TOData.
    data_class = TOData


    def __init__(self, files):

        # If files is a filename, or pattern, turn into list of files.
        if isinstance(files, str):
            files = sorted(glob.glob(files))

        data_empty = data_class.from_files(files, datasets=())

        # Fetch all meta data.
        time = data_empty.time
        datasets = _get_dataset_names(files[0])

        # Set the metadata attributes.
        self._files = tuple(files)
        self._time = time
        self._datasets = datasets
        # Set the default selections of the data.
        self.time_sel = (0, len(self.time))
        self.dataset_sel = datasets


    @property
    def files(self):
        """Data files."""
        return self._files

    @property
    def time(self):
        """Time bin centres in data files."""
        return self._time.copy()

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

        Parameters
        ----------
        start_time : float or :class:`datetime.datetime`
            If a float, this is a Unix/POSIX time. Afftect the first element of
            :attr:`~Reader.time_sel`.  Default leaves it unchanged.
        stop_time : float or :class:`datetime.datetime`
            If a float, this is a Unix/POSIX time. Afftect the second element
            of :attr:`~Reader.time_sel`.  Default leaves it unchanged.

        """
        try:
            from .ephemeris import ensure_unix
        except ValueError:
            from ephemeris import ensure_unix

        if not start_time is None:
            start_time = ensure_unix(start_time)
            start = np.where(self.time >= start_time)[0][0]
        else:
            start = self.time_sel[0]
        if not stop_time is None:
            stop_time = ensure_unix(stop_time)
            stop = np.where(self.time < stop_time)[0][-1] + 1
        else:
            stop = self.time_sel[1]
        self.time_sel = (start, stop)

    def read(self, out_group):
        pass




def concatenate(data_list, out_group=None, start=None, stop=None,
                datasets=None, dataset_filter=None):
    """Concatenate data along the time axis.

    All :class:`TOData` objects to be concatenated are assumed to have the
    same datasets and index_maps with compatible shapes and data types.

    Currently only 'time' axis concatenation is supported, and it must be the
    fastest varying index.

    All attributes, calibration information and history information is copied
    from the first item.

    Parameters
    ----------
    data_list : list of :class:`TOData`. These are assumed to be identicle in
            every way except along the axes representing time, over which they
            are concatenated. All other data and attributes are simply copied
            from the first entry of the list.
    data_group : `h5py.Group`, hdf5 filename or `memh5.Group`
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
    dataset_filter : callable
        Function for preprocessing all datasets.  Useful for changing data
        types etc.  Should return a dataset.


    Returns
    -------
    data : :class:`TOData`

    """

    if dataset_filter is None:
        dataset_filter = lambda d : d

    # Inspect first entry in the list to get constant parts..
    first_data = data_list[0]
    concatenation_axes = first_data.time_axes

    # Ensure *start* and *stop* are mappings.
    if not hasattr(start, '__getitem__'):
        start = { axis : start for axis in concatenation_axes }
    if not hasattr(stop, '__getitem__'):
        stop = { axis : stop for axis in concatenation_axes }

    if concatenation_axes != ('time',):
        print concatenation_axes
        raise NotImplementedError("Generalized concatenation not working yet.")

    # Get the length of all axes for which we are concatenating.
    concat_index_lengths = { axis : 0 for axis in concatenation_axes }
    for data in data_list:
        for index_name in concatenation_axes:
            concat_index_lengths[index_name] += len(data.index_map[index_name])

    # Get real start and stop indexes.
    for axis in concatenation_axes:
        start[axis], stop[axis] = _start_stop_inds(start.get(axis, None),
                stop.get(axis, None), concat_index_lengths[axis])

    # Choose return class and initialize the object.
    out = first_data.__class__(out_group)

    # Resolve the index maps.
    for axis, index_map in first_data.index_map.items():
        if axis in concatenation_axes:
            # Initialize the dataset.
            dtype = index_map.dtype
            out.create_index_map(axis,
                    np.empty(shape=(stop[axis] - start[axis],), dtype=dtype))
        else:
            # Just copy it.
            out.create_index_map(axis, index_map)

    all_dataset_names = _copy_non_time_data(data_list, out)
    all_dataset_names = [n[1:] if n[0] == '/' else n for n in all_dataset_names]
    if datasets is None:
        dataset_names = all_dataset_names
    else:
        dataset_names = datasets

    # XXX cant reference data.datasets or data.flags

    current_concat_index_start = { axis : 0 for axis in concatenation_axes}
    # Now loop over the list and copy the data.
    for data in data_list:
        # Get the concatenation axis lengths for this BaseData.
        current_concat_index_n = { axis : len(data.index_map[axis]) for axis in
                                   concatenation_axes }
        # Start with the index_map.
        for axis in concatenation_axes:
            axis_finished = current_concat_index_start[axis] >= stop[axis]
            axis_not_started = (current_concat_index_start[axis]
                                + current_concat_index_n[axis] <= start[axis])
            if (axis_finished or axis_not_started):
                continue
            in_slice, out_slice = _get_in_out_slice(start[axis], stop[axis],
                    current_concat_index_start[axis],
                    current_concat_index_n[axis])
            out.index_map[axis][out_slice] = data.index_map[axis][in_slice]
        # Now copy over the datasets and flags.
        datasets_and_flags = {'flags/' + k : v for k, v in data.flags.items()}
        datasets_and_flags.update(data.datasets)
        for name, dataset in datasets_and_flags.items():
            if name not in dataset_names:
                continue
            attrs = dataset.attrs
            dataset = dataset_filter(dataset)
            if hasattr(dataset, "attrs"):
                # Some filters modify the attributes; others return a thing
                # without attributes. So we need to check.
                attrs = dataset.attrs

            # For now only support concatenation over minor axis.
            axis = attrs['axis'][-1]
            if axis not in concatenation_axes:
                msg = "Dataset %s does not have a valid concatenation axis."
                raise ValueError(msg % name)
            axis_finished = current_concat_index_start[axis] >= stop[axis]
            axis_not_started = (current_concat_index_start[axis]
                                + current_concat_index_n[axis] <= start[axis])
            if (axis_finished or axis_not_started):
                continue
            # Place holder for eventual implementation of 'axis_rate' attribute.
            axis_rate = 1
            # If this is the first piece of data, initialize the output
            # dataset.
            out_keys = ['flags/' + n for n in  out.flags.keys()]
            out_keys += out.datasets.keys()
            if name not in out_keys:
                shape = dataset.shape
                dtype = dataset.dtype
                full_shape = shape[:-1] + ((stop[axis] - start[axis]) * \
                             axis_rate,)
                new_dset = out.create_dataset(name, shape=full_shape,
                                              dtype=dtype)
                memh5.copyattrs(attrs, new_dset.attrs)
            out_dset = out._data[name]
            in_slice, out_slice = _get_in_out_slice(
                    start[axis] * axis_rate,
                    stop[axis] * axis_rate,
                    current_concat_index_start[axis] * axis_rate,
                    current_concat_index_n[axis] * axis_rate,
                    )
            # Awkward special case for pure subarray dtypes, which h5py and
            # numpy treat differently.
            out_dtype = out_dset.dtype
            if (out_dtype.kind == 'V' and not out_dtype.fields
                and out_dtype.shape and isinstance(out_dset, h5py.Dataset)):
                #index_pairs = zip(range(dataset.shape[-1])[in_slice],
                #                  range(out_dset.shape[-1])[out_slice])
                # Drop down to low level interface. I think this is only
                # nessisary for pretty old h5py.
                from h5py import h5t
                from h5py._hl import selections
                mtype = h5t.py_create(out_dtype)
                mdata = dataset[...,in_slice].copy().flat[:]
                mspace = selections.SimpleSelection(
                        (mdata.size // out_dtype.itemsize,)).id
                fspace = selections.select(out_dset.shape, out_slice,
                                          out_dset.id).id
                out_dset.id.write(mspace, fspace, mdata, mtype)
            else:
                out_dset[...,out_slice] = dataset[...,in_slice]
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
    elif hasattr(files, '__iter__'):
        # Copy the sequence and make sure it's mutable.
        files = list(files)
    else:
        raise ValueError('Input could not be interpreted as a list of files.')
    return files


def _copy_non_time_data(data, out, to_dataset_names=None):

    if to_dataset_names is None:
        to_dataset_names = []

    if isinstance(data, list):
        # XXX Do something more sophisticated here when/if we aren't getting all
        # our non-time dependant information from the first entry.
        data = data[0]._data
        out = out._data

    memh5.copyattrs(data.attrs, out.attrs)
    for key, entry in data.iteritems():
        if key == 'index_map':
            # XXX exclude index map.
            continue
        if memh5.is_group(entry):
            sub_out = out.require_group(key)
            _copy_non_time_data(entry, sub_out, to_dataset_names)
        else:
            # XXX 'time'
            if 'axis' in entry.attrs and 'time' in entry.attrs['axis']:
                to_dataset_names.append(entry.name)
            else:
                out.create_dataset(key, shape=entry.shape, dtype=entry.dtype,
                    data=entry)
                memh5.copyattrs(entry.attrs, g2[key].attrs)
    return to_dataset_names


# XXX andata still calls these.

def _start_stop_inds(start, stop, n):
    if start is None:
        start = 0
    elif start < 0:
        start = n + start
    if stop is None:
        stop = n
    elif stop < 0:
        stop = n + stop
    elif stop > n:
        stop = n
    return start, stop


def _get_in_out_slice(start, stop, current, n):
        out_slice = np.s_[max(0, current - start):current - start + n]
        in_slice = np.s_[max(0, start - current):min(n, stop - current)]
        return in_slice, out_slice

