"""Core classes for a distributed pipeline container.

Classes
-------
- :py:class:`ContainerBase`
- :py:class:`TableBase`
"""

import inspect
from typing import ClassVar

import numpy as np

from ..memdata import memh5


class ContainerBase(memh5.BasicCont):
    """A base class for pipeline containers.

    This class is designed to do much of the work of setting up pipeline
    containers. It should be derived from, and two variables set `_axes` and
    `_dataset_spec`. See the :ref:`Notes <containerbase_notes>` section for details.

    Parameters
    ----------
    data_group : `memh5.MemDiskGroup`
        A container to pass through for making a shallow copy. This is used by
        routine like `caput.tod.concatenate` and generally shouldn't be used
        directly. Either a keyword argument, or the first positional argument.
    axes_from : `memh5.BasicCont`, optional
        Another container to copy axis definitions from. Must be supplied as
        keyword argument.
    attrs_from : `memh5.BasicCont`, optional
        Another container to copy attributes from. Must be supplied as keyword
        argument. This applies to attributes in default datasets too.
    dsets_from : `memh5.BasicCont`, optional
        A container to copy datasets from. Any dataset which an axis whose definition
        has been explicitly set (i.e. does not come from `axes_from`) will not be
        copied.
    copy_from : `memh5.BasicCont`, optional
        Set `axes_from`, `attrs_from` and `dsets_from` to this instance if they are
        not set explicitly.
    skip_datasets : bool, optional
        Skip creating datasets. They must all be added manually with
        `.add_dataset` regardless of the entry in `.dataset_spec`. Default is False.
    distributed : bool, optional
        Should this be a distributed container. Defaults to True.
    comm : mpi4py.MPI.Comm, optional
        The MPI communicator to distribute over. Use COMM_WORLD if not set.
    allow_chunked : bool, optional
        Allow the datasets to be chunked. Default is True.

    kwargs : dict
        Should contain entries for all other axes.

    Notes
    -----
    .. _containerbase_notes:

    Inheritance from other `ContainerBase` subclasses should work as expected,
    with datasets defined in super classes appearing as expected, and being
    overridden where they are redefined in the derived class.

    The variable `_axes` should be a tuple containing the names of axes that
    datasets in this container will use.

    The variable `_dataset_spec` should define the datasets. It's a dictionary
    with the name of the dataset as key. Each entry should be another
    dictionary, the entry 'axes' is mandatory and should be a list of the axes
    the dataset has (these should correspond to entries in `_axes`), as is
    `dtype` which should be a datatype understood by numpy. Other possible
    entries are:

    - `initialise` : if set to `True` the dataset will be created as the
      container is initialised.

    - `distributed` : the dataset will be distributed if the entry is `True`, if
      `False` it won't be, and if not set it will be distributed if the
      container is set to be.

    - `distributed_axis` : the axis to distribute over. Should be a name given
      in the `axes` entry.
    """

    _axes = ()

    _dataset_spec: ClassVar = {}

    convert_attribute_strings = True
    convert_dataset_strings = True
    allow_chunked = True

    def __init__(self, *args, **kwargs):
        # Arguments for pulling in definitions from other containers
        copy_from = kwargs.pop("copy_from", None)
        axes_from = kwargs.pop("axes_from", copy_from)
        attrs_from = kwargs.pop("attrs_from", copy_from)
        dsets_from = kwargs.pop("dsets_from", copy_from)

        # MPI distribution arguments
        dist = kwargs.pop("distributed", True)
        comm = kwargs.pop("comm", None)

        # Extract misc options
        self.allow_chunked = kwargs.pop("allow_chunked", self.allow_chunked)
        skip_datasets = kwargs.pop("skip_datasets", False)

        # Handle the data_group argument. We need to identify if the argument
        # was actually supplied or not (both as a positional or keyword
        # argument), and infer what its value should be, or None if not
        # provided
        if args and "data_group" in kwargs:
            raise TypeError(
                "Received conflicting definitions of `data_group`, as both the first "
                "positional and a keyword argument."
            )
        has_data_group = args or ("data_group" in kwargs)
        data_group = args[0] if args else kwargs.get("data_group", None)

        # Run base initialiser, and exit early if data_group was provided
        super().__init__(data_group=data_group, distributed=dist, comm=comm)

        # If data_group was provided we need to exit early to behave like
        # memh5.MemDiskGroup would have. In this case we're probably trying to
        # create a bare container, or a shallow clone, so don't initialise any
        # datasets. This behaviour is needed to support tod.concatenate
        if has_data_group:
            return

        # Create axis entries
        for axis in self.axes:
            axis_map = None

            # Check if axis is specified in initialiser
            if axis in kwargs:
                axis_map = kwargs[axis]
                copy_axis_attrs = False

                # If axis is an integer, turn into an arange as a default definition
                if isinstance(axis_map, int | np.integer):
                    axis_map = np.arange(axis_map)

            # If no valid map provided in arguments copy from another object if set
            elif axes_from is not None:
                axis_map = axes_from.index_map.get(axis, None)
                copy_axis_attrs = True

            # Set the index_map[axis] if we have a definition, otherwise throw an error
            if axis_map is None:
                raise RuntimeError(f"No definition of axis {axis} supplied.")

            self.create_index_map(axis, axis_map)

            if copy_axis_attrs:
                # Copy over axis attributes if we're copying the axis from another dataset
                memh5.copyattrs(axes_from.index_attrs[axis], self.index_attrs[axis])

        # Iterate over datasets and initialise any that specify it
        if not skip_datasets:
            for name, spec in self.dataset_spec.items():
                if spec.get("initialise"):
                    self.add_dataset(name)

        # Copy over datasets that have compatible axes
        if dsets_from is not None:
            # Get the list of axes names that have been overriden
            changed_axes = {ax for ax in self.axes if ax in kwargs}

            for name in self.dataset_spec.keys():
                if name not in dsets_from:
                    continue

                source_dset = dsets_from[name]
                source_axes = set(source_dset.attrs["axis"])

                # Check if any of the axes of this dataset have been changed, if that's
                # the case then we can't copy the data over
                if not source_axes.isdisjoint(changed_axes):
                    continue

                # The dataset may not have been initialised by default, if not, create
                # it
                if name not in self:
                    self.add_dataset(name)

                self[name][:] = source_dset[:]

        # Copy over attributes
        if attrs_from is not None:
            # Copy attributes from container root
            memh5.copyattrs(attrs_from.attrs, self.attrs)

            # Copy attributes over from any common datasets
            for name in self.dataset_spec.keys():
                if name in self.datasets and name in attrs_from.datasets:
                    attrs_no_axis = {
                        k: v
                        for k, v in attrs_from.datasets[name].attrs.items()
                        if k != "axis"
                    }
                    memh5.copyattrs(attrs_no_axis, self.datasets[name].attrs)

            # Make sure that the __memh5_subclass attribute is accurate
            clspath = self.__class__.__module__ + "." + self.__class__.__name__
            clsattr = self.attrs.get("__memh5_subclass", None)
            if clsattr and (clsattr != clspath):
                self.attrs["__memh5_subclass"] = clspath

    def add_dataset(self, name):
        """Create an empty dataset.

        The dataset must be defined in the specification for the container.

        Parameters
        ----------
        name : string
            Name of the dataset to create.

        Returns
        -------
        dset : `memh5.MemDataset`
        """
        # Normalise name
        name = name.strip("/")

        # Dataset must be specified
        if name not in self.dataset_spec:
            raise RuntimeError(f"Dataset {name} not known.")

        dspec = self.dataset_spec[name]

        # Fetch dataset properties
        axes = dspec["axes"]
        dtype = dspec["dtype"]
        chunks, compression, compression_opts = None, None, None
        if self.allow_chunked:
            chunks = dspec.get("chunks", None)
            compression = dspec.get("compression", None)
            compression_opts = dspec.get("compression_opts", None)

        # Get distribution properties
        dist = self.distributed and dspec.get("distributed", True)
        shape = ()

        # Check that all the specified axes are defined, and fetch their lengths
        for axis in axes:
            if axis not in self.index_map:
                if isinstance(axis, int):
                    l = axis
                else:
                    raise RuntimeError(f"Axis {axis} not defined in index_map")
            else:
                l = len(self.index_map[axis])

            shape += (l,)

        # Fetch distributed axis, and turn into axis index
        dist_axis = (
            dspec["distributed_axis"] if "distributed_axis" in dspec else axes[0]
        )
        dist_axis = list(axes).index(dist_axis)

        # Check chunk dimensions are consistent with axis
        if chunks is not None:
            final_chunks = ()
            for i, l in enumerate(shape):
                final_chunks += (min(chunks[i], l),)
            chunks = final_chunks

        # Create dataset
        dset = self.create_dataset(
            name,
            shape=shape,
            dtype=dtype,
            distributed=dist,
            distributed_axis=dist_axis,
            chunks=chunks,
            compression=compression,
            compression_opts=compression_opts,
        )

        dset.attrs["axis"] = np.array(axes)

        return dset

    def _ensure_chunked(self):
        """Ensure datasets that have chunk/compression specs are chunked.

        For every dataset, check if chunks and compression are set, and
        if not set them to dataset_spec values.
        """
        for dset in self.dataset_spec:
            if dset not in self:
                continue
            if "chunks" in self.dataset_spec[dset] and self[dset].chunks is None:
                # ensure chunks aren't larger than dataset shape
                chunks = ()
                for i, l in enumerate(self[dset].shape):
                    chunks += (min(self.dataset_spec[dset]["chunks"][i], l),)
                self._data._storage_root[dset].chunks = chunks
            if (
                "compression" in self.dataset_spec[dset]
                and self[dset].compression is None
            ):
                self._data._storage_root[dset].compression = self.dataset_spec[dset][
                    "compression"
                ]
            if (
                "compression_opts" in self.dataset_spec[dset]
                and self[dset].compression_opts is None
            ):
                self._data._storage_root[dset].compression_opts = self.dataset_spec[
                    dset
                ]["compression_opts"]

    @property
    def datasets(self):
        """Return the datasets in this container.

        Do not try to add a new dataset by assigning to an item of this
        property. Use `create_dataset` instead.

        Returns
        -------
        datasets : read only dictionary
            Entries are :mod:`caput.memh5` datasets.

        """
        out = {}
        for name, value in self._data.items():
            if not memh5.is_group(value):
                out[name] = value
        return memh5.ro_dict(out)

    @classmethod
    def _class_dataset_spec(cls):
        """Get the inherited set of dataset spec entries."""
        ddict = {}

        # Iterate over the reversed MRO and look for _dataset_spec attributes
        # which get added to a temporary dict. We go over the reversed MRO so
        # that the `ddict.update` overrides datasets in base classes.`
        for cls in inspect.getmro(cls)[::-1]:
            try:
                # NOTE: this is a little ugly as the following line will drop
                # down to base classes if dataset_spec isn't present, and thus
                # try and `update` with the same values again.
                ddict.update(cls._dataset_spec)
            except AttributeError:
                pass

        # Ensure that the dataset_spec is the same order on all ranks
        return {k: ddict[k] for k in sorted(ddict)}

    @property
    def dataset_spec(self):
        """Return a copy of the fully resolved dataset specifiction as a dictionary."""
        ddict = self.__class__._class_dataset_spec()

        # Add in any _dataset_spec found on the instance
        ddict.update(self.__dict__.get("_dataset_spec", {}))

        # Ensure that the dataset_spec is the same order on all ranks
        return {k: ddict[k] for k in sorted(ddict)}

    @classmethod
    def _class_axes(cls):
        """Get the set of axes defined by the container and it's base classes."""
        axes = set()

        # Iterate over the reversed MRO and look for _table_spec attributes
        # which get added to a temporary dict. We go over the reversed MRO so
        # that the `tdict.update` overrides tables in base classes.
        for c in inspect.getmro(cls)[::-1]:
            try:
                axes |= set(c._axes)
            except AttributeError:
                pass

        # This must be the same order on all ranks, so we need to explicitly sort to get around the
        # hash randomization
        return tuple(sorted(axes))

    @property
    def axes(self):
        """The set of axes for this container including any defined on the instance."""
        axes = set(self._class_axes())

        # Add in any axes found on the instance (this is needed to support the table
        # classes where the axes get added at run time)
        axes |= set(self.__dict__.get("_axes", []))

        # This must be the same order on all ranks, so we need to explicitly sort to
        # get around the hash randomization
        return tuple(sorted(axes))

    @classmethod
    def _make_selections(cls, sel_args):
        """Match down-selection arguments to axes of datasets.

        Parses sel_* argument and returns dict mapping dataset names to selections.

        Parameters
        ----------
        sel_args : dict
            Should contain valid numpy indexes as values and axis names (str) as keys.

        Returns
        -------
        dict
            Mapping of dataset names to numpy indexes for downselection of the data.
            Also includes another dict under the key "index_map" that includes
            the selections for those.
        """
        # Check if all those axes exist
        for axis in sel_args.keys():
            if axis not in cls._class_axes():
                raise RuntimeError(f"No '{axis}' axis found to select from.")

        # Build selections dict
        selections = {}
        for name, dataset in cls._class_dataset_spec().items():
            ds_axes = dataset["axes"]
            sel = []
            ds_relevant = False
            for axis in ds_axes:
                if axis in sel_args:
                    sel.append(sel_args[axis])
                    ds_relevant = True
                else:
                    sel.append(slice(None))
            if ds_relevant:
                selections["/" + name] = tuple(sel)

        # add index maps selections
        for axis, sel in sel_args.items():
            selections["/index_map/" + axis] = sel

        return selections


class TableBase(ContainerBase):
    """A base class for containers holding tables of data.

    Similar to the `ContainerBase` class, the container is defined through a
    dictionary given as a `_table_spec` class attribute. The container may also
    hold generic datasets by specifying `_dataset_spec` as with `ContainerBase`.
    See :ref:`Notes <tablebase_notes>` for details.

    Parameters
    ----------
    axes_from : `memh5.BasicCont`, optional
        Another container to copy axis definitions from. Must be supplied as
        keyword argument.
    attrs_from : `memh5.BasicCont`, optional
        Another container to copy attributes from. Must be supplied as keyword
        argument. This applies to attributes in default datasets too.
    kwargs : dict
        Should contain definitions for all other table axes.

    Notes
    -----
    .. _tablebase_notes:

    A `_table_spec` consists of a dictionary mapping table names into a
    description of the table. That description is another dictionary containing
    several entries.

    - `columns` : the set of columns in the table. Given as a list of
      `(name, dtype)` pairs.

    - `axis` : an optional name for the rows of the table. This is automatically
      generated as `'<tablename>_index'` if not explicitly set. This corresponds
      to an `index_map` entry on the container.

    - `initialise` : whether to create the table by default.

    - `distributed` : whether the table is distributed, or common across all MPI ranks.

    An example `_table_spec` entry is::

        _table_spec = {
            'quasars': {
                'columns': [
                    ['ra': np.float64],
                    ['dec': np.float64],
                    ['z': np.float64]
                ],
                'distributed': False,
                'axis': 'quasar_id'
            }
            'quasar_mask': {
                'columns': [
                    ['mask', bool]
                ],
                'axis': 'quasar_id'
            }
        }
    """

    _table_spec: ClassVar = {}

    def __init__(self, *args, **kwargs):
        # Get the dataset specifiction for this class (not any base classes), or
        # an empty dictionary if it does not exist. Do the same for the axes entry..
        dspec = self.__class__.__dict__.get("_dataset_spec", {})
        axes = self.__class__.__dict__.get("_axes", ())

        # Iterate over all table_spec entries and construct dataset specifications for
        # them.
        for name, spec in self.table_spec.items():
            # Get the specifieid axis or if not present create a unique one for
            # this table entry
            axis = spec.get("axis", name + "_index")

            dtype = self._create_dtype(spec["columns"])

            _dataset = {
                "axes": [axis],
                "dtype": dtype,
                "initialise": spec.get("initialise", True),
                "distributed": spec.get("distributed", False),
                "distributed_axis": axis,
            }

            dspec[name] = _dataset

            if axis not in axes:
                axes += (axis,)

        self._dataset_spec = dspec
        self._axes = axes

        super().__init__(*args, **kwargs)

    def _create_dtype(self, columns):
        """Take a dictionary of columns and turn into the appropriate compound data type."""
        dt = []
        for ci, (name, dtype) in enumerate(columns):
            if not isinstance(name, str):
                raise ValueError(f"Column {ci:d} is invalid")
            dt.append((name, dtype))

        return dt

    @property
    def table_spec(self):
        """Return a copy of the fully resolved table specifiction as a dictionary."""
        import inspect

        tdict = {}

        for cls in inspect.getmro(self.__class__)[::-1]:
            try:
                tdict.update(cls._table_spec)
            except AttributeError:
                pass

        return tdict
