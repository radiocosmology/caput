"""Core classes for a distributed pipeline container."""

from __future__ import annotations

import inspect
import posixpath
import warnings
from ast import literal_eval
from typing import TYPE_CHECKING

import numpy as np

from .. import memdata
from ..memdata import _typeutils

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any, ClassVar, Literal

    import numpy.typing as npt

    from ..mpiarray import SelectionLike


__all__ = ["Container", "ContainerPrototype", "TableSpec"]


class Container(memdata.MemDiskGroup):
    """Basic high level data container.

    Basic one-level data container that allows any number of datasets in the
    root group but no nesting. Data history tracking (in
    :py:attr:`~.Container.history`) and array axis interpretation (in
    :py:attr:`~.Container.index_map`) is also provided.

    Instead of subclassing this directly, most use casees will be better served
    by subclassing :py:class:`.ContainerPrototype`, which provides additional facilities
    to simplify the specification of container datasets and axes. This class is
    intended to be more flexible - datasets can be added or deleted arbitrarily.

    Notes
    -----
    Parameters are passed through to the :py:class:`~caput.memdata.MemDiskGroup`
    constructor.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Initialize new groups only if writable.
        if self._data.file.mode == "r+":
            self._data.require_group("history")
            self._data.require_group("index_map")
            self._data.require_group("reverse_map")

            if "order" not in self._data["history"].attrs:
                self._data["history"].attrs["order"] = "[]"

    @property
    def history(self) -> memdata.ro_dict:
        """Stores the analysis history for this data.

        Do not try to add a new entry by assigning to an element of this
        property. Use :py:meth:`~.Container.add_history` instead.

        Returns
        -------
        history : ro_dict
            Each entry is a dictionary containing metadata about that stage in
            history.  There is also an 'order' entry which specifies how the
            other entries are ordered in time.
        """
        out = {}
        for name, value in self._data["history"].items():
            warnings.warn(
                f"dataset {self.name} is using a deprecated history format. Read support of "
                "files using this format will be continued for now, but you should "
                "update the instance of caput that wrote this file.",
                DeprecationWarning,
            )
            out[name] = value.attrs

        for name, value in self._data["history"].attrs.items():
            out[name] = value

        # TODO: this seems like a trememndous hack. I've changed it to a safer version of
        # eval, but this should probably be removed
        out["order"] = literal_eval(
            _typeutils.bytes_to_unicode(self._data["history"].attrs["order"])
        )

        return memdata.ro_dict(out)

    @property
    def index_map(self) -> memdata.ro_dict:
        """Stores representions of the axes of datasets.

        The index map contains arrays used to interpret the axes of the
        various datasets. For instance, the 'time', 'prod' and 'freq' axes of
        the visibilities are described in the index map.

        Do not try to add a new index_map by assigning to an item of this
        property. Use :py:meth:`~.Container.create_index_map` instead.

        Returns
        -------
        index_map : ro_dict
            Entries are 1D arrays used to interpret the axes of datasets.
        """
        return memdata.ro_dict({k: v[:] for k, v in self._data["index_map"].items()})

    @property
    def index_attrs(self) -> memdata.ro_dict:
        """Exposes the attributes of each index_map entry.

        Allows the user to implement custom behaviour associated with
        the axis. Assignment to this dictionary does nothing but it does
        allow attribute values to be changed.

        Returns
        -------
        index_attrs : ro_dict
            Attribute dicts for each index_map entry.
        """
        return memdata.ro_dict({k: v.attrs for k, v in self._data["index_map"].items()})

    @property
    def reverse_map(self) -> memdata.ro_dict:
        """Stores mappings between :py:attr:`~.Container.index_map` entries.

        Do not try to add a new reverse_map by assigning to an item of this
        property. Use :py:meth:`~.Container.create_reverse_map` instead.

        Returns
        -------
        reverse_map : ro_dict
            Entries are 1D arrays used to map from product index to stack index.
        """
        out = {}
        for name, value in self._data.get("reverse_map", {}).items():
            out[name] = value[:]
        return memdata.ro_dict(out)

    def group_name_allowed(self, name: str) -> Literal[False]:
        """No groups are exposed to the user. Returns ``False``."""
        return False

    def dataset_name_allowed(self, name: str) -> bool:
        """Datasets may only be created and accessed in the root level group.

        Returns ``True`` if *name* is a path in the root group i.e. '/dataset'.
        """
        parent_name, _ = posixpath.split(name)

        return parent_name == "/"

    def create_index_map(self, axis_name: str, index_map: npt.ArrayLike[Any]) -> None:
        """Create a new index map."""
        self._data["index_map"].create_dataset(axis_name, data=index_map)

    def del_index_map(self, axis_name: str) -> None:
        """Delete an index map."""
        del self._data["index_map"][axis_name]

    def create_reverse_map(self, axis_name: str, index_map: npt.ArrayLike[Any]) -> None:
        """Create a new reverse map."""
        self._data["reverse_map"].create_dataset(axis_name, data=index_map)

    def del_reverse_map(self, axis_name: str) -> None:
        """Delete an index map."""
        del self._data["reverse_map"][axis_name]

    def add_history(self, name: str, history: Mapping | None = None) -> None:
        """Create a new history entry.

        Parameters
        ----------
        name : str
            Name for history entry.
        history : dict | None
            History entry (optional). Needs to be json serializable.

        Notes
        -----
        Previously only dictionaries with depth=1 were supported here. The key/value pairs of these
        where added as attributes to the history group when written to disk. Reading the old
        history format is still supported, however the history is now an attribute itself and
        dictionaries of any depth are allowed as history entries.
        """
        if name == "order":
            raise ValueError(
                '"order" is a reserved name and may not be the name of a history entry.'
            )
        if history is None:
            history = {}
        order = self.history["order"]
        order = [*order, name]

        history_group = self._data["history"]
        history_group.attrs["order"] = str(order)
        history_group.attrs[name] = history

    def redistribute(self, dist_axis: int | str | Sequence[int | str]) -> None:
        """Redistribute parallel datasets along a specified axis.

        Walks the tree of datasets and redistributes any distributed datasets
        found with the specified axis. The underlying dataset objects remain
        the same, but the underlying data arrays are new.

        Parameters
        ----------
        dist_axis : int | str | Sequence[int | str]
            The axis can be specified by an integer index (positive or
            negative), or by a string label which must correspond to an entry in
            the `axis` attribute on the dataset. If a list is supplied, each
            entry is tried in turn, which allows different datasets to be
            redistributed along differently labelled axes.
        """
        if not isinstance(dist_axis, list | tuple):
            dist_axis: list = [dist_axis]

        stack = list(self._data._storage_root.items())

        # Crawl over the dataset tree and redistribute any matching datasets.
        # NOTE: this is done using a non-recursive stack-based tree walk, the previous
        # implementation used a recursive closure which generated a reference
        # cycle and caused the entire container to be kept alive until an
        # explicit gc run. So let this be a warning to be careful in this code.
        while stack:
            _, item = stack.pop()

            # Recurse into subgroups
            if isinstance(item, memdata._Storage):
                stack += list(item.items())

            # Okay, we've found a distributed dataset, let's try and redistribute it
            if isinstance(item, memdata.MemDatasetDistributed):
                naxis = len(item.shape)

                for axis in dist_axis:
                    # Try processing if this is a string
                    if isinstance(axis, str):
                        if "axis" in item.attrs and axis in item.attrs["axis"]:
                            axis = list(item.attrs["axis"]).index(axis)
                        else:
                            continue

                    # Process if axis is an integer
                    elif isinstance(axis, int):
                        # Deal with negative axis index
                        if axis < 0:
                            axis = naxis + axis

                    # Check axis is within bounds
                    if axis >= naxis:
                        continue

                    # Excellent, found a matching axis, time to redistribute
                    item.redistribute(axis)
                    break

                # Note that this clause is on the FOR.
                else:
                    # If we are here we didn't find a matching axis, emit a warning
                    warnings.warn(
                        "Could not find axis (from {dist_axis}) to distribute dataset {name} over."
                    )


class ContainerPrototype(Container):
    """A prototype class used to design containers using axis and dataset specifications.

    This class is designed to do much of the work of setting up pipeline
    containers. It should be subclassed, with two class variables set:
    :py:attr:`~.ContainerPrototype._axes` and :py:attr:`~.ContainerPrototype._dataset_spec`.
    See the :ref:`Notes <containerbase_notes>` section for details.

    Optional Parameters
    ~~~~~~~~~~~~~~~~~~~
    Some combination of the following parameters must br provided when creating
    a new container. In particular, the container requires axis definitions to
    be provided for all axes in :py:attr:`~.ContainerPrototype._axes`, whether by
    direct keyword arguments, or by copying from another container.

    data_group : :py:class:`~caput.memdata.MemDiskGroup`
        A container to pass through for making a shallow copy. This is used by
        routines like :py:func:`~caput.memdata.tod.concatenate` and generally
        shouldn't be used directly. Either a keyword argument, or the first
        positional argument.
    axes_from : :py:class:`.Container`, optional
        Another container to copy axis definitions from. Must be supplied as
        keyword argument.
    attrs_from : :py:class:`.Container`, optional
        Another container to copy dataset attributes from. Must be supplied as keyword
        argument. This applies to attributes in default datasets too.
    dsets_from : :py:class:`.Container`, optional
        A container to copy datasets from. Any dataset which an axis whose definition
        has been explicitly set (i.e. does not come from `axes_from`) will not be
        copied.
    copy_from : :py:class:`.Container`, optional
        Set ``axes_from``, ``attrs_from`` and ``dsets_from`` to this instance if they are
        not set explicitly.
    skip_datasets : bool, optional
        Skip creating datasets. Instead, they will all need to be added manually
        with :py:meth:`~.ContainerPrototype.add_dataset` regardless of the entry in
        :py:attr:`~.ContainerPrototype._dataset_spec`. Default is ``False``.
    distributed : bool, optional
        Should this be a distributed container? Defaults to ``True``.
    comm : :py:obj:`~mpi4py.MPI.Comm`, optional
        The MPI communicator to distribute over. Use COMM_WORLD if not set.
    allow_chunked : bool, optional
        Allow the datasets to be chunked. Default is True.
    kwargs : Any
        Should contain entries for all other axes.

    Notes
    -----
    .. _containerbase_notes:

    Inheritance from other :py:class:`ContainerPrototype` subclasses should work as expected,
    with datasets defined in parent classes appearing as expected, and being
    overridden where they are redefined in the derived class.

    The variable :py:attr:`~.ContainerPrototype._axes` should be a tuple containing the names of axes that
    datasets in this container will use.

    The variable :py:attr:`~.ContainerPrototype._dataset_spec` should define the datasets. It's a dictionary
    with the names of the datasets as keys. The value for each  key should be another dictionary. In that
    sub-dictionary, the key `axes` is mandatory and should be a list of the axes the dataset has (these
    should correspondto entries in `_axes`), as is the key `dtype` which should be a datatype understood by
    numpy. Other possible keys are:

    - `initialise` : if set to `True` the dataset will be created as the
      container is initialised.

    - `distributed` : the dataset will be distributed if the entry is `True`, if
      `False` it won't be, and if not set it will be distributed if the
      container is set to be.

    - `distributed_axis` : the axis to distribute over. Should be a name given
      in the `axes` entry.
    """

    _axes: ClassVar[tuple[str, ...]] = ()

    _dataset_spec: ClassVar[dict[str, dict]] = {}

    convert_attribute_strings: bool = True
    convert_dataset_strings: bool = True
    allow_chunked: bool = True

    def __init__(self, *args: tuple, **kwargs: dict):
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
        # memdata.MemDiskGroup would have. In this case we're probably trying to
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
                memdata.copyattrs(axes_from.index_attrs[axis], self.index_attrs[axis])

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
            memdata.copyattrs(attrs_from.attrs, self.attrs)

            # Copy attributes over from any common datasets
            for name in self.dataset_spec.keys():
                if name in self.datasets and name in attrs_from.datasets:
                    attrs_no_axis = {
                        k: v
                        for k, v in attrs_from.datasets[name].attrs.items()
                        if k != "axis"
                    }
                    memdata.copyattrs(attrs_no_axis, self.datasets[name].attrs)

            # Make sure that the __memh5_subclass attribute is accurate
            clspath = self.__class__.__module__ + "." + self.__class__.__name__
            clsattr = self.attrs.get("__memh5_subclass", None)
            if clsattr and (clsattr != clspath):
                self.attrs["__memh5_subclass"] = clspath

    def add_dataset(self, name: str) -> memdata.MemDataset:
        """Add a new, empty dataset to the container.

        The dataset must be defined in the specification for the container.

        Parameters
        ----------
        name : str
            Name of the dataset to create.

        Returns
        -------
        dataset : memdata.MemDataset
            The created dataset.
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

    def _ensure_chunked(self) -> None:
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
    def datasets(self) -> memdata.ro_dict[str, memdata.MemDataset]:
        """A read-only view of the datasets in this container.

        Do not try to add a new dataset by adding keys to this property.
        Use :py:attr:`~.ContainerPrototype.create_dataset` instead.

        Returns
        -------
        datasets : ro_dict
            Entries are :py:class:`~caput.memdata.MemDataset` datasets.
        """
        out = {}
        for name, value in self._data.items():
            if not memdata.is_group(value):
                out[name] = value
        return memdata.ro_dict(out)

    @classmethod
    def _class_dataset_spec(cls) -> dict[str, dict]:
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
    def dataset_spec(self) -> dict[str, dict]:
        """Return a copy of the fully resolved dataset specifiction as a dictionary."""
        ddict = self.__class__._class_dataset_spec()

        # Add in any _dataset_spec found on the instance
        ddict.update(self.__dict__.get("_dataset_spec", {}))

        # Ensure that the dataset_spec is the same order on all ranks
        return {k: ddict[k] for k in sorted(ddict)}

    @classmethod
    def _class_axes(cls) -> tuple[str, ...]:
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
    def axes(self) -> tuple[str, ...]:
        """The set of axes for this container including any defined on the instance."""
        axes = set(self._class_axes())

        # Add in any axes found on the instance (this is needed to support the table
        # classes where the axes get added at run time)
        axes |= set(self.__dict__.get("_axes", []))

        # This must be the same order on all ranks, so we need to explicitly sort to
        # get around the hash randomization
        return tuple(sorted(axes))

    @classmethod
    def _make_selections(cls, sel_args: dict[str, SelectionLike]) -> dict[str, tuple]:
        """Match down-selection arguments to axes of datasets.

        Parses sel_* argument and returns dict mapping dataset names to selections.

        Parameters
        ----------
        sel_args : dict
            Should contain valid selection entries for axes defined in this container.

        Returns
        -------
        selections : dict
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


class TableSpec(ContainerPrototype):
    """A base class for containers holding tables of data.

    Similar to the :py:class:`ContainerPrototype` class, a container is defined through a
    dictionary given as a :py:attr:`~.TableSpec._table_spec` class attribute when subclassing
    this base class. The container may also hold generic datasets by specifying
    :py:attr:`~.ContainerPrototype._dataset_spec` as with :py:class:`ContainerPrototype`.
    See :ref:`Notes <tablebase_notes>` for details.

    Optional Parameters
    ~~~~~~~~~~~~~~~~~~~
    kwargs : dict
        Should contain definitions for all other table axes.

    Notes
    -----
    .. _tablebase_notes:

    A :py:attr:`_table_spec` consists of a dictionary mapping table names into a
    description of the table. That description is another dictionary containing
    several entries.

    - ``columns`` : the set of columns in the table. Given as a list of
      `(name, dtype)` pairs.

    - ``axis`` : an optional name for the rows of the table. This is automatically
      generated as ``<tablename>_index`` if not explicitly set. This corresponds
      to an `index_map` entry on the container.

    - ``initialise`` : whether to create the table by default.
    - ``distributed`` : whether the table is distributed, or common across all MPI ranks.

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

    _table_spec: ClassVar[dict[str, dict]] = {}

    def __init__(self, *args: tuple, **kwargs: dict):
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

    def _create_dtype(
        self, columns: dict[str, npt.DTypeLike]
    ) -> list[tuple[str, np.dtype]]:
        """Take a dictionary of columns and turn into the appropriate compound data type."""
        dt = []
        for ci, (name, dtype) in enumerate(columns):
            if not isinstance(name, str):
                raise ValueError(f"Column {ci:d} is invalid")
            dt.append((name, dtype))

        return dt

    @property
    def table_spec(self) -> dict[str, dict]:
        """Return a copy of the fully resolved table specifiction as a dictionary."""
        import inspect

        tdict = {}

        for cls in inspect.getmro(self.__class__)[::-1]:
            try:
                tdict.update(cls._table_spec)
            except AttributeError:
                pass

        return tdict
