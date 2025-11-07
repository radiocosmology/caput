r"""A collection of tasks for loading and saving files.

Most of these are just variations on loading and saving :py:class:`~caput.memdata.BasicCont`
containers from different input strings or containers attributes (or combinations thereof).

File Groups
~~~~~~~~~~~
Several tasks accept groups of files as arguments.
These are specified in the YAML file as a dictionary like below.

.. code-block:: yaml

    list_of_file_groups:
        -   tag: first_group  # An optional tag naming the group
            files:
                -   'file1.h5'
                -   'file[3-4].h5'  # Globs are processed
                -   'file7.h5'

        -   files:  # No tag specified, implicitly gets the tag 'group_2'
                -   'another_file1.h5'
                -   'another_file2.h5'


    single_group:
        files: ['file1.h5', 'file2.h5']
"""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from ... import config
from ...memdata import BasicCont, MemDataset, MemDiskGroup, fileformats
from ..exceptions import PipelineStopIteration
from .base import ContainerTask, MPILoggedTask

if TYPE_CHECKING:
    from typing import ClassVar


def list_of_filelists(files):
    r"""Take in a list of lists/glob patterns of filenames.

    Parameters
    ----------
    files : Sequence[PathLike] | Sequence[list[Pathlike]]
        A path or glob pattern (e.g. "/my/data/\*.h5)" or a list of those (or a list of
        lists of those).

    Raises
    ------
    :py:exc:`~caput.config.CaputConfigError`
        If `files` has the wrong format or refers to a file that doesn't exist.

    Returns
    -------
    flattened_list : list[PathLike]
        The input file list. Any glob patterns will be flattened to file path string lists.
    """
    import glob

    f2 = []

    for filelist in files:
        if isinstance(filelist, str):
            if "*" not in filelist and not os.path.isfile(filelist):
                raise config.CaputConfigError(f"File not found: {filelist!s}")
            filelist = glob.glob(filelist)
        elif isinstance(filelist, list):
            for i in range(len(filelist)):
                filelist[i] = list_or_glob(filelist[i])
        else:
            raise config.CaputConfigError("Must be list or glob pattern.")
        f2 = f2 + filelist

    return f2


def list_or_glob(files):
    r"""Take in a list of lists/glob patterns of filenames.

    Parameters
    ----------
    files : PathLike | Sequence[PathLike]
        A path or glob pattern (e.g. /my/data/\*.h5) or a list of those

    Returns
    -------
    file_list : list[PathLike]
        The input file list. Any glob patterns will be flattened to file path string lists.

    Raises
    ------
    :py:exc:`~caput.config.CaputConfigError`
        If `files` has the wrong type or if it refers to a file that doesn't exist.
    """
    import glob

    # If the input was a list, process each element and return as a single flat list
    if isinstance(files, list):
        parsed_files = []
        for f in files:
            parsed_files = parsed_files + list_or_glob(f)
        return parsed_files

    # If it's a glob we need to expand the glob and then call again
    if isinstance(files, str) and any(c in files for c in "*?["):
        return list_or_glob(sorted(glob.glob(files)))

    # We presume a string is an actual path...
    if isinstance(files, str):
        # Check that it exists and is a file (or dir if zarr format)
        if files.endswith(".zarr"):
            if not os.path.isdir(files):
                raise config.CaputConfigError(
                    f"Expected a zarr directory store, but directory not found: {files}"
                )
        else:
            if not os.path.isfile(files):
                raise config.CaputConfigError(f"File not found: {files!s}")

        return [files]

    raise config.CaputConfigError(
        f"Argument must be list, glob pattern, or file path, got {files!r}"
    )


def list_of_filegroups(groups):
    """Process a file group/groups.

    Parameters
    ----------
    groups : dict[str, PathLike] | list[dict[str, PathLike]]
        Dicts should contain keys 'files': An iterable with file path or glob pattern
        strings, 'tag': the group tag str

    Returns
    -------
    file_list : list[PathLike]
        The input groups. Any glob patterns in the 'files' list will be flattened to file
        path strings.

    Raises
    ------
    :py:exc:`~caput.config.CaputConfigError`
        If `groups` has the wrong format.
    """
    import glob

    # Convert to list if the group was not included in a list
    if not isinstance(groups, list):
        groups = [groups]

    # Iterate over groups, set the tag if needed, and process the file list
    # through glob
    for gi, group in enumerate(groups):
        try:
            files = group["files"]
        except KeyError:
            raise config.CaputConfigError("File group is missing key 'files'.")
        except TypeError:
            raise config.CaputConfigError(
                f"Expected type dict in file groups (got {type(group)})."
            )

        if "tag" not in group:
            group["tag"] = f"group_{gi:d}"

        flist = []

        for fname in files:
            if "*" not in fname and not os.path.isfile(fname):
                raise config.CaputConfigError(f"File not found: {fname!s}")
            flist += glob.glob(fname)

        if not len(flist):
            raise config.CaputConfigError(f"No files in group exist ({files!s}).")

        group["files"] = flist

    return groups


class FindFiles(MPILoggedTask):
    """Take a glob or list of files and pass on to other tasks.

    Files are specified as a parameter in the configuration file.

    Attributes
    ----------
    files : PathLike | list[PathLike]
        Can either be a glob pattern, or lists of actual files.
    """

    files = config.Property(proptype=list_or_glob)

    def setup(self):
        """Return list of files specified in the parameters."""
        if not isinstance(self.files, list | tuple):
            raise RuntimeError("Argument must be list of files.")

        return self.files


class SelectionsMixin:
    """Mixin for parsing axis selections, typically from a yaml config.

    Attributes
    ----------
    selections : dict[str, SelectionLike], optional
        A dictionary of axis selections. See below for details.
    allow_index_map : bool, optonal
        If true, selections can be made based on an index_map dataset.
        This cannot be implemented when reading from disk. See below for
        details. Default is ``False``.

    Notes
    -----
    Selections can be given to limit the data read to specified subsets. They can be
    given for any named axis in the container.

    Selections can be given as a slice with an `<axis name>_range` key with either
    `[start, stop]` or `[start, stop, step]` as the value. Alternatively a list of
    explicit indices to extract can be given with the `<axis name>_index` key, and
    the value is a list of the indices. Finally, selection based on an `index_map`
    can be given with specific `index_map` entries with the `<axis name>_map` key,
    which will be converted to axis indices. `<axis name>_range` will take precedence
    over `<axis name>_index`, which will in turn take precedence over `<axis_name>_map`,
    but you should clearly avoid doing this.

    Additionally, index-based selections currently don't work for distributed reads.

    Here's an example in the YAML format that the pipeline uses:

    .. code-block:: yaml

        selections:
            freq_range: [256, 512, 4]  # A strided slice
            stack_index: [1, 2, 4, 9, 16, 25, 36, 49, 64]  # A sparse selection
            stack_range: [1, 14]  # Will override the selection above
            pol_map: ["XX", "YY"] # Select the indices corresponding to these entries
    """

    selections = config.Property(proptype=dict, default={})
    allow_index_map = config.Property(proptype=bool, default=False)

    def setup(self):
        """Resolve the selections."""
        self._sel = self._resolve_sel()

    def _resolve_sel(self):
        """Turn the selection parameters into actual selectable types."""
        sel = {}

        sel_parsers = {
            "range": self._parse_range,
            "index": partial(self._parse_index, type_=int),
            "map": self._parse_index,
        }

        if not self.allow_index_map:
            del sel_parsers["map"]

        # To enforce the precedence of range vs index selections, we rely on the fact
        # that a sort will place the axis_range keys after axis_index keys
        for k in sorted(self.selections or []):
            # Parse the key to get the axis name and type, accounting for the fact the
            # axis name may contain an underscore
            *axis, type_ = k.split("_")
            axis_name = "_".join(axis)

            if type_ not in sel_parsers:
                raise ValueError(
                    f'Unsupported selection type "{type_}", or invalid key "{k}". '
                    "Note that map-type selections require `allow_index_map=True`."
                )

            sel[f"{axis_name}_sel"] = sel_parsers[type_](self.selections[k])

        return sel

    def _parse_range(self, x):
        """Parse and validate a range type selection."""
        if not isinstance(x, list | tuple) or len(x) > 3 or len(x) < 2:
            raise ValueError(
                f"Range spec must be a length 2 or 3 list or tuple. Got {x}."
            )

        for v in x:
            if not isinstance(v, int):
                raise ValueError(f"All elements of range spec must be ints. Got {x}")

        return slice(*x)

    def _parse_index(self, x, type_=object):
        """Parse and validate an index type selection."""
        if not isinstance(x, list | tuple) or len(x) == 0:
            raise ValueError(f"Index spec must be a non-empty list or tuple. Got {x}.")

        for v in x:
            if not isinstance(v, type_):
                raise ValueError(f"All elements of index spec must be {type_}. Got {x}")

        return list(x)


class BaseLoadFiles(SelectionsMixin, ContainerTask):
    """Base class for loading containers from a file on disk.

    Provides the capability to make selections along axes.

    Attributes
    ----------
    distributed : bool
        Whether the file should be loaded distributed across ranks.
        Default is ``True``.
    convert_strings : bool
        Convert strings to unicode when loading. Default is ``True``.
    redistribute : str | None
        An optional axis name to redistribute the container over after it has
        been read. Default is ``None``.
    """

    distributed = config.Property(proptype=bool, default=True)
    convert_strings = config.Property(proptype=bool, default=True)
    redistribute = config.Property(proptype=str, default=None)

    def _load_file(self, filename, extra_message=""):
        # Load the file into the relevant container

        if not os.path.exists(filename):
            raise RuntimeError(f"File does not exist: {filename}")

        self.log.info(f"Loading file {filename} {extra_message}")
        self.log.debug(f"Reading with selections: {self._sel}")

        # If we are applying selections we need to dispatch the `from_file` via the
        # correct subclass, rather than relying on the internal detection of the
        # subclass. To minimise the number of files being opened this is only done on
        # rank=0 and is then broadcast
        if self._sel:
            if self.comm.rank == 0:
                with fileformats.guess_file_format(filename).open(filename, "r") as fh:
                    clspath = MemDiskGroup._detect_subclass_path(fh)
            else:
                clspath = None
            clspath = self.comm.bcast(clspath, root=0)
            new_cls = MemDiskGroup._resolve_subclass(clspath)
        else:
            new_cls = BasicCont

        cont = new_cls.from_file(
            filename,
            distributed=self.distributed,
            comm=self.comm,
            convert_attribute_strings=self.convert_strings,
            convert_dataset_strings=self.convert_strings,
            **self._sel,
        )

        if self.redistribute is not None:
            cont.redistribute(self.redistribute)

        return cont


class LoadFilesFromParams(BaseLoadFiles):
    """Load data from files given in the tasks parameters.

    Attributes
    ----------
    files : Sequence[PathLike]
        Can either be a glob pattern, or lists of actual files.
    """

    files = config.Property(proptype=list_or_glob)

    _file_ind = 0

    def process(self):
        """Load the given files in turn and pass on.

        Returns
        -------
        container : BasicCont
            A container populated with data from the loaded file.
        """
        # Garbage collect to workaround leaking memory from containers.
        # TODO: find actual source of leak
        import gc

        gc.collect()

        if self._file_ind == len(self.files):
            raise PipelineStopIteration

        # Fetch and remove the first item in the list
        file_ = self.files[self._file_ind]

        # Load into a container
        nfiles_str = str(len(self.files))
        message = f"[{self._file_ind + 1: {len(nfiles_str)}}/{nfiles_str}]"
        cont = self._load_file(file_, extra_message=message)

        if "tag" not in cont.attrs:
            # Get the first part of the actual filename and use it as the tag
            tag = os.path.splitext(os.path.basename(file_))[0]

            cont.attrs["tag"] = tag

        self._file_ind += 1

        return cont


class LoadFilesFromAttrs(BaseLoadFiles):
    """Load files from paths constructed using the attributes of another container.

    This class enables the dynamic generation of file paths by formatting a specified
    filename template with attributes from an input container.  It inherits from
    :py:class:`BaseLoadFiles` and provides functionality to load files into a container.

    Attributes
    ----------
    filename : PathLike
        Template for the file path, which can include placeholders referencing attributes
        in the input container.  For example: `rfi_mask_lsd_{lsd}.h5`.  The placeholders
        will be replaced with corresponding attribute values from the input container.
    """

    filename = config.Property(proptype=str)

    def process(self, incont):
        """Load a file based on attributes from the input container.

        Parameters
        ----------
        incont : BasicCont
            Input container whose attributes are used to construct the file path.

        Returns
        -------
        container : BasicCont
            A container populated with data from the loaded file.
        """
        # Construct the filename from the attributes in the input container
        attrs = dict(incont.attrs)
        filename = self.filename.format(**attrs)

        # Use the base class method to load the file
        return self._load_file(filename)


class LoadFilesAndSelect(BaseLoadFiles):
    """Load a collection of files on setup and select specific entries on process.

    Attributes
    ----------
    files : Sequence[PathLike]
        A list of file paths or a glob pattern specifying the files to load.
    key_format : str
        A format string used to generate keys for file selection.  Can reference
        any variables contained in the attributes of the containers.  If `None`,
        files are stored with numerical indices.
    """

    files = config.Property(proptype=list_or_glob)
    key_format = config.Property(proptype=str)

    def setup(self):
        """Load and store files in a dictionary.

        This method iterates through the list of files, loads their contents,
        and stores them in the `self.collection` dictionary. If `key_format`
        is provided, it is used to generate a key based on the file attributes.
        Otherwise, the index of the file in the list is used as the key.
        """
        # Call the baseclass setup to resolve any selections
        super().setup()

        self.collection = {}
        for ff, filename in enumerate(self.files):
            cont = self._load_file(filename)

            if self.key_format is None:
                self.collection[ff] = cont

            else:
                attrs = dict(cont.attrs)
                key = self.key_format.format(**attrs)
                self.collection[key] = cont

    def process(self, incont):
        """Select and return a file from the collection based on the input container.

        If ``key_format`` is provided, the selection key is derived from the attributes
        of the input container.  If the resulting key is not found in the collection,
        a warning is logged, and `None` is returned.

        If ``key_format`` is not provided, files are selected sequentially from
        the collection, cycling back to the beginning if more input containers
        are received than the number of available files.

        Parameters
        ----------
        incont : BasicCont
            Container whose attributes are used to determine the selection key.

        Returns
        -------
        container : BasicCont | None
            The selected file if found, otherwise ``None``.
        """
        if self.key_format is None:
            key = self._count % len(self.collection)
            self.log.info(f"Selecting file in position {key}.")
        else:
            attrs = dict(incont.attrs)
            key = self.key_format.format(**attrs)

            if key not in self.collection:
                self.log.warning(f"Could not find file with label {key}.")
                return None

            self.log.info(f"Selecting file with label {key}.")

        # Return the file with the desired key
        return self.collection[key]


class LoadFilesFromPathAndTag(LoadFilesFromParams):
    """Load files using all combinations of given paths and tags.

    :py:attr:`~.LoadFilesFromPathAndTag.paths` should be provided with `{tag}`
    as a stand-in for places where the tag is inserted. Specifically, tags are insterted
    by calling `path.format(tag=tag)`.

    This is intended to replicate specific patterns which arent available
    with `glob`, such as `/path/to/files/*[tag1, tag2]/*.h5`.

    Attributes
    ----------
    paths : list[PathLike]
        List of files paths, with `{tag}` as a stand-in where tags
        should be inserted.
    tags : list[str]
        List of tags
    """

    paths = config.Property(proptype=list)
    tags = config.Property(proptype=list)

    files = None

    def setup(self):
        """Construct the list of files."""
        # Call baseclass setup to resolve selections
        super().setup()

        self.files = []

        for path in self.paths:
            for tag in self.tags:
                self.files.append(path.format(tag=tag))


class LoadFiles(LoadFilesFromParams):
    """Load data from files passed into the setup routine."""

    files = None

    def setup(self, files):
        """Set the list of files to load.

        Parameters
        ----------
        files : Sequence[PathLike]
            Files to load.
        """
        # Call the baseclass setup to resolve any selections
        super().setup()

        if not isinstance(files, list | tuple):
            raise RuntimeError(f'Argument must be list of files. Got "{files}"')

        self.files = files


class Save(ContainerTask):
    """Save out the input, and pass it on.

    Assumes that the input has a :py:meth:`to_hdf5` method. Appends a *tag* if there is
    a `tag` entry in the attributes, otherwise just appends a count.

    Attributes
    ----------
    root : PathLike
        Root of the file name to output to.
    """

    root = config.Property(proptype=str)

    count = 0

    def next(self, data):
        """Write out the data file.

        Assumes it has a :py:class:`~caput.memdata.MemGroup` interface.

        Parameters
        ----------
        data : MemGroup
            Data to write out.
        """
        if "tag" not in data.attrs:
            tag = self.count
            self.count += 1
        else:
            tag = data.attrs["tag"]

        fname = f"{self.root}_{tag!s}.h5"

        data.to_hdf5(fname)

        return data


class Truncate(ContainerTask):
    """Precision truncate data prior to saving with compression.

    If no configuration is provided, will look for preset values for the
    input container. Any properties defined in the config will override the
    presets.

    If available, each specified dataset will be truncated relative to a
    (specified) weight dataset with the truncation increasing the variance up
    to the specified maximum in `variance_increase`. If there is no specified
    weight dataset then the truncation falls back to using the
    `fixed_precision`.

    Attributes
    ----------
    dataset : dict[str, bool | float | dict[str, float]]
        Datasets to be truncated as keys. Possible values are:

        - ``bool`` : Whether or not to truncate, using default fixed precision.
        - ``float`` : Truncate to this relative precision.
        - ``dict`` : Specify values for `weight_dataset`, `fixed_precision`, `variance_increase`.

    ensure_chunked : bool
        If True, ensure datasets are chunked according to their dataset_spec.
    """

    dataset = config.Property(proptype=dict, default=None)
    ensure_chunked = config.Property(proptype=bool, default=True)

    default_params: ClassVar = {
        "weight_dataset": None,
        "fixed_precision": 1e-4,
        "variance_increase": 1e-3,
    }

    def _get_params(self, container, dset):
        """Load truncation parameters for a dataset from config or container defaults.

        Parameters
        ----------
        container : ContainerBase
            Container class.
        dset : str
            Dataset name

        Returns
        -------
        truncation_params : dict | None
            Returns ``None`` if the dataset shouldn't get truncated.
        """
        # Check if dataset should get truncated at all
        if (self.dataset is None) or (dset not in self.dataset):
            cdspec = container._class_dataset_spec()
            if dset not in cdspec or not cdspec[dset].get("truncate", False):
                self.log.debug(f"Not truncating dataset '{dset}' in {container}.")
                return None
            # Use the dataset spec if nothing specified in config
            given_params = cdspec[dset].get("truncate", False)
        else:
            given_params = self.dataset[dset]

        # Parse config parameters
        params = self.default_params.copy()
        if isinstance(given_params, dict):
            params.update(given_params)
        elif isinstance(given_params, float):
            params["fixed_precision"] = given_params
        elif not given_params:
            self.log.debug(f"Not truncating dataset '{dset}' in {container}.")
            return None

        # Factor of 3 for variance over uniform distribution of truncation errors
        if params["variance_increase"] is not None:
            params["variance_increase"] *= 3

        return params

    def _get_weights(self, container, dset, wdset):
        """Extract the weight dataset and broadcast against the truncation dataset.

        Parameters
        ----------
        container : ContainerBase
            Container class.
        dset : str
            Dataset name
        wdset : str
            Weight dataset name

        Returns
        -------
        weights : array_like
            Array of weights to use in truncation. If `dset` is complex,
            this is scaled by a factor of 2.

        Raises
        ------
        :pe:exc:`KeyError`
            Raised if either `dset` or `wdset` does not exist.
        :py:exc:`ValueError`
            Raised if the weight dataset cannot be broadcast to
            the shape of the dataset to be truncated.
        """
        # Try to get weights from an attribute first
        if hasattr(container, wdset):
            weight = getattr(container, wdset)
        else:
            weight = container[wdset]

        data = container[dset]

        if isinstance(weight, MemDataset):
            # Add missing broadcast axes to the weights dataset
            waxes = weight.attrs.get("axis", [])
            daxes = data.attrs.get("axis", [])
            # Add length-one axes
            slobj = tuple(slice(None) if ax in waxes else np.newaxis for ax in daxes)
            weight = weight[:][slobj]

        # Broadcast `weight` against the shape of the truncation array
        weight = np.broadcast_to(weight, data[:].shape).copy().reshape(-1)

        if np.iscomplexobj(data):
            weight *= 2.0

        return weight

    def process(self, data):
        """Truncate the incoming data.

        The truncation is done *in place*.

        Parameters
        ----------
        data : ContainerBase
            Data to truncate.

        Returns
        -------
        truncated : ContainerBase
            Container with truncated datasets.

        Raises
        ------
        :py:exc:`~caput.config.CaputConfigError`
             If the input data container has no preset values and `fixed_precision` or
             `variance_increase` are not set in the config.
        """
        from ...util import truncate

        if self.ensure_chunked:
            data._ensure_chunked()

        for dset in data.dataset_spec:
            # get truncation parameters from config or container defaults
            specs = self._get_params(type(data), dset)

            if (specs is None) or (dset not in data):
                # Don't truncate this dataset
                continue

            self.log.debug(f"Truncating {dset}")

            old_shape = data[dset][:].shape
            # np.ndarray.reshape must be used with ndarrays
            # MPIArrays use MPIArray.reshape()
            val = np.ndarray.reshape(data[dset][:].view(np.ndarray), data[dset][:].size)

            if specs["weight_dataset"] is None:
                if np.iscomplexobj(data[dset]):
                    data[dset][:].real = truncate.bit_truncate_relative(
                        val.real, specs["fixed_precision"]
                    ).reshape(old_shape)
                    data[dset][:].imag = truncate.bit_truncate_relative(
                        val.imag, specs["fixed_precision"]
                    ).reshape(old_shape)
                else:
                    data[dset][:] = truncate.bit_truncate_relative(
                        val, specs["fixed_precision"]
                    ).reshape(old_shape)
            else:
                wdset = self._get_weights(data, dset, specs["weight_dataset"])
                wdset /= specs["variance_increase"]

                if np.iscomplexobj(data[dset]):
                    data[dset][:].real = truncate.bit_truncate_weights(
                        val.real,
                        wdset,
                        specs["fixed_precision"],
                    ).reshape(old_shape)
                    data[dset][:].imag = truncate.bit_truncate_weights(
                        val.imag,
                        wdset,
                        specs["fixed_precision"],
                    ).reshape(old_shape)
                else:
                    data[dset][:] = truncate.bit_truncate_weights(
                        val,
                        wdset,
                        specs["fixed_precision"],
                    ).reshape(old_shape)

        return data


class ZipZarrContainers(ContainerTask):
    """Zip up a Zarr container into a single file.

    This is useful to save on file quota and speed up IO by combining the chunk
    data into a single file. Note that the file cannot really be updated after
    this process has been performed.

    As this process is IO limited in most cases, it will attempt to parallelise
    the compression across different distinct nodes. That means at most only
    one rank per node will participate.

    Attributes
    ----------
    containers : list[str]
        The names of the Zarr containers to compress. The zipped files will
        have the same names with `.zip` appended.
    remove : bool
        Remove the original data when finished. Defaults to True.
    """

    containers = config.Property(proptype=list)
    remove = config.Property(proptype=bool, default=True)

    _host_rank = None

    def setup(self, _=None):
        """Setup the task.

        This routine does nothing at all with the input, but it means the
        process won't run until the (optional) requirement is received. This
        can be used to delay evaluation until you know that all the files are
        available.
        """
        import socket

        # See if we can find 7z
        path_7z = shutil.which("7z")
        if path_7z is None:
            raise RuntimeError("Could not find 7z on the PATH")
        self._path_7z = path_7z

        # Get the rank -> hostname mapping for all ranks
        my_host = socket.gethostname()
        my_rank = self.comm.rank
        all_host_ranks = self.comm.allgather((my_host, my_rank))

        # Identify the lowest rank running on each node
        unique_hosts = {}
        for host, rank in all_host_ranks:
            if host not in unique_hosts:
                unique_hosts[host] = rank
            else:
                if unique_hosts[host] > rank:
                    unique_hosts[host] = rank

        self._num_hosts = len(unique_hosts)

        # Figure out if this rank is one that needs to do anything
        if unique_hosts[my_host] != my_rank:
            # This is not the lowest rank on the host, so we don't do anything
            self._host_rank = None
        else:
            # This is the lowest rank, so find where we are in the sorted list of all hosts
            self._host_rank = sorted(unique_hosts).index(my_host)
            self.log.debug(f"Lowest rank on {my_host}")

    def process(self):
        """Compress the listed zarr containers.

        Only the lowest rank on each node will participate.
        """
        if self._host_rank is not None:
            # Get the set of containers this rank is responsible for compressing
            my_containers = self.containers[self._host_rank :: self._num_hosts]

            for container in my_containers:
                self.log.info(f"Zipping {container}")

                if not container.endswith(".zarr") or not os.path.isdir(container):
                    raise ValueError(f"{container} is not a valid .zarr directory")

                # Run 7z to zip up the file
                dest_file = container + ".zip"
                src_dir = container + "/."
                command = [self._path_7z, "a", "-tzip", "-mx=0", dest_file, src_dir]
                status = subprocess.run(command, capture_output=True)

                if status.returncode != 0:
                    self.log.debug("Error occurred while zipping. Debug logs follow...")
                    self.log.debug(f"stdout={status.stdout}")
                    self.log.debug(f"stderr={status.stderr}")
                    raise RuntimeError(f"Error occurred while zipping {container}.")

                self.log.info(f"Done zipping. Generated {dest_file}.")

                if self.remove:
                    shutil.rmtree(container)
                    self.log.info(f"Removed original container {container}.")

        self.comm.Barrier()

        raise PipelineStopIteration


class ZarrZipHandle:
    """A handle for keeping track of background Zarr-zipping job."""

    def __init__(self, filename, handle):
        self.filename = filename
        self.handle = handle


class SaveZarrZip(ZipZarrContainers):
    """Save a container as a .zarr.zip file.

    This task saves the output first as a .zarr container, and then starts a background
    job to start turning it into a zip file. It returns a handle to this job. All these
    handles should be fed into a :py:class:`WaitZarrZip` task to ensure the pipeline run does not
    terminate before they are complete.

    This accepts most parameters that a standard task would for saving, including
    compression parameter overrides.
    """

    # This keeps track of the global number of operations run such that we can dispatch
    # the background jobs to different ranks
    _operation_counter = 0

    def setup(self):
        """Check the parameters and determine the ranks to use."""
        if not self.output_name.endswith(".zarr.zip"):
            raise config.CaputConfigError("File output name must end in `.zarr.zip`.")

        # Trim off the .zip suffix and fix the file format
        self.output_name = self.output_name[:-4]
        self.output_format = fileformats.Zarr
        self.save = True

        # Call the baseclass to determine which ranks will do the work
        super().setup()

    # Override next as we don't want the usual mechanism
    def next(self, container):
        """Take a container and save it out as a .zarr.zip file.

        Parameters
        ----------
        container : BasicCont
            Container to save out.

        Returns
        -------
        handle : ZarrZipHandle
            A handle to use to determine if the job has successfully completed. This
            should be given to the :py:class:`.WaitZarrZip` task.
        """
        outfile = self._save_output(container)
        dest_file = outfile + ".zip"
        self.comm.Barrier()

        bg_process = None

        host_rank_to_use = self._operation_counter % self._num_hosts

        if self._host_rank == host_rank_to_use:
            self.log.info(f"Starting background job to zip {outfile}")

            # Run 7z to zip up the file
            dest_file = outfile + ".zip"
            src_dir = outfile + "/."
            command = f"{self._path_7z} a -tzip -mx=0 {dest_file} {src_dir}"

            # If we are to remove the file get the background job to do it immediately
            # after zipping succeeds
            if self.remove:
                command += f" && rm -r {outfile}"

            bg_process = subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

        # Increment the global operations counter
        self.__class__._operation_counter += 1

        return ZarrZipHandle(dest_file, bg_process)


class WaitZarrZip(MPILoggedTask):
    """Collect Zarr-zipping jobs and wait for them to complete."""

    _handles = None

    def next(self, handle):
        """Receive the handles to wait on.

        Parameters
        ----------
        handle : ZarrZipHandle
            The handle to wait on.
        """
        if self._handles is None:
            self._handles = []

        self._handles.append(handle)

    def finish(self):
        """Wait for all Zarr zipping jobs to complete."""
        for h in self._handles:
            self.log.debug(f"Waiting on job processing {h.filename}")

            if h.handle is not None:
                returncode = h.handle.wait()

                if returncode != 0 or not os.path.exists(h.filename):
                    self.log.debug("Error occurred while zipping. Debug logs follow...")
                    self.log.debug(f"stdout={h.handle.stdout}")
                    self.log.debug(f"stderr={h.handle.stderr}")
                    raise RuntimeError(f"Error occurred while zipping {h.filename}.")

            self.comm.Barrier()
            self.log.info(f"Processing job for {h.filename} successful.")
