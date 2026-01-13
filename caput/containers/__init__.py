"""A high-level in-memory data container format for :py:mod:`caput.pipeline`.

Containers are built on top of memdata's :py:class:`~caput.memdata.MemDiskGroup` and
provide a structured way to hold multi-dimensional data arrays. A basic interface is
provided through :py:class:`.Container`, while :py:class:`.ContainerPrototype` and
:py:class:`.TableSpec` act as base classes to define new containers based on named
axis and dataset specifications.

Examples
--------
Containers are subclasses of :py:class:`ContainerPrototype`. New containers are defined by
specifying the axes and datasets in the subclass:

>>> from caput.containers import ContainerPrototype
>>> import numpy as np
>>>
>>> class MyContainer(ContainerPrototype):
...     _axes = (
...         "time",
...         "freq",
...     )
...     _dataset_spec = {
...         "data": {
...             "axes": [
...                 "time",
...                 "freq",
...             ],
...             "dtype": np.float32,
...             "initialise": True,
...             "compression": None,
...             "distributed": False,
...         },
...     }

When a container is instantiated, the datasets defined in :py:attr:`~.ContainerPrototype._dataset_spec`
are created with the specified properties. The axes define the dimensions of the datasets,
and must be provided in the constructor, either by specifying the `index_map` directly
or by copying from another container.

>>> sample1 = MyContainer(time=100, freq=256)
>>> sample2 = MyContainer(copy_from=sample1)
>>> assert np.array_equal(sample1["data"][:], sample2["data"][:])
>>> assert np.array_equal(sample1.index_map["freq"][:], sample2.index_map["freq"][:])

Container Specification
~~~~~~~~~~~~~~~~~~~~~~~
Axes
^^^^
A container must specify a set of named axes that define the dimensions of its
datasets. Each axis must be associated with a corresponding entry in the `index_map`
dataset. Axes are specified via the :py:attr:`_axes` class attribute, which is a list
of strings, and are inherited from parent classes.

Dataset Spec
^^^^^^^^^^^^
Each container must define a :py:attr:`_dataset_spec` class attribute, which is a dictionary
mapping dataset names to their specifications. Each specification is itself a dictionary
that defines properties such as axes, data type, and compression options. Dataset specs
are **not** inherited from parent classes; each container must define its own complete spec.
More information on dataset specifications can be found in :py:class:`.ContainerPrototype`.
"""

from ._basic import (
    DataWeightContainer as DataWeightContainer,
    FreqContainer as FreqContainer,
)
from ._core import (
    Container as Container,
    ContainerPrototype as ContainerPrototype,
    TableSpec as TableSpec,
)
from ._util import (
    empty_like as empty_like,
    copy_datasets_filter as copy_datasets_filter,
)
from . import tod as tod

# Try to import bitshuffle to set the default compression options
try:
    import bitshuffle.h5

    COMPRESSION = bitshuffle.h5.H5FILTER
    COMPRESSION_OPTS = (0, bitshuffle.h5.H5_COMPRESS_LZ4)
except ImportError:
    COMPRESSION = None
    COMPRESSION_OPTS = None
