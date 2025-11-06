"""Basic caput containers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import ClassVar

    import numpy as np

from .. import memdata
from ._core import ContainerBase


class DataWeightContainer(ContainerBase):
    """A base class for containers with generic data/weight datasets.

    This is meant to be a general-purpose container providing a common structure
    for generic operations . The data and weight datasets are expected to have the
    same size, though this isn't checked. Subclasses must define
    :py:attr:`~.DataWeightContainer._data_dset_name` and
    :py:attr:`~.DataWeightContainer._weight_dset_name`.
    """

    _data_dset_name: ClassVar[str | None] = None
    _weight_dset_name: ClassVar[str | None] = None

    @property
    def data(self) -> memdata.MemDataset:
        """The main dataset."""
        if self._data_dset_name is None:
            raise RuntimeError(f"Type {type(self)} has not defined `_data_dset_name`.")

        dset = self[self._data_dset_name]

        if not isinstance(dset, memdata.MemDataset):
            raise TypeError(f"/{self._data_dset_name} is not a dataset")

        return dset

    @property
    def weight(self) -> memdata.MemDataset:
        """The weights for each data point."""
        if not self._weight_dset_name:
            raise RuntimeError(
                f"Type {type(self)} has not defined `_weight_dset_name`."
            )

        dset = self[self._weight_dset_name]

        if not isinstance(dset, memdata.MemDataset):
            raise TypeError(f"/{self._weight_dset_name} is not a dataset")

        return dset


class FreqContainer(ContainerBase):
    """A simple container with a frequency axis."""

    _axes: ClassVar[tuple[str, ...]] = ("freq",)

    @property
    def freq(self) -> np.ndarray:
        """The physical frequencies associated with each index."""
        return self.index_map["freq"][:]
