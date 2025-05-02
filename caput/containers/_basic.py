"""Basic caput containers."""

from .. import memh5
from ._core import ContainerBase


class DataWeightContainer(ContainerBase):
    """A base class for containers with generic data/weight datasets.

    This is meant such that tasks can operate generically over containers with this
    common structure. The data and weight datasets are expected to have the same size,
    though this isn't checked. Subclasses must define `_data_dset_name` and
    `_weight_dset_name`.
    """

    _data_dset_name: str | None = None
    _weight_dset_name: str | None = None

    @property
    def data(self) -> memh5.MemDataset:
        """The main dataset."""
        if self._data_dset_name is None:
            raise RuntimeError(f"Type {type(self)} has not defined `_data_dset_name`.")

        dset = self[self._data_dset_name]

        if not isinstance(dset, memh5.MemDataset):
            raise TypeError(f"/{self._data_dset_name} is not a dataset")

        return dset

    @property
    def weight(self) -> memh5.MemDataset:
        """The weights for each data point."""
        if not self._weight_dset_name:
            raise RuntimeError(
                f"Type {type(self)} has not defined `_weight_dset_name`."
            )

        dset = self[self._weight_dset_name]

        if not isinstance(dset, memh5.MemDataset):
            raise TypeError(f"/{self._weight_dset_name} is not a dataset")

        return dset


class FreqContainer(ContainerBase):
    """Simple container with a frequency axis."""

    _axes = ("freq",)

    @property
    def freq(self):
        """The physical frequencies associated with each index."""
        return self.index_map["freq"][:]
