"""Property config types."""

from ...config import CaputConfigError, Property
from ...memdata import fileformats

__all__ = ["file_format"]


def file_format(default: str | fileformats.FileFormat | None = None) -> Property:
    """Property type that accepts only "zarr", or "hdf5".

    Returns the selected `caput.fileformat.FileFormat` subclass or `caput.fileformats.HDF5` if `value == default`.

    Parameters
    ----------
    default
        A string or type object specifying the fileformat

    Returns
    -------
    prop
        A property instance setup to validate a file format.

    Raises
    ------
    ValueError
        If the default value is not `"hdf5"` or `"zarr"`.

    Examples
    --------
    Should be used like::

        class Project:

            mode = file_format(default='zarr')
    """
    options = ("hdf5", "zarr")

    def _prop(val):
        if val is None:
            return None

        if issubclass(val, fileformats.FileFormat):
            return val

        if not isinstance(val, str):
            raise CaputConfigError(
                f"Input {val!r} is of type {type(val).__name__} (expected str or None)."
            )

        val = val.lower()

        if val == "hdf5":
            return fileformats.HDF5
        if val == "zarr":
            return fileformats.Zarr

        raise CaputConfigError(f"Input {val!r} needs to be one of {options})")

    if default is not None and (
        (not isinstance(default, str)) or (default.lower() not in options)
    ):
        raise CaputConfigError(f"Default value {default!r} must be in {options}")

    return Property(proptype=_prop, default=default)
