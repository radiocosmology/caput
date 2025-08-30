""":py:mod:`~.caput.memdata` data type utilities.

This internal module contains utilities to help deal with incompatibilities
between :py:mod:`numpy`, :py:mod:`mpi4py`, and/or HDF5 data types.

This is mostly just conversions between :py:mod:`numpy` "U" unicode strings and
"S" bytestrings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt

    from ._memh5 import DatasetLike


def bytes_to_unicode(s: Any) -> Any:
    """Ensure that a string (or collection of) are unicode.

    Any byte strings found will be transformed into unicode. Standard
    collections are processed recursively. Numpy arrays of byte strings
    are converted. Any other types are returned as is.

    Note that as HDF5 files will often contain ASCII strings which h5py
    converts to byte strings this will be needed even when fully
    transitioned to Python 3.

    Parameters
    ----------
    s : Any
        Object to convert.

    Returns
    -------
    converted_object : Any
        Converted object. If the input is not a byte string, or
        is a collection which does not contain a byte string, it
        is returned as-is.
    """
    if isinstance(s, bytes):
        return s.decode("utf8")

    if isinstance(s, np.ndarray) and s.dtype.kind == "S":
        return s.astype(str)

    if isinstance(s, list | tuple | set):
        return s.__class__(bytes_to_unicode(t) for t in s)

    if isinstance(s, dict):
        return {bytes_to_unicode(k): bytes_to_unicode(v) for k, v in s.items()}

    return s


def dtype_to_unicode(dt: npt.DTypeLike) -> np.dtype:
    """Convert byte strings in a dtype to unicode.

    This will attempt to parse a numpy dtype and convert strings to unicode.

    .. warning:: Custom alignment will not be preserved in these type conversions as
                 the byte and unicode string types are of different sizes.

    Parameters
    ----------
    dt : dtype
        Data type to convert.

    Returns
    -------
    unicode : dtype
        A new datatype with the converted string type.
    """
    return _convert_dtype(dt, "|S", "<U")


def dtype_to_bytestring(dt: npt.DTypeLike) -> np.dtype:
    """Convert unicode strings in a dtype to byte strings.

    This will attempt to parse a numpy dtype and convert strings to bytes.

    .. warning:: Custom alignment will not be preserved in these type conversions as
                 the byte and unicode string types are of different sizes.

    Parameters
    ----------
    dt : dtype
        Data type to convert.

    Returns
    -------
    bytestring : dtype
        A new datatype with the converted string type.
    """
    return _convert_dtype(dt, "<U", "|S")


def _convert_dtype(
    dt: npt.DTypeLike, type_from: np._ByteOrderChar, type_to: np._ByteOrderChar
) -> np.dtype:
    """Convert types in a numpy dtype to another type.

    .. warning:: Custom alignment will not be preserved in these type conversions as
                 the byte and unicode string types are of different sizes.

    Parameters
    ----------
    dt : dtype
        Data type to convert.
    type_from : str
        Type code (with alignment) to find.
    type_to : str
        Type code (with alignment) to convert to.

    Returns
    -------
    datatype : dtype
        A new datatype with the converted string types.
    """

    def _conv(t):
        # If we get a tuple that should mean it's a type with some extended metadata, extract the
        # type and throw away the metadata
        if isinstance(t, tuple):
            t = t[0]
        return t.replace(type_from, type_to)

    # For compound types we must recurse over the full compound type structure
    def _iter_conv(x):
        items = []

        for item in x:
            name = item[0]
            type_ = item[1]

            # Recursively convert the type
            newtype = _iter_conv(type_) if isinstance(type_, list) else _conv(type_)

            items.append((name, newtype))

        return items

    # For scalar types the conversion is easy
    if not dt.names:
        return np.dtype(_conv(dt.str))
    # For compound types we need to iterate through
    return np.dtype(_iter_conv(dt.descr))


def check_byteorder(arr_byteorder: np._ByteOrderChar) -> bool:
    """Test if a native byteorder; if not, check if byteorder matches the architecture.

    Parameters
    ----------
    arr_byteorder : str
        Array byteorder to check.

    Returns
    -------
    byteorder_needs_set : bool
        True if the byteorder should be set to native. False, otherwise.
    """
    if arr_byteorder == "=":
        return False

    if has_matching_byteorder(arr_byteorder):
        return True

    return False


def has_matching_byteorder(arr_byteorder: np._ByteOrderChar) -> bool:
    """Test if byteorder marches the architecture.

    Parameters
    ----------
    arr_byteorder : str
        Array byteorder to check.

    Returns
    -------
    byteorder_matches_architecture : bool
        True if the byteorder matches the architecture.
    """
    from sys import byteorder

    return (arr_byteorder == "<" and byteorder == "little") or (
        arr_byteorder == ">" and byteorder == "big"
    )


def has_kind(dt: npt.DTypeLike, kind: npt._DTypeKind) -> bool:
    """Test if a numpy datatype has any fields of a specified type.

    Parameters
    ----------
    dt : dtype
        Data type to convert.
    kind : str
        Numpy type code character. e.g. "S" for bytestring and "U" for unicode.

    Returns
    -------
    dtype_has_kind : bool
        True if it contains the requested kind.
    """
    # For scalar types the conversion is easy
    if not dt.names:
        return dt.kind == kind

    # For compound types we must recurse over the full compound type structure
    def _iter_conv(x):
        for item in x:
            type_ = item[1]

            # Recursively convert the type
            if isinstance(type_, list) and _iter_conv(type_):
                return True
            if isinstance(type_, tuple) and type_[0][1] == kind:
                return True
            if type_[1] == kind:
                return True

        return False

    return _iter_conv(dt.descr)


def has_unicode(dt: npt.DTypeLike) -> bool:
    """Test if data type contains any unicode fields.

    See :py:func:`.has_kind`.
    """
    return has_kind(dt, "U")


def has_bytestring(dt: npt.DTypeLike) -> bool:
    """Test if data type contains any unicode fields.

    See :py:func:`.has_kind`.
    """
    return has_kind(dt, "S")


def ensure_native_byteorder(arr: npt.ArrayLike[Any]) -> npt.ArrayLike[Any]:
    """If architecture and arr byteorder are the same, ensure byteorder is native.

    Because of https://github.com/mpi4py/mpi4py/issues/177 mpi4py does not handle
    explicit byte order of little endian. A byteorder of native ("=" in numpy) however,
    works fine.

    Parameters
    ----------
    arr : array_like
        Input array.

    Returns
    -------
    converted_array : array_like
    The converted array. If no conversion was required, just returns `arr`.
    """
    if check_byteorder(arr.dtype.byteorder):
        return arr.view(arr.dtype.newbyteorder("="))

    return arr


def ensure_bytestring(arr: npt.ArrayLike[Any]) -> npt.ArrayLike[Any]:
    """If needed convert the array to contain bytestrings not unicode.

    Parameters
    ----------
    arr : array_like
        Input array.

    Returns
    -------
    converted_array : array_like
        The converted array. If no conversion was required, just returns `arr`.
    """
    if has_unicode(arr.dtype):
        return arr.astype(dtype_to_bytestring(arr.dtype))

    return arr


def ensure_unicode(arr: npt.ArrayLike[Any]) -> npt.ArrayLike[Any]:
    """If needed convert the array to contain unicode strings not bytestrings.

    Parameters
    ----------
    arr : array_like
        Input array.

    Returns
    -------
    converted_array : array_like
        The converted array. If no conversion was required, just returns `arr`.
    """
    if has_bytestring(arr.dtype):
        return arr.astype(dtype_to_unicode(arr.dtype))

    return arr


def check_unicode(dset: DatasetLike[Any]) -> npt.ArrayLike[Any]:
    """Test if dataset contains unicode so we can raise an appropriate error.

    If there is no unicode, return the data from the array.

    Parameters
    ----------
    dset : DatasetLike
        Dataset to check.

    Returns
    -------
    converted_array : array_like
        The converted array. If no conversion was required, just returns `arr`.
    """
    if has_unicode(dset.dtype):
        raise TypeError(
            f'Can not write dataset "{dset.name!s}" of unicode type into HDF5.'
        )

    return dset.data
