from typing import overload

import numpy as np
import numpy.typing as npt
import skyfield.iokit
import skyfield.jpllib
from skyfield import timelib
from skyfield.api import Loader
from skyfield.starlib import Star

__all__ = [
    "SkyfieldWrapper",
    "skyfield_star_from_ra_dec",
    "skyfield_time_to_unix",
    "unix_to_skyfield_time",
]

class SkyfieldWrapper:
    mirror_url: str
    _ephemeris_name: str
    _load: Loader | None
    def __init__(
        self,
        path: str | bytes | None = None,
        expire: bool | None = None,
        ephemeris: str = "de421.bsp",
    ) -> None: ...
    @property
    def load(self) -> skyfield.iokit.Loader: ...
    @property
    def path(self) -> str | bytes: ...
    _timescale: timelib.Timescale | None
    @property
    def timescale(self) -> timelib.Timescale: ...
    _ephemeris: skyfield.jpllib.SpiceKernel | UnicodeTranslateError
    @property
    def ephemeris(self) -> skyfield.jpllib.SpiceKernel: ...
    def reload(self) -> None: ...

skyfield_wrapper: SkyfieldWrapper = ...

@overload
def skyfield_time_to_unix(skyfield_time: timelib.Time) -> float: ...
@overload
def skyfield_time_to_unix(
    skyfield_time: npt.ArrayLike[timelib.Time],
) -> np.ndarray[np.floating]: ...
@overload
def unix_to_skyfield_time(unix_time: float) -> timelib.Time: ...
@overload
def unix_to_skyfield_time(
    unix_time: npt.ArrayLike[float],
) -> np.ndarray[timelib.Time]: ...
def skyfield_star_from_ra_dec(
    ra: float, dec: float, name: str | tuple[str, ...] | list[str] = ()
) -> Star: ...
