from collections.abc import Callable
from typing import TypeVar, overload

import numpy as np
import numpy.typing as npt
from skyfield.jpllib import ChebyshevPosition
from skyfield.starlib import Star
from skyfield.timelib import Time
from skyfield.toposlib import Topos
from skyfield.units import Angle
from skyfield.vectorlib import VectorSum

from . import skyfield as csf
from . import time as ctime

__all__ = ["Observer"]

_SkySourceLike = TypeVar("_SkySourceLike", bound=Star | VectorSum | ChebyshevPosition)

class Observer:
    longitude: float
    latitude: float
    altitude: float
    lsd_start_day: float
    skyfield: csf.SkyfieldWrapper
    def __init__(
        self,
        lon: float = 0.0,
        lat: float = 0.0,
        alt: float = 0.0,
        lsd_start: int | float | None = None,
        sf_wrapper: csf.SkyfieldWrapper | None = None,
    ) -> None: ...
    _obs: Topos | None
    def get_current_lsd(self) -> np.ndarray[np.floating]: ...
    def skyfield_obs(self) -> Topos: ...
    @overload
    def unix_to_lsa(self, time: np.floating) -> np.floating: ...
    @overload
    def unix_to_lsa(
        self, time: npt.ArrayLike[np.floating]
    ) -> np.ndarray[np.floating]: ...
    lsa = unix_to_lsa
    @overload
    def lsa_to_unix(self, lsa: np.floating, time0: np.floating) -> np.floating: ...
    @overload
    def lsa_to_unix(
        self, lsa: npt.ArrayLike[np.floating], time0: npt.ArrayLike[np.floating]
    ) -> np.ndarray[np.floating]: ...
    def lsd_zero(self) -> np.ndarray[np.floating]: ...
    @overload
    def unix_to_lsd(self, time: np.floating) -> np.floating: ...
    @overload
    def unix_to_lsd(
        self, time: npt.ArrayLike[np.floating]
    ) -> np.ndarray[np.floating]: ...
    lsd = unix_to_lsd
    @overload
    def lsd_to_unix(self, lsd: np.floating) -> np.floating: ...
    @overload
    def lsd_to_unix(
        self, lsd: npt.ArrayLike[np.floating]
    ) -> np.ndarray[np.floating]: ...
    @overload
    def unix_to_lst(self, unix: np.floating) -> np.floating: ...
    @overload
    def unix_to_lst(
        self, unix: npt.ArrayLike[np.floating]
    ) -> np.ndarray[np.floating]: ...
    lst = unix_to_lst
    def transit_RA(
        self, time: npt.ArrayLike[np.floating] | np.floating
    ) -> np.floating: ...
    @overload
    def transit_times(
        self,
        source: _SkySourceLike | float,
        t0: ctime.TimeLike,
        t1: ctime.TimeLike | None = None,
        step: float | None = None,
        lower: bool = False,
        return_dec: bool = False,
    ) -> np.ndarray[np.floating]: ...
    @overload
    def transit_times(
        self,
        source: _SkySourceLike | float,
        t0: ctime.TimeLike,
        t1: ctime.TimeLike | None = None,
        step: float | None = None,
        lower: bool = False,
        return_dec: bool = True,
    ) -> tuple[np.ndarray[np.floating], np.ndarray[np.floating]]: ...
    def rise_set_times(
        self,
        source: _SkySourceLike,
        t0: ctime.TimeLike,
        t1: ctime.TimeLike | None = None,
        step: float | None = None,
        diameter: float = 0.0,
    ) -> tuple[np.ndarray[np.floating], np.ndarray[bool]]: ...
    def rise_times(
        self,
        source: _SkySourceLike,
        t0: ctime.TimeLike,
        t1: ctime.TimeLike | None = None,
        step: float | None = None,
        diameter: float = 0.0,
    ) -> np.ndarray[np.floating]: ...
    def set_times(
        self,
        source: _SkySourceLike,
        t0: ctime.TimeLike,
        t1: ctime.TimeLike | None = None,
        step: float | None = None,
        diameter: float = 0.0,
    ) -> np.ndarray[np.floating]: ...
    @overload
    def solar_transit(
        self,
        t0: ctime.TimeLike,
        t1: ctime.TimeLike | None = None,
        step: float | None = None,
        lower: bool = False,
        return_dec: bool = False,
    ) -> np.ndarray[np.floating]: ...
    @overload
    def solar_transit(
        self,
        t0: ctime.TimeLike,
        t1: ctime.TimeLike | None = None,
        step: float | None = None,
        lower: bool = False,
        return_dec: bool = True,
    ) -> tuple[np.ndarray[np.floating], np.ndarray[np.floating]]: ...
    @overload
    def lunar_transit(
        self,
        t0: ctime.TimeLike,
        t1: ctime.TimeLike | None = None,
        step: float | None = None,
        lower: bool = False,
        return_dec: bool = False,
    ) -> np.ndarray[np.floating]: ...
    @overload
    def lunar_transit(
        self,
        t0: ctime.TimeLike,
        t1: ctime.TimeLike | None = None,
        step: float | None = None,
        lower: bool = False,
        return_dec: bool = True,
    ) -> tuple[np.ndarray[np.floating], np.ndarray[np.floating]]: ...
    def solar_setting(
        self,
        t0: ctime.TimeLike,
        t1: ctime.TimeLike | None = None,
        step: float | None = None,
    ) -> np.ndarray[np.floating]: ...
    def lunar_setting(
        self,
        t0: ctime.TimeLike,
        t1: ctime.TimeLike | None = None,
        step: float | None = None,
    ) -> np.ndarray[np.floating]: ...
    def solar_rising(
        self,
        t0: ctime.TimeLike,
        t1: ctime.TimeLike | None = None,
        step: float | None = None,
    ) -> np.ndarray[np.floating]: ...
    def lunar_rising(
        self,
        t0: ctime.TimeLike,
        t1: ctime.TimeLike | None = None,
        step: float | None = None,
    ) -> np.ndarray[np.floating]: ...
    def cirs_radec(self, source: Star) -> Star: ...
    def star_cirs(self, ra: Angle, dec: Angle, epoch: Time) -> Star: ...
    def object_coords(
        self, source: _SkySourceLike, date: float | None = None, deg: bool = False
    ) -> tuple[float, float]: ...
    def _sr_work(
        self,
        source: _SkySourceLike,
        t0: float,
        t1: float | None = None,
        step: float | None = None,
        diameter: float = 100.0 / 60,
        skip_rise: bool = False,
        skip_set: bool = False,
    ): ...
    def _source_ha(self, source: _SkySourceLike, t: float) -> float: ...
    def _source_alt(self, source: _SkySourceLike, t: float) -> float: ...

@overload
def _fixup_interval_and_step(
    t0: ctime.TimeLike, t1: ctime.TimeLike, step: int | float
) -> tuple[tuple[ctime.TimeLike, ctime.TimeLike], float | int]: ...
@overload
def _fixup_interval_and_step(
    t0: npt.ArrayLike[ctime.TimeLike],
    t1: npt.ArrayLike[ctime.TimeLike],
    step: int | float,
) -> tuple[
    tuple[np.ndarray[ctime.TimeLike], np.ndarray[ctime.TimeLike]], float | int
]: ...
def _solve_all(
    f: Callable,
    x0: float,
    x1: float,
    dx: float,
    skip_increasing: bool = False,
    skip_decreasing: bool = False,
    **kwargs: dict,
) -> tuple[np.ndarray[np.float64], np.ndarray[bool]]: ...
