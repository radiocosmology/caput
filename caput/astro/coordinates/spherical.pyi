from typing import overload

import numpy as np
import numpy.typing as npt
from skyfield.units import Angle

from ._spherical import *  # noqa: F403

__all__ = [
    "cart_to_sph",  # noqa: F405
    "cosine_rule",  # noqa: F405
    "great_circle_points",  # noqa: F405
    "ground_to_sph",
    "norm_vec2",  # noqa: F405
    "projected_distance",
    "rotate_ypr",
    "sph_dot",  # noqa: F405
    "sph_to_cart",  # noqa: F405
    "sph_to_ground",
    "sphdist",
    "thetaphi_plane",  # noqa: F405
    "thetaphi_plane_cart",  # noqa: F405
]

@overload
def sph_to_ground(
    ha: np.number, lat: np.number, dec: np.number
) -> tuple[np.number, np.number, np.number]: ...
@overload
def sph_to_ground(
    ha: np.number, lat: npt.NDArray[np.number], dec: np.number
) -> tuple[np.number, np.ndarray[np.number], np.ndarray[np.number]]: ...
@overload
def sph_to_ground(
    ha: npt.NDArray[np.number], lat: npt.NDArray[np.number], dec: npt.NDArray[np.number]
) -> tuple[np.ndarray[np.number], np.ndarray[np.number], np.ndarray[np.number]]: ...
@overload
def sph_to_ground(
    ha: np.number | npt.NDArray[np.number],
    lat: np.number | npt.NDArray[np.number],
    dec: npt.NDArray[np.number],
) -> tuple[np.ndarray[np.number], np.ndarray[np.number], np.ndarray[np.number]]: ...
@overload
def ground_to_sph(
    x: np.number, y: np.number, lat: np.number
) -> tuple[np.number, np.number]: ...
@overload
def ground_to_sph(
    x: np.number,
    y: np.number | npt.NDArray[np.number],
    lat: np.number | npt.NDArray[np.number],
) -> tuple[np.ndarray[np.number], np.ndarray[np.number]]: ...
@overload
def ground_to_sph(
    x: np.number | npt.NDArray[np.number],
    y: np.number,
    lat: np.number | npt.NDArray[np.number],
) -> tuple[np.ndarray[np.number], np.ndarray[np.number]]: ...
@overload
def ground_to_sph(
    x: np.number | npt.NDArray[np.number],
    y: np.number | npt.NDArray[np.number],
    lat: np.number,
) -> tuple[np.ndarray[np.number], np.ndarray[np.number]]: ...
@overload
def ground_to_sph(
    x: npt.NDArray[np.number], y: npt.NDArray[np.number], lat: npt.NDArray[np.number]
) -> tuple[np.ndarray[np.number], np.ndarray[np.number]]: ...
def projected_distance(
    ha: np.number | npt.NDArray[np.number],
    lat: np.number | npt.NDArray[np.number],
    dec: np.number | npt.NDArray[np.number],
    x: np.number | npt.NDArray[np.number],
    y: np.number | npt.NDArray[np.number],
    z: np.number | npt.NDArray[np.number] = 0.0,
) -> np.number | np.ndarray[np.number]: ...
def rotate_ypr(
    rot: tuple[np.number, np.number, np.number],
    xhat: npt.ArrayLike,
    yhat: npt.ArrayLike,
    zhat: npt.ArrayLike,
) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]: ...
def sphdist(long1: Angle, lat1: Angle, long2: Angle, lat2: Angle) -> Angle: ...

# Names in __all__ with no definition are provided by _spherical.pyi
#   cart_to_sph
#   cosine_rule
#   great_circle_points
#   norm_vec2
#   sph_dot
#   sph_to_cart
#   thetaphi_plane
#   thetaphi_plane_cart
