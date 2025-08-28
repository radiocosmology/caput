from typing import overload

import numpy as np
import numpy.typing as npt

__all__ = ["jl", "jl_d", "jl_d2"]

@overload
def jl(l: np.number, z: np.number) -> np.number: ...
@overload
def jl(
    l: npt.NDArray[np.number], z: npt.NDArray[np.number]
) -> np.ndarray[np.number]: ...
@overload
def jl_d(l: np.number, z: np.number) -> np.number: ...
@overload
def jl_d(
    l: npt.NDArray[np.number], z: npt.NDArray[np.number]
) -> np.ndarray[np.number]: ...
@overload
def jl_d2(l: np.number, z: np.number) -> np.number: ...
@overload
def jl_d2(
    l: npt.NDArray[np.number], z: npt.NDArray[np.number]
) -> np.ndarray[np.number]: ...
