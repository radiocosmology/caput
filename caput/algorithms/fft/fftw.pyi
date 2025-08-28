import numpy as np
import numpy.typing as npt
import pyfftw

__all__ = ["FFTW", "fft", "fftconvolve", "fftwindow", "ifft"]

class FFTW:
    _nsimd: int
    _params: dict
    _fft: pyfftw.FFTW
    _ifft: pyfftw.FFTW
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: npt.DTypeLike,
        axes: int | tuple[int, ...] | list[int] | None = None,
        forward: bool = True,
        backward: bool = True,
    ) -> None: ...
    @property
    def params(self) -> dict: ...
    def fft(
        self, x: npt.NDArray[np.complexfloating]
    ) -> np.ndarray[np.complexfloating]: ...
    def ifft(
        self, x: npt.NDArray[np.complexfloating]
    ) -> np.ndarray[np.complexfloating]: ...
    def fftconvolve(
        self, in1: npt.NDArray[np.complexfloating], in2: npt.NDArray[np.complexfloating]
    ) -> np.ndarray[np.complexfloating]: ...
    def fftwindow(
        self,
        x: npt.NDArray[np.complexfloating],
        window: npt.NDArray[np.floating | np.complexfloating],
    ) -> np.ndarray[np.complexfloating]: ...

def fft(
    x: npt.NDArray[np.complexfloating],
    axes: int | tuple[int, ...] | list[int] | None = None,
) -> np.ndarray[np.complexfloating]: ...
def ifft(
    x: npt.NDArray[np.complexfloating],
    axes: int | tuple[int, ...] | list[int] | None = None,
) -> np.ndarray[np.complexfloating]: ...
def fftconvolve(
    in1: npt.NDArray[np.complexfloating],
    in2: npt.NDArray[np.complexfloating],
    axes: int | tuple[int, ...] | list[int] | None = None,
) -> np.ndarray[np.complexfloating]: ...
def fftwindow(
    x: npt.NDArray[np.complexfloating],
    window: npt.NDArray[np.floating | np.complexfloating],
    axes: int | tuple[int, ...] | list[int] | None = None,
) -> np.ndarray[np.complexfloating]: ...
