import numpy as np

from caput import truncate


def test_bit_truncate():
    assert truncate.bit_truncate_int(129, 1) == 128

    assert truncate.bit_truncate_long(129, 1) == 128
    assert truncate.bit_truncate_long(576460752303423489, 1) == 576460752303423488
    assert (
        truncate.bit_truncate_long(4520628863461491, 140737488355328)
        == 4503599627370496
    )

    assert truncate.bit_truncate_int(54321, 0) == 54321

    assert truncate.bit_truncate_long(576460752303423489, 0) == 576460752303423489


def test_truncate_float():
    assert truncate.bit_truncate_float(32.121, 1) == 32
    # fails assert truncate.bit_truncate_float(float(0.010101), 0) == float(0.010101)

    assert truncate.bit_truncate_double(32.121, 1) == 32
    assert truncate.bit_truncate_double(0.9191919191, 0) == 0.9191919191


def test_truncate_array():
    assert (
        truncate.bit_truncate_relative(
            np.asarray([32.121, 32.5], dtype=np.float32), 1 / 32
        )
        == np.asarray([32, 32], dtype=np.float32)
    ).all()
    assert (
        truncate.bit_truncate_relative_double(
            np.asarray([32.121, 32.5], dtype=np.float64), 1 / 32
        )
        == np.asarray([32, 32], dtype=np.float64)
    ).all()


def test_truncate_weights():
    assert (
        truncate.bit_truncate_weights(
            np.asarray([32.121, 32.5], dtype=np.float32),
            np.asarray([1 / 32, 1 / 32], dtype=np.float32),
            0.001,
        )
        == np.asarray([32, 32], dtype=np.float32)
    ).all()
    assert (
        truncate.bit_truncate_weights(
            np.asarray([32.121, 32.5], dtype=np.float64),
            np.asarray([1 / 32, 1 / 32], dtype=np.float64),
            0.001,
        )
        == np.asarray([32, 32], dtype=np.float64)
    ).all()
