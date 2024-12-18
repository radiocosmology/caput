import numpy as np
import pytest

from caput import truncate


def test_bit_truncate():
    assert truncate.bit_truncate_int(129, 1) == 128
    assert truncate.bit_truncate_int(-129, 1) == -128
    assert truncate.bit_truncate_int(1, 1) == 0

    assert truncate.bit_truncate_long(129, 1) == 128
    assert truncate.bit_truncate_long(-129, 1) == -128
    assert truncate.bit_truncate_long(576460752303423489, 1) == 576460752303423488
    assert (
        truncate.bit_truncate_long(4520628863461491, 140737488355328)
        == 4503599627370496
    )
    assert truncate.bit_truncate_long(1, 1) == 0

    assert truncate.bit_truncate_int(54321, 0) == 54321

    assert truncate.bit_truncate_long(576460752303423489, 0) == 576460752303423489

    # special cases
    assert truncate.bit_truncate_int(129, 0) == 129
    assert truncate.bit_truncate_int(0, 1) == 0
    assert truncate.bit_truncate_int(129, -1) == 0
    assert truncate.bit_truncate_long(129, 0) == 129
    assert truncate.bit_truncate_long(0, 1) == 0
    assert truncate.bit_truncate_long(129, -1) == 0


def test_truncate_float():
    assert truncate.bit_truncate_float(32.121, 1) == 32
    assert truncate.bit_truncate_float(-32.121, 1) == -32
    assert truncate.bit_truncate_float(32.125, 0) == 32.125
    assert truncate.bit_truncate_float(1, 1) == 0

    assert truncate.bit_truncate_float(1 + 1 / 1024, 1 / 2048) == 1 + 1 / 1024
    assert (
        truncate.bit_truncate_float(1 + 1 / 1024 + 1 / 2048, 1 / 2048) == 1 + 2 / 1024
    )
    assert truncate.bit_truncate_double(1 + 1 / 1024, 1 / 2048) == 1 + 1 / 1024
    assert (
        truncate.bit_truncate_double(1 + 1 / 1024 + 1 / 2048, 1 / 2048) == 1 + 2 / 1024
    )

    assert truncate.bit_truncate_double(32.121, 1) == 32
    assert truncate.bit_truncate_double(-32.121, 1) == -32
    assert truncate.bit_truncate_double(32.121, 0) == 32.121
    assert truncate.bit_truncate_double(0.9191919191, 0.001) == 0.919921875
    assert truncate.bit_truncate_double(0.9191919191, 0) == 0.9191919191
    assert truncate.bit_truncate_double(0.010101, 0) == 0.010101
    assert truncate.bit_truncate_double(1, 1) == 0

    # special cases
    assert truncate.bit_truncate_float(32.121, -1) == 0
    assert truncate.bit_truncate_double(32.121, -1) == 0
    assert truncate.bit_truncate_float(32.121, np.inf) == 0
    assert truncate.bit_truncate_double(32.121, np.inf) == 0
    assert truncate.bit_truncate_float(np.inf, 1) == np.inf
    assert truncate.bit_truncate_double(np.inf, 1) == np.inf
    assert np.isnan(truncate.bit_truncate_float(np.nan, 1))
    assert np.isnan(truncate.bit_truncate_double(np.nan, 1))

    assert truncate.bit_truncate_float(np.inf, np.inf) == 0
    assert truncate.bit_truncate_double(np.inf, np.inf) == 0
    assert truncate.bit_truncate_float(np.nan, np.inf) == 0
    assert truncate.bit_truncate_double(np.nan, np.inf) == 0

    # Test that an error is raised when `err` is `NaN`
    with pytest.raises(ValueError):
        truncate.bit_truncate_float(32.121, np.nan)

    with pytest.raises(ValueError):
        truncate.bit_truncate_double(32.121, np.nan)


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


def test_truncate_relative():
    assert (
        truncate.bit_truncate_relative(
            np.asarray([32.121, 32.5], dtype=np.float32),
            0.1,
        )
        == np.asarray([32, 32], dtype=np.float32)
    ).all()
    assert (
        truncate.bit_truncate_relative(
            np.asarray([32.121, 32.5], dtype=np.float64),
            0.1,
        )
        == np.asarray([32, 32], dtype=np.float64)
    ).all()

    # Check the case where values are negative
    assert (
        truncate.bit_truncate_relative(
            np.asarray([-32.121, 32.5], dtype=np.float32),
            0.1,
        )
        == np.asarray([-32, 32], dtype=np.float32)
    ).all()
    assert (
        truncate.bit_truncate_relative(
            np.asarray([-32.121, 32.5], dtype=np.float64),
            0.1,
        )
        == np.asarray([-32, 32], dtype=np.float64)
    ).all()
