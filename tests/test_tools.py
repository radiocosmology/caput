"""Unit tests for the tools module."""

from mpi4py import MPI
import numpy as np
import pytest

from caput import mpiarray, tools


ARRAY_SIZE = (100, 111)
SEED = 12345
ATOL = 0.0
rng = np.random.Generator(np.random.SFC64(SEED))

random_float_array = rng.standard_normal(size=ARRAY_SIZE, dtype=np.float32)
random_double_array = rng.standard_normal(size=ARRAY_SIZE, dtype=np.float64)
random_complex_array = rng.standard_normal(
    size=ARRAY_SIZE
) + 1.0j * rng.standard_normal(size=ARRAY_SIZE)


@pytest.mark.parametrize(
    "a", [random_complex_array, random_float_array, random_double_array]
)
def test_invert_no_zero(a):
    zero_ind = ((0, 10, 12), (56, 34, 78))
    good_ind = np.ones(a.shape, dtype=bool)
    good_ind[zero_ind] = False

    # set up some invalid values for inverse
    a[zero_ind[0][0], zero_ind[1][0]] = 0.0
    a[zero_ind[0][1], zero_ind[1][1]] = 0.5 / np.finfo(a.real.dtype).max

    if np.iscomplexobj(a):
        # these should be inverted fine
        a[10, 0] = 1.0
        a[10, 1] = 1.0j
        # also test invalid in the imaginary part
        a[zero_ind[0][2], zero_ind[1][2]] = 0.5j / np.finfo(a.real.dtype).max
    else:
        a[zero_ind[0][2], zero_ind[1][2]] = -0.5 / np.finfo(a.real.dtype).max

    b = tools.invert_no_zero(a)
    assert np.allclose(b[good_ind], 1.0 / a[good_ind], atol=ATOL)
    assert (b[zero_ind] == 0).all()


def test_invert_no_zero_mpiarray():
    comm = MPI.COMM_WORLD
    comm.Barrier()

    a = mpiarray.MPIArray((20, 30), axis=0, comm=comm)
    a[:] = comm.rank

    b = tools.invert_no_zero(a)

    assert b.shape == a.shape
    assert b.comm == a.comm
    assert b.axis == a.axis
    assert b.local_shape == a.local_shape

    assert (a * b).local_array == pytest.approx(0.0 if comm.rank == 0 else 1.0)
    comm.Barrier()


def test_invert_no_zero_noncontiguous():
    a = np.arange(100, dtype=np.float64).reshape(10, 10)

    res = np.ones((10, 10), dtype=np.float64)
    res[0, 0] = 0.0

    # Check the contiguous layout is working
    b_cont = tools.invert_no_zero(a.T.copy())
    assert a.T * b_cont == pytest.approx(res)

    # Check that a Fortran contiguous array works
    b_noncont = tools.invert_no_zero(a.T)
    assert a.T * b_noncont == pytest.approx(res)

    # Check a complex sub slicing that is neither C nor F contiguous
    a_noncont = a.T[1::2, 1::2]
    b_noncont = tools.invert_no_zero(a_noncont)
    res_cont = tools.invert_no_zero(a_noncont.copy(order="C"))
    assert np.all(b_noncont == res_cont)
    assert b_noncont.flags["C_CONTIGUOUS"]


def test_allequal():
    # Test some basic types
    assert tools.allequal(1, 1)
    assert tools.allequal("x", "x")
    assert tools.allequal([1, 2, 3], [1, 2, 3])
    assert tools.allequal({"a": 1}, {"a": 1})
    assert tools.allequal({1, 2, 3}, {1, 2, 3})
    assert tools.allequal((1, 2, 3), (1, 2, 3))

    # Test numpy arrays and mpiarrays
    assert tools.allequal(np.array([1, 2, 3]), np.array([1, 2, 3]))
    assert tools.allequal(
        mpiarray.MPIArray.wrap(np.array([1, 2, 3]), axis=0),
        mpiarray.MPIArray.wrap(np.array([1, 2, 3]), axis=0),
    )
    assert not tools.allequal(
        np.array([1, 2, 3]),
        mpiarray.MPIArray.wrap(np.array([1, 2, 3]), axis=0),
    )

    # Test objects with numpy arrays in them
    assert tools.allequal(
        [np.array([1]), np.array([2])], [np.array([1]), np.array([2])]
    )
    assert tools.allequal({"a": np.array([1])}, {"a": np.array([1])})

    # Test objects with different types
    assert tools.allequal([1, 3.4, "a"], [1, 3.4, "a"])

    # Test for items that are not equal
    assert not tools.allequal(1, 3)
    assert not tools.allequal(1, 1.1)
    assert not tools.allequal([np.array([1])], [np.array([2])])
    assert not tools.allequal(
        np.array(["x"], dtype="U32"), np.array(["x"], dtype="S32")
    )

    # Test for lengths that are not equal
    assert not tools.allequal([1], [1, 2])
    assert not tools.allequal({"a": 1}, {"a": 1, "b": 2})
