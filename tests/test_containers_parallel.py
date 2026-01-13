"""Unit tests for the containers module."""

from caput import containers


def test_redistribute():
    """Test redistribute in the base :py:class:`~caput.containers.Container`."""

    g = containers.Container(distributed=True)

    
    # Create an array from data
    g.create_dataset("data", shape=(10, 10), distributed=True, distributed_axis=0)
    assert g["data"].distributed_axis == 0
    g.redistribute(1)
    assert g["data"].distributed_axis == 1
