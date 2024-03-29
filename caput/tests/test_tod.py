"""Unit tests for the tod module."""

import unittest

import numpy as np

from caput import tod


class TestConcatenation(unittest.TestCase):
    def setUp(self):
        chan = np.arange(15)
        ntime = [15, 67, 35, 23, 91]
        delta_t = 12.0
        todlist = []
        self.nchan = len(chan)
        self.ntime = np.sum(ntime)

        this_t_start = 262.0
        for nt in ntime:
            this_time = this_t_start + np.arange(nt) * delta_t
            this_t_start += nt * delta_t

            data = tod.TOData()
            data.create_index_map("time", this_time)
            data.create_index_map("chan", chan)

            ds1 = data.create_dataset("dset1", data=chan[:, None] * this_time)
            ds1.attrs["axis"] = ["chan", "time"]

            ds2 = data.create_dataset("dset2", data=this_time)
            ds2.attrs["axis"] = ["time"]

            todlist.append(data)
        self.todlist = todlist

    def test_from_files(self):
        data = tod.TOData.from_mult_files(self.todlist)
        self.assertTrue(
            np.all(
                data["dset1"][:]
                == data.index_map["chan"][:, None] * data.index_map["time"]
            )
        )
        self.assertEqual(data["dset1"].shape, (self.nchan, self.ntime))

    def test_reader(self):
        reader = tod.Reader(self.todlist)
        reader.time_sel = (7, 87)
        reader.dataset_sel = ("dset1",)

        data = reader.read()

        self.assertTrue(
            np.all(
                data["dset1"][:]
                == data.index_map["chan"][:, None] * data.index_map["time"]
            )
        )
        self.assertEqual(data["dset1"].shape, (self.nchan, 80))
        self.assertEqual(list(data.keys()), ["dset1"])

    def test_reader_time_range(self):
        reader = tod.Reader(self.todlist)
        reader.select_time_range(start_time=300.0, stop_time=500.0)

        data = reader.read()

        self.assertTrue(
            np.all(
                data["dset1"][:]
                == data.index_map["chan"][:, None] * data.index_map["time"]
            )
        )
        self.assertTrue(np.all(data["dset2"][:] == data.index_map["time"]))

    def test_index_attr(self):
        for d in self.todlist:
            d.index_attrs["time"]["alignment"] = 1

        data = tod.TOData.from_mult_files(self.todlist)

        # Check that the alignment has been copied properly
        self.assertTrue(data.index_attrs["time"]["alignment"] == 1)

        # Check that the shift has been applied properly
        self.assertTrue(
            np.allclose(
                data.index_map["time"][:],
                data.time[:] - (abs(np.median(np.diff(data.time[:]))) / 2),
            )
        )


if __name__ == "__main__":
    unittest.main()
