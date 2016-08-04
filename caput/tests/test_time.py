"""Unit tests for ephemeris module."""

import unittest
import random
import time
import math
from datetime import datetime

import numpy as np
import ephem

from caput import time as ctime


class TestUT2RATransit(unittest.TestCase):

    def test_epoch(self):
        # At the J2000 epoch, sidereal time and transit RA should be the same.
        epoch = datetime(2000, 1, 01, 11, 58, 56)

        # Create an observer at an arbitrary location
        obs = ctime.Observer(118.3, 36.1)

        # Extract the ephem observer and get the Sidereal Time
        e_obs = obs.ephem_obs()
        e_obs.date = epoch
        ST = math.degrees(e_obs.sidereal_time())

        # Calculate the transit_RA
        unix_epoch = ctime.datetime_to_unix(epoch)
        TRA = obs.transit_RA(unix_epoch)

        # Tolerance limited by stellar aberation
        self.assertTrue(np.allclose(ST, TRA, atol=0.01, rtol=1e-10))

    def test_array(self):
        # Do a simple test of transit_RA in an array. Use the fact that the RA
        # advances predictably to predict the answers

        epoch = datetime(2000, 1, 01, 11, 58, 56)

        # Create an observer at an arbitrary location
        obs = ctime.Observer(118.3, 36.1)

        # Extract the ephem observer and get the Sidereal Time
        e_obs = obs.ephem_obs()
        e_obs.date = epoch
        ST = math.degrees(e_obs.sidereal_time())

        # Drift rate should be very close to 1 degree/4minutes.
        # Fetch times calculated by ephem
        delta_deg = np.arange(20)
        delta_deg.shape = (5, 4)
        ST = ST + delta_deg

        # Calculate RA using transit_RA
        unix_epoch = ctime.datetime_to_unix(epoch)
        unix_times = unix_epoch + (delta_deg * 60 * 4 * ctime.SIDEREAL_S)
        TRA = obs.transit_RA(unix_times)

        # Compare
        self.assertTrue(np.allclose(ST, TRA, atol=0.02, rtol=1e-10))

    def test_delta(self):

        delta = np.arange(0, 200000, 1000)  # Seconds.
        # time.time() when I wrote this.  No leap seconds for the next few
        # days.

        obs = ctime.Observer(118.3, 36.1)

        start = 1383679008.816173
        times = start + delta
        start_ra = obs.transit_RA(start)
        ra = obs.transit_RA(times)
        delta_ra = ra - start_ra
        expected = delta / 3600. * 15. / ctime.SIDEREAL_S
        error = ((expected - delta_ra + 180.) % 360) - 180
        # Tolerance limited by stellar aberation (40" peak to peak).
        self.assertTrue(np.allclose(error, 0, atol=0.02))


class TestLST(unittest.TestCase):

    def test_lst_pyephem(self):
        # Check an LST calculated by caput.time against one calculated by PyEphem

        dt = datetime(2014, 10, 2, 13, 4, 5)

        t1 = ctime.datetime_to_unix(dt)
        obs = ctime.Observer(42.8, 4.7)
        lst1 = obs.unix_to_lst(t1)

        e_obs = ephem.Observer()
        e_obs.lat = np.radians(4.7)
        e_obs.long = np.radians(42.8)
        e_obs.date = ctime.unix_to_ephem_time(ctime.datetime_to_unix(dt))
        lst2 = np.degrees(e_obs.sidereal_time())

        self.assertAlmostEqual(lst1, lst2, 4)

    def test_lst_known(self):
        # Check an LST calculated by caput.time against one near the Epoch.
        # Correacted to GAST by
        # http://dc.zah.uni-heidelberg.de/apfs/times/q/form

        dt = datetime(2000, 1, 1, 12, 0, 0)

        t1 = ctime.datetime_to_unix(dt)
        obs = ctime.Observer(0.0, 0.0)
        lst1 = obs.unix_to_lst(t1)

        lst2 = (18.0 + 41 / 60.0 + 49.6974 / 3600.0) / 24.0 * 360.0

        self.assertAlmostEqual(lst1, lst2, 4)

    def test_reverse_lst(self):
        # Check that the lst_to_unix routine correctly inverts a call to
        # unix_to_lst

        dt1 = datetime(2018, 3, 12, 1, 2, 3)
        t1 = ctime.datetime_to_unix(dt1)

        dt0 = datetime(2018, 3, 12)
        t0 = ctime.datetime_to_unix(dt0)

        obs = ctime.Observer(42.8, 4.7)
        lst = obs.unix_to_lst(t1)

        t2 = obs.lst_to_unix(lst, t0)

        self.assertAlmostEqual(t1, t2, 2)

    def test_array(self):

        dt = datetime(2000, 1, 1, 12, 0, 0)

        t1 = ctime.datetime_to_unix(dt)

        obs = ctime.Observer(0.0, 0.0)
        times = t1 + np.linspace(0, 24 * 3600.0, 25)
        lsts = obs.unix_to_lst(times)

        # Check that the vectorization works correctly
        for t, lst in zip(times, lsts):
            self.assertEqual(lst, obs.unix_to_lst(t))

        # Check the inverse is correct. The first 24 entries should be correct,
        # but the last one should be one sidereal day behind (because times[0]
        # was not in the correct sidereal day)
        itimes = obs.lst_to_unix(lsts, times[0])
        self.assertTrue(np.allclose(times[:-1], itimes[:-1], rtol=1.e-5, atol=1.e-5))
        self.assertAlmostEqual(times[-1] - itimes[-1], 24 * 3600.0 * ctime.SIDEREAL_S, 1)


class TestLSD(unittest.TestCase):

    def test_lsd(self):

        """Test CHIME sidereal day definition."""

        obs = ctime.Observer(113.2, 62.4)
        obs.lsd_start_day = ctime.datetime_to_unix(datetime(2014, 1, 2))

        # Check the zero point is correct
        self.assertEqual(obs.lsd_zero(), obs.lst_to_unix(0.0, obs.lsd_start_day))

        dt = datetime(2017, 3, 4, 5, 6, 7)
        ut = ctime.datetime_to_unix(dt)

        # Check that the fractional part if equal to the transit RA
        self.assertAlmostEqual(360.0 * (obs.unix_to_lsd(ut) % 1.0), obs.unix_to_lst(ut), places=4)

        # Check a specific precalculated CSD
        # csd1 = -1.1848262244129479
        # self.assertAlmostEqual(ephemeris.csd(et1), csd1, places=7)

    def test_array(self):

        dt = datetime(2025, 1, 1, 12, 0, 0)

        t1 = ctime.datetime_to_unix(dt)

        obs = ctime.Observer(0.0, 0.0)
        times = t1 + np.linspace(0, 48 * 3600.0, 25)
        lsds = obs.unix_to_lsd(times)

        # Check that the vectorization works correctly
        for t, lsd in zip(times, lsds):
            self.assertEqual(lsd, obs.unix_to_lsd(t))

        # Check the inverse is correct.
        itimes = obs.lsd_to_unix(lsds)
        self.assertTrue(np.allclose(times, itimes, rtol=1.e-5, atol=1.e-5))


class TestTime(unittest.TestCase):

    def test_datetime_to_string(self):
        dt = datetime(2014, 04, 21, 16, 33, 12, 12356)
        fdt = ctime.datetime_to_timestr(dt)
        self.assertEqual(fdt, "20140421T163312Z")

    def test_string_to_datetime(self):
        dt = ctime.timestr_to_datetime("20140421T163312Z_stone")
        ans = datetime(2014, 04, 21, 16, 33, 12)
        self.assertEqual(dt, ans)

    def test_from_unix_time(self):
        """Make sure we are properly parsing the unix time.

        This is as much a test of ephem as our code. See issue #29 on the
        PyEphem github page.
        """

        unix_time = random.random() * 2e6
        dt = datetime.utcfromtimestamp(unix_time)
        et = ctime.unix_to_ephem_time(unix_time)
        new_dt = et.datetime()
        self.assertEqual(dt.year, new_dt.year)
        self.assertEqual(dt.month, new_dt.month)
        self.assertEqual(dt.day, new_dt.day)
        self.assertEqual(dt.hour, new_dt.hour)
        self.assertEqual(dt.minute, new_dt.minute)
        self.assertEqual(dt.second, new_dt.second)
        self.assertTrue(abs(dt.microsecond - new_dt.microsecond) < 5)

    def test_time_precision(self):
        """Make sure we have ~millisecond precision and that we aren't
        overflowing anything at double precision."""

        delta = 0.0001
        unix_time = time.time()
        unix_time2 = unix_time + delta
        dt1 = ctime.unix_to_ephem_time(unix_time).datetime()
        dt2 = ctime.unix_to_ephem_time(unix_time2).datetime()
        err = abs(dt2.microsecond - dt1.microsecond - 1e6 * delta)
        self.assertTrue(err < 10)

    def test_datetime_to_unix(self):

        unix_time = time.time()
        dt = datetime.utcfromtimestamp(unix_time)
        new_unix_time = ctime.datetime_to_unix(dt)
        self.assertAlmostEqual(new_unix_time, unix_time, 5)

    def quarry_leap_second(self):
        # 'test_' removed from name to deactivate the test untill this can be
        # implemented.
        l_second_date = datetime(2009, 01, 01, 00, 00, 00)
        l_second_date = ctime.datetime_to_unix(l_second_date)
        before = l_second_date - 10000
        after = l_second_date + 10000
        after_after = l_second_date + 200
        self.assertTrue(time.leap_second_between(before, after))
        self.assertFalse(time.leap_second_between(after, after_after))


if __name__ == '__main__':
    unittest.main()
