# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import unittest
import random
import time
from datetime import datetime

import numpy as np
from skyfield import earthlib, api
from pytest import approx

from caput import time as ctime


# Download the required Skyfield files from a mirror on a CHIME server.
#
# The upstream servers for the timescale and ephemeris data can be
# flaky. Use this to ensure a copy will be downloaded at the risk of it
# being potentially out of date. This is useful for things like CI
# servers, but otherwise letting Skyfield do it's downloading is a
# better idea.
#
mirror_url = "https://bao.chimenet.ca/skyfield/"

files = ["Leap_Second.dat", "deltat.data", "deltat.preds", "de421.bsp"]

for file in files:
    ctime.skyfield_wrapper.load(mirror_url + file)


def test_epoch():

    # At the J2000 epoch, sidereal time and transit RA should be the same.
    epoch = datetime(2000, 1, 1, 11, 58, 56)

    # Create an observer at an arbitrary location
    obs = ctime.Observer(118.3, 36.1)

    # Calculate the transit_RA
    unix_epoch = ctime.datetime_to_unix(epoch)
    TRA = obs.transit_RA(unix_epoch)

    # Calculate LST
    t = ctime.unix_to_skyfield_time(unix_epoch)
    gst = earthlib.sidereal_time(t)
    lst = (360.0 * gst / 24.0 + obs.longitude) % 360.0

    # Tolerance limited by stellar aberation
    assert lst == approx(TRA, abs=0.01, rel=1e-10)


def test_transit_array():
    # Do a simple test of transit_RA in an array. Use the fact that the RA
    # advances predictably to predict the answers

    epoch = datetime(2000, 1, 1, 11, 58, 56)

    # Create an observer at an arbitrary location
    obs = ctime.Observer(118.3, 36.1)

    # Calculate LST
    t = ctime.unix_to_skyfield_time(ctime.datetime_to_unix(epoch))
    gst = earthlib.sidereal_time(t)
    lst = (360.0 * gst / 24.0 + obs.longitude) % 360.0

    # Drift rate should be very close to 1 degree/4minutes.
    # Fetch times calculated by ephem
    delta_deg = np.arange(20)
    delta_deg.shape = (5, 4)
    lst = lst + delta_deg

    # Calculate RA using transit_RA
    unix_epoch = ctime.datetime_to_unix(epoch)
    unix_times = unix_epoch + (delta_deg * 60 * 4 * ctime.SIDEREAL_S)
    TRA = obs.transit_RA(unix_times)

    # Compare
    assert lst == approx(TRA, abs=0.02, rel=1e-10)


def test_delta():

    delta = np.arange(0, 200000, 1000)  # Seconds.
    # time.time() when I wrote this.  No leap seconds for the next few
    # days.

    obs = ctime.Observer(118.3, 36.1)

    start = 1383679008.816173
    times = start + delta
    start_ra = obs.transit_RA(start)
    ra = obs.transit_RA(times)
    delta_ra = ra - start_ra
    expected = delta / 3600.0 * 15.0 / ctime.SIDEREAL_S
    error = ((expected - delta_ra + 180.0) % 360) - 180
    # Tolerance limited by stellar aberation (40" peak to peak).
    assert error == approx(0, abs=0.02)


def test_lsa_skyfield():
    # Check an lsa calculated by caput.time against one calculated by PyEphem

    dt = datetime(2014, 10, 2, 13, 4, 5)
    dt_utc = dt.replace(tzinfo=api.utc)

    t1 = ctime.datetime_to_unix(dt)
    obs = ctime.Observer(42.8, 4.7)
    lsa1 = obs.unix_to_lsa(t1)

    t = ctime.skyfield_wrapper.timescale.utc(dt_utc)
    lsa2 = (earthlib.earth_rotation_angle(t.ut1) * 360.0 + obs.longitude) % 360.0

    assert lsa1 == approx(lsa2, abs=1e-4)


def test_lsa_tra():
    # Near the epoch transit RA and LRA should be extremely close

    dt = datetime(2001, 2, 3, 4, 5, 6)

    t1 = ctime.datetime_to_unix(dt)
    obs = ctime.Observer(118.0, 31.0)
    lsa = obs.unix_to_lsa(t1)
    tra = obs.transit_RA(t1)

    assert lsa == approx(tra, abs=1e-5)


def test_reverse_lsa():
    # Check that the lsa_to_unix routine correctly inverts a call to
    # unix_to_lsa

    dt1 = datetime(2018, 3, 12, 1, 2, 3)
    t1 = ctime.datetime_to_unix(dt1)

    dt0 = datetime(2018, 3, 12)
    t0 = ctime.datetime_to_unix(dt0)

    obs = ctime.Observer(42.8, 4.7)
    lsa = obs.unix_to_lsa(t1)

    t2 = obs.lsa_to_unix(lsa, t0)

    assert t1 == approx(t2, abs=1e-2)


def test_lsa_array():

    dt = datetime(2000, 1, 1, 12, 0, 0)

    t1 = ctime.datetime_to_unix(dt)

    obs = ctime.Observer(0.0, 0.0)
    times = t1 + np.linspace(0, 24 * 3600.0, 25)
    lsas = obs.unix_to_lsa(times)

    # Check that the vectorization works correctly
    for t, lsa in zip(times, lsas):
        assert lsa == obs.unix_to_lsa(t)

    # Check the inverse is correct. The first 24 entries should be correct,
    # but the last one should be one sidereal day behind (because times[0]
    # was not in the correct sidereal day)
    itimes = obs.lsa_to_unix(lsas, times[0])
    assert times[:-1] == approx(itimes[:-1], rel=1e-5, abs=1e-5)
    assert (times[-1] - itimes[-1]) == approx(24 * 3600.0 * ctime.SIDEREAL_S, abs=0.1)

    # Check that it works with zero length arrays
    assert obs.lsa_to_unix(np.array([]), np.array([])).size == 0


def test_lsd():

    """Test Local Earth Rotation Day (LSD) definition."""

    obs = ctime.Observer(113.2, 62.4)
    obs.lsd_start_day = ctime.datetime_to_unix(datetime(2014, 1, 2))

    # Check the zero point is correct
    assert obs.lsd_zero() == obs.lsa_to_unix(0.0, obs.lsd_start_day)

    dt = datetime(2017, 3, 4, 5, 6, 7)
    ut = ctime.datetime_to_unix(dt)

    # Check that the fractional part if equal to the transit RA

    assert 360.0 * (obs.unix_to_lsd(ut) % 1.0) == approx(obs.unix_to_lsa(ut), abs=1e-4)

    # Check a specific precalculated CSD
    # csd1 = -1.1848262244129479
    # self.assertAlmostEqual(ephemeris.csd(et1), csd1, places=7)


def test_lsd_array():

    dt = datetime(2025, 1, 1, 12, 0, 0)

    t1 = ctime.datetime_to_unix(dt)

    obs = ctime.Observer(0.0, 0.0)
    times = t1 + np.linspace(0, 48 * 3600.0, 25)
    lsds = obs.unix_to_lsd(times)

    # Check that the vectorization works correctly
    for t, lsd in zip(times, lsds):
        assert lsd == obs.unix_to_lsd(t)

    # Check the inverse is correct.
    itimes = obs.lsd_to_unix(lsds)
    assert times == approx(itimes, rel=1e-5, abs=1e-5)

    # Check that it works with zero length arrays
    assert obs.lsd_to_unix(np.array([])).size == 0

    # Check that is works with lists (this was previously a bug)
    assert obs.lsd_zero() == approx(obs.lsd_to_unix([0.0, 0.0]), abs=1e-3, rel=0)


def test_era_accuracy():

    # Pick a time to check the ERA around
    dts = ctime.ensure_unix(datetime(2000, 1, 1))

    # These should give back the same time, but the accuracy of the STELLAR_S constant
    # and Skyfields interpolation of dUT1 limit this.
    t0 = ctime.era_to_unix(0, dts)
    t1 = ctime.era_to_unix(0, dts - 5 * 3600)

    # The accuracy should be better than a millisecond
    assert t0 == approx(t1, abs=1e-3)


def test_datetime_to_string():
    dt = datetime(2014, 4, 21, 16, 33, 12, 12356)
    fdt = ctime.datetime_to_timestr(dt)
    assert fdt == "20140421T163312Z"


def test_string_to_datetime():
    dt = ctime.timestr_to_datetime("20140421T163312Z_stone")
    ans = datetime(2014, 4, 21, 16, 33, 12)
    assert dt == ans


def test_from_unix_time():
    """Make sure we are properly parsing the unix time.

    This is as much a test of Skyfield as our code.
    """

    unix_time = random.random() * 2e6
    dt = datetime.utcfromtimestamp(unix_time)
    st = ctime.unix_to_skyfield_time(unix_time)
    new_dt = st.utc_datetime()
    assert dt.year == new_dt.year
    assert dt.month == new_dt.month
    assert dt.day == new_dt.day
    assert dt.hour == new_dt.hour
    assert dt.minute == new_dt.minute
    assert dt.second == new_dt.second

    # Skyfield rounds its output at the millisecond level.
    assert dt.microsecond == approx(new_dt.microsecond, abs=1000)


def test_time_precision():
    """Make sure we have ~0.03 ms precision and that we aren't overflowing
    anything at double precision. This number comes from the precision on
    Julian date time representations:
    http://aa.usno.navy.mil/software/novas/USNOAA-TN2011-02.pdf
    """

    delta = 0.001  # Try a 1 ms shift
    unix_time = time.time()
    unix_time2 = unix_time + delta
    tt1 = ctime.unix_to_skyfield_time(unix_time).tt_calendar()
    tt2 = ctime.unix_to_skyfield_time(unix_time2).tt_calendar()
    err = abs(tt2[-1] - tt1[-1] - delta)

    assert err < 4e-5  # Check that it is accurate at the 0.03 ms level.


def test_datetime_to_unix():

    unix_time = time.time()
    dt = datetime.utcfromtimestamp(unix_time)
    new_unix_time = ctime.datetime_to_unix(dt)
    assert new_unix_time == approx(unix_time, abs=1e-5)


def test_leap_seconds():
    # 'test_' removed from name to deactivate the test untill this can be
    # implemented.
    l_second_date = datetime(2009, 1, 1, 0, 0, 0)
    l_second_date = ctime.datetime_to_unix(l_second_date)
    before = l_second_date - 10000
    after = l_second_date + 10000
    after_after = l_second_date + 200
    assert ctime.leap_seconds_between(before, after) == 1
    assert ctime.leap_seconds_between(after, after_after) == 0

    # Check that a period including an extra leap seconds has two increments
    l_second2_date = ctime.datetime_to_unix(datetime(2012, 7, 1, 0, 0, 0))
    after2 = l_second2_date + 10000

    assert ctime.leap_seconds_between(before, after2) == 2


def test_era_known():
    # Check an ERA calculated by caput.time against one calculated by
    # http://dc.zah.uni-heidelberg.de/apfs/times/q/form (note the latter
    # uses UT1, so we have maximum precision of 1s)

    dt = datetime(2016, 4, 3, 2, 1, 0)

    t1 = ctime.datetime_to_unix(dt)
    era1 = ctime.unix_to_era(t1)
    era2 = 221.0 + (52.0 + 50.828 / 60.0) / 60.0

    assert era1 == approx(era2, abs=1e-3)

    # Test another one
    dt = datetime(2001, 2, 3, 4, 5, 6)

    t1 = ctime.datetime_to_unix(dt)
    era1 = ctime.unix_to_era(t1)
    era2 = 194.0 + (40.0 + 11.549 / 60.0) / 60.0

    assert era1 == approx(era2, abs=1e-3)


def test_era_inverse():

    # Check a full forward/inverse cycle
    dt = datetime(2016, 4, 3, 2, 1, 0)
    t1 = ctime.datetime_to_unix(dt)
    era = ctime.unix_to_era(t1)
    t2 = ctime.era_to_unix(era, t1 - 3600.0)

    # Should be accurate at the 1 ms level
    assert t1 == approx(t2, abs=1e-3)

    # Check a full forward/inverse cycle over a leap second boundary
    dt = datetime(2009, 1, 1, 3, 0, 0)
    t1 = ctime.datetime_to_unix(dt)
    era = ctime.unix_to_era(t1)
    t2 = ctime.era_to_unix(era, t1 - 6 * 3600.0)

    # Should be accurate at the 10 ms level
    assert t1 == approx(t2, abs=1e-2)


def test_ensure_unix():
    # Check that ensure_unix is doing its job for both scalar and array
    # inputs

    dt = datetime(2016, 4, 3, 2, 1, 0)
    dt_list = [datetime(2016, 4, 3, 2, 1, 0), datetime(2016, 4, 3, 2, 1, 0)]

    ut = ctime.datetime_to_unix(dt)
    ut_array = ctime.datetime_to_unix(dt_list)

    sf = ctime.unix_to_skyfield_time(ut)
    sf_array = ctime.unix_to_skyfield_time(ut_array)

    assert ctime.ensure_unix(dt) == ut
    assert ctime.ensure_unix(ut) == ut

    assert ctime.ensure_unix(sf) == approx(ut, abs=1e-3)

    assert (ctime.ensure_unix(dt_list) == ut_array).all()
    assert (ctime.ensure_unix(ut_array) == ut_array).all()
    assert ctime.ensure_unix(sf_array) == approx(ut_array, rel=1e-10, abs=1e-4)

    # Check that it works for zero length arrays
    assert ctime.ensure_unix(np.array([])).size == 0


