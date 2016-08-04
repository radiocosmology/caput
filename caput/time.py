"""
======================================
Time calculations (:mod:`~caput.time`)
======================================

A set of useful routines for calculation of solar and sidereal times.

Constants
=========

:const:`SIDEREAL_S`
    Number of SI seconds in a sidereal second [s/sidereal s].

Time Utilities
==============

.. autosummary::
    :toctree: generated/

    unix_to_ephem_time
    unix_to_datetime
    ensure_unix
    datetime_to_unix
    datetime_to_timestr
    timestr_to_datetime
    leap_second_between
    time_of_day


Issues
======

* :func:`unix_to_ephem_time` Unix time is based on UTC and :mod:`ephem` time is
  based on UT1.  The difference is ignored.  See `pyephem` github issue #30.


Notes
=====

The distinctions between some of the quantities are a little subtle. Here are a
few notes.

First, Local Sidereal Time and the transiting RA are superificially similar, but
differ in the fact that the latter is fixed to a specific epoch. If the
co-ordinate epoch is the time we are calculating they coincide, otherwise they
differ because of the precession of the Earth's rotation axis. The shift in RA
at the equator is about 0.3 arc minutes per year, but after removing this there
is still a declination dependent variation of a similar magnitude.

Local Mean Sidereal Time and Local Apparent Sidereal Time differ by the equation
of the equinoxes. This is the nutation of the Earth's axis throughout the year.,
which is included in the Apparent Sidereal Time. This effect is at most 1s in
time, and so 0.25 arc minutes in angle. Here we use Local Apparent Sidereal Time
throughout.

UT1 and UTC are similar but fundamentally different time standards. UT1 is an
astronomical time standard based on the Earth's rotation. UTC is an atomic time
standard kept within 1s of UT1 by the addition (or subtraction) of leap seconds.
PyEphem uses UT1 internally (though is confused about this), where as we use
UTC. We do not correct for this which gives a maximum error of 0.25 arc minutes.

A more modern definition of the celestial co-ordinate system uses ICRS/ICRF and
Earth Rotation Angle. This is now all included in AstroPy/SkyField which we should
probably use instead of PyEphem.
"""


import math
from datetime import datetime

import ephem
import numpy as np

from .misc import vectorize


# Approximate number of seconds in a sidereal second.
SIDEREAL_S = 1. / (1. + 1.0 / 365.259636)


class Observer(object):
    """Time calculations for a local observer.

    Parameters
    ----------
    longitude : float
        Longitude of observer in degrees.
    latitude : float
        Latitude of observer in degrees.
    altitude : float, optional
        Altitude of observer in metres.
    lsd_start : float or datetime, optional
        The zeroth LSD. If not set use the J2000 epoch start.

    Attributes
    ----------
    longitude : float
        Longitude of observer in degrees.
    latitude : float
        Latitude of observer in degrees.
    altitude : float
        Altitude of observer in metres.
    epoch : string
        The epoch for RA calculations. Defaults to `'2000'`. Accepts any format
        that `PyEphem` understands.
    lsd_start_day : float
        UNIX time on the zeroth LSD. The actual zero point is the first time of
        LST=0.0 after the `lsd_zero`.

    Methods
    -------
    unix_to_lst
    lst_to_unix
    unix_to_lsd
    lsd_to_unix
    transit_RA
    ephem_obs
    """

    latitude = 0.0
    longitude = 0.0

    altitude = 0.0

    epoch = '2000'

    lsd_start_day = 0.0

    def __init__(self, lon, lat, alt=0.0, lsd_start=None):

        self.longitude = lon
        self.latitude = lat
        self.altitude = alt

        if lsd_start is None:
            lsd_start = datetime(2000, 1, 1, 11, 58, 56)

        self.lsd_start_day = ensure_unix(lsd_start)

    def ephem_obs(self):
        """Create a PyEphem observer object.

        Returns
        -------
        obs : :class:`ephem.Observer`
        """
        obs = ephem.Observer()
        obs.lat = math.radians(self.latitude)
        obs.long = math.radians(self.longitude)
        obs.elevation = self.altitude

        # Could do with some validation here
        obs.epoch = self.epoch

        return obs

    @vectorize()
    def unix_to_lst(self, time):
        """Calculate the Local Apparent Sidereal Time.

        Parameters
        ----------
        time : float
            Unix time.

        Returns
        -------
        lst : float
        """

        obs = self.ephem_obs()
        obs.date = unix_to_ephem_time(time)
        return np.degrees(obs.sidereal_time())

    lst = unix_to_lst

    def lst_to_unix(self, lst, time0):
        """Convert a Local Apparent Sidereal Time on a given
        day to a UNIX time.

        Parameters
        ----------
        lst : scalar or np.ndarray
            Sidereal time in degrees to convert.
        time0 : scalar or np.ndarray
            An earlier time within 24 sidereal hours. For example,
            the start of the solar day of the observation.

        Returns
        -------
        time : scalar or np.ndarray
            Corresponding UNIX time.
        """

        lst0 = self.unix_to_lst(time0)

        diff_lst_sec = ((lst - lst0) % 360.0) * 240.0  # Convert from degrees in seconds (time)

        return time0 + diff_lst_sec * SIDEREAL_S

    def lsd_zero(self):
        """Return the zero point of LSD as a UNIX time.

        Returns
        -------
        lsd_zero : float
        """
        return self.lst_to_unix(0.0, self.lsd_start_day)

    def unix_to_lsd(self, time):
        """Calculate the Local Sidereal Day corresponding to the given time.

        The Local Sidereal Day is the number of sidereal days that have passed
        since 0 deg LST on the solar day 15/11/2013 (in UTC).

        Parameters
        ----------
        time :  float (UNIX time)

        Returns
        -------
        lsd : float
        """

        # Get fractional part from LST
        frac_part = self.unix_to_lst(time) / 360.0

        # Calculate the approximate CSD by crudely dividing the time difference by
        # the length of a sidereal day
        approx_lsd = (time - self.lsd_zero()) / (24.0 * 3600.0 * SIDEREAL_S)

        # Subtract the accurate part, and round to the nearest integer to get the
        # number of whole days elapsed (should be accurate for a very long time)
        whole_days = np.rint(approx_lsd - frac_part)

        return whole_days + frac_part

    lsd = unix_to_lsd

    def lsd_to_unix(self, lsd):
        """Calculate the UNIX time corresponding to CSD.

        The CHIME Sidereal Day is the number of sidereal days that have passed
        since RA=0, DEC=0 transited on the day 15/11/2013 (in UTC).

        Parameters
        ----------
        csd : float

        Returns
        -------
        time :  float (UNIX time)
        """

        # Find the approximate UNIX time
        approx_unix = self.lsd_zero() + lsd * 3600 * 24 * SIDEREAL_S

        # Shift to 12 hours before to give the start of the search period
        start_unix = approx_unix - 12 * 3600

        # Get the LST from the LSD in degrees
        lst = 360.0 * (lsd % 1.0)

        # Solve for the next transit of that RA after start_unix
        return self.lst_to_unix(lst, start_unix)

    @vectorize()
    def transit_RA(self, time):
        """Transiting RA for the observer at given Unix Time.

        Because the RA is defined with repect to the specified epoch (J2000 by
        default), the elevation actually matters here. The elevation of the
        equator is used, which minimizes this effect.

        Parameters
        ----------
        time : float or array of floats
            Time as specified by the Unix/POSIX time.

        Returns
        -------
        transit_RA : float or array of floats
            Transiting RA in degrees.

        Notes
        -----

        It is not clear that this calculation includes nutation and stellar
        aberration.  See the discussion here_. Some testing does seem to
        indicate that these effects are accounted for.

        This calculates the RA in the given epoch which by default is J2000, but
        it might be more appropriate to use an epoch that is closer to the
        observation time. The mismatch in the celestial poles is not
        insignificant (~5 arcmin from J2000 to J2016).

        PyEphem uses all geocentric latitudes, which I don't think affects
        this calculation.

        .. _here: http://stackoverflow.com/questions/11970713
        """

        # Initialize ephem location object.
        obs = self.ephem_obs()

        # Want the RA at the equator, which is much less affected by the celestial
        # pole mismatch between now and J2000 epoch.
        az = '180'
        el = str(90 - self.latitude)
        obs.pressure = 0
        obs.date = unix_to_ephem_time(time)
        ra, dec = obs.radec_of(az, el)
        ra = math.degrees(ra)
        return ra


def unix_to_ephem_time(unix_time):
    """Formats the Unix time into a time that can be interpreted by ephem.

    Parameters
    ----------
    unix_time : float
        Unix/POSIX time.

    Returns
    -------
    date : :class:`ephem.Date`

    See Also
    --------
    :meth:`datetime.datetime.utcfromtimestamp`
    :func:`datetime_to_unix`

    """

    # `ephem` documentation claims that all times are UTC, but unclear if
    # `ephem` distinguishes between UT1 and UTC. Difference is always less than
    # 10 arcseconds, or 0.3% of a beam width.  See pyephem github issue #30.

    dt = unix_to_datetime(unix_time)

    # Be careful of a bug in old versions of ephem. See issue #29 on Pyephem
    # github page.
    date = ephem.Date(dt)
    return date


def unix_to_datetime(unix_time):
    """Converts unix time to a :class:`datetime.datetime` object.

    Equivalent to :meth:`datetime.datetime.utcfromtimestamp`.

    Parameters
    ----------
    unix_time : float
        Unix/POSIX time.

    Returns
    --------
    dt : :class:`datetime.datetime`

    See Also
    --------
    :func:`datetime_to_unix`

    """

    return datetime.utcfromtimestamp(unix_time)


def datetime_to_unix(dt):
    """Converts a :class:`datetime.datetime` object to the unix time.

    This is the inverse of :meth:`datetime.datetime.utcfromtimestamp`.

    Parameters
    ----------
    dt : :class:`datetime.datetime`

    Returns
    -------
    unix_time : float
        Unix/POSIX time.

    See Also
    --------
    :func:`unix_to_datetime`
    :meth:`datetime.datetime.utcfromtimestamp`

    """

    # Noting that this operation is ignorant of leap seconds.
    since_epoch = dt - datetime.utcfromtimestamp(0)
    return since_epoch.total_seconds()


def datetime_to_timestr(dt):
    """Converts a :class:`datetime.datetime` to "YYYYMMDDTHHMMSSZ" format.

    Partial seconds are ignored.

    Parameters
    ----------
    dt : :class:`datetime.datetime`

    Returns
    -------
    time_str : string
        Formated date and time.

    See Also
    --------
    :func:`timestr_to_datetime`

    """

    return dt.strftime("%Y%m%dT%H%M%SZ")


def timestr_to_datetime(time_str):
    """Converts date "YYYYMMDDTHHMMSS*" to a :class:`datetime.datetime`.

    Parameters
    ----------
    time_str : string
        Formated date and time.

    Returns
    -------
    time_str : :class:`datetime.datetime`

    See Also
    --------
    :func:`datetime_to_timestr`

    """

    return datetime.strptime(time_str[:15], "%Y%m%dT%H%M%S")


def leap_second_between(time_a, time_b):
    """Determine whether a leap second occurred between two Unix times.

    Not yet implemented.

    Parameters
    ----------
    time_a : float
        First Unix/POSIX time.
    time_b : float
        Second Unix/POSIX time.

    Returns
    -------
    occurred : bool
        If there was a leap second between *time_a* and *time_b*.

    """

    # This doesn't work because delta_t is
    # dt_a = ephem.delta_t(unix_to_ephem_time(time_a))
    # dt_b = ephem.delta_t(unix_to_ephem_time(time_b))
    # print dt_a, dt_b
    # return dt_a != dt_b

    raise NotImplementedError()


def ensure_unix(time):
    """Try and convert the input time to Unix time."""

    if isinstance(time, datetime):
        return datetime_to_unix(time)
    elif isinstance(time, ephem.Date):
        return datetime_to_unix(time.datetime())
    elif isinstance(time, basestring):
        return datetime_to_unix(timestr_to_datetime(time))
    else:
        return float(time)


@vectorize()
def time_of_day(time):
    """Return the time since the start of the UTC day in seconds.

    Parameters
    ----------
    time_date : float (UNIX time), or datetime
        Find the start time of the day that `time` is in.

    Returns
    -------
    time : float
        Time since start of UTC day in seconds.
    """
    dt = datetime.utcfromtimestamp(ensure_unix(time))
    d = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return (dt - d).total_seconds()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
