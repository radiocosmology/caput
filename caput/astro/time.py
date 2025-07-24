r"""Routines for calculation and of solar and sidereal times.

Time Utilities
==============
Time conversion routine which are location independent.

- :py:meth:`unix_to_skyfield_time`
- :py:meth: `skyfield_time_to_unix`
- :py:meth:`datetime_to_unix`
- :py:meth:`unix_to_datetime`
- :py:meth:`datetime_to_timestr`
- :py:meth:`timestr_to_datetime`
- :py:meth:`unix_to_era`
- :py:meth:`era_to_unix`
- :py:meth:`ensure_unix`
- :py:meth:`leap_seconds_between`
- :py:meth:`time_of_day`
- :py:meth:`naive_datetime_to_utc`

Constants
=========

These are re-exported from `caput.astro.constants`, and should be accessed
from there instead of here.

:const:`SIDEREAL_S`
    Number of SI seconds in a sidereal second [s/sidereal s].
:const:`STELLAR_S`
    Approximate number of SI seconds in a stellar second (i.e. 1/86400 of a stellar day) [s/stellar s].


Why not Sidereal Time?
----------------------

Sidereal time is a long established and well known quantity that is very similar
to the Earth Rotation Angle (and the derived quantities, LSA and LSD). The
reason we don't use that is that Sidereal Time is essentially the Earth's
rotation measured with respect to the vernal equinox, but this quantity moves
significantly with respect to the fixed frame due to precession of the Earth's
pole.

The Earth Rotation Angle is measured with respect to the Celestial Intermediate
Origin (CIO), which essentially moves the minimal amount as the Earth precesses
to remain on the celestial equator. For experiments like CHIME which map the sky
as a function of the Earth's rotation, this gives the minimal coordinate shifts
when combining maps made on different days.

See `USNO Circular 179`_ for more details about the CIO based coordinates.


Accuracy
--------

Conversions between time standards (as opposed to representations of UTC) in
this module are mostly handled by Skyfield_, which has a comprehensive and
modern handling of the various effects. This includes leap seconds, which
``PyEphem`` mostly ignored, just pretending that UTC and UT1 were equivalent.


Skyfield uses Julian Dates for its internal representations. For a double
precision floating point (``np.float64``), the effective precision on
calculations is around 30 us. A technical note discussing this further is
available `here <http://aa.usno.navy.mil/software/novas/USNOAA-TN2011-02.pdf>`_.
Most conversions should be accurate to around this accuracy 0.1 ms. However, the
`era_to_unix` routine (and related routines), are accurate to only around 5 ms
at the moment as they ignore the difference in time between the Sidereal day,
and a complete cycle of ERA.

.. _Skyfield: http://rhodesmill.org/skyfield/

.. _`USNO Circular 179`: http://arxiv.org/abs/astro-ph/0602086

.. _`NFA Glossary`: http://syrte.obspm.fr/iauWGnfa/NFA_Glossary.pdf

.. _`IERS constants`: http://hpiers.obspm.fr/eop-pc/models/constants.html
"""

__all__ = [
    "datetime_to_timestr",
    "datetime_to_unix",
    "ensure_unix",
    "era_to_unix",
    "leap_seconds_between",
    "naive_datetime_to_utc",
    "time_of_day",
    "timestr_to_datetime",
    "unix_to_datetime",
    "unix_to_era",
]

import warnings
from datetime import datetime, timezone

import numpy as np
from skyfield import timelib

from ..lib import array_utils
from . import constants
from .skyfield import unix_to_skyfield_time

# Re-export a few constants for compatibility
UT1_S = constants.UT1_second
SIDEREAL_S = constants.sidereal_second
STELLAR_S = constants.stellar_second


@array_utils.scalarize()
def unix_to_era(unix_time):
    """Calculate the Earth Rotation Angle for a given time.

    The Earth Rotation Angle is the angle between the Celestial and Terrestrial
    Intermediate origins, and is a modern replacement for the Greenwich Sidereal
    Time.

    Parameters
    ----------
    unix_time : float or array of.
        Unix/POSIX time.

    Returns
    -------
    era : float or array of
        The Earth Rotation Angle in degrees.
    """
    from skyfield import earthlib

    t = unix_to_skyfield_time(unix_time)

    era = earthlib.earth_rotation_angle(t.ut1)  # in cycles

    return 360.0 * era


@array_utils.scalarize()
def era_to_unix(era, time0):
    """Calculate the UNIX time for a given Earth Rotation Angle.

    The Earth Rotation Angle is the angle between the Celetial and Terrestrial
    Intermediate origins, and is a modern replacement for the Greenwich Sidereal
    Time.

    This routine is accurate at about the 1 ms level.

    Parameters
    ----------
    era : float or array of
        The Earth Rotation Angle in degrees.
    time0 : scalar or np.ndarray
        An earlier time within 24 sidereal hours. For example,
        the start of the solar day of the observation.

    Returns
    -------
    unix_time : float or array of.
        Unix/POSIX time.
    """
    era0 = unix_to_era(time0)

    diff_era_deg = (era - era0) % 360.0  # Convert from degrees in seconds (time)

    # Convert to time difference using the rough estimate of the Stellar second
    # (~50 us accuracy). Could be improved with better estimate of the Stellar
    # Second.
    diff_time = diff_era_deg * 240.0 * constants.stellar_second

    # Did if any leap seconds occurred between the search start and the final value
    leap_seconds = leap_seconds_between(time0, time0 + diff_time)

    return time0 + diff_time - leap_seconds


@array_utils.vectorize(otypes=[object])
def unix_to_datetime(unix_time):
    """Converts unix time to a :class:`~datetime.datetime` object.

    Equivalent to timezone-aware :meth:`datetime.datetime.fromtimestamp`.

    Parameters
    ----------
    unix_time : float
        Unix/POSIX time.

    Returns
    -------
    dt : :class:`datetime.datetime`
        datetime object from the provided time

    See Also
    --------
    :func:`datetime_to_unix`
    """
    dt = datetime.fromtimestamp(unix_time, timezone.utc)

    return naive_datetime_to_utc(dt)


@array_utils.vectorize(otypes=[np.float64])
def datetime_to_unix(dt):
    """Converts a :class:`~datetime.datetime` object to the unix time.

    This is the inverse of :meth:`datetime.datetime.utcfromtimestamp`.

    Parameters
    ----------
    dt : :class:`datetime.datetime`
        datetime object to convert to unix

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
    dt = naive_datetime_to_utc(dt)
    epoch_start = naive_datetime_to_utc(datetime.fromtimestamp(0, timezone.utc))
    since_epoch = dt - epoch_start
    return since_epoch.total_seconds()


@array_utils.vectorize(otypes=[str])
def datetime_to_timestr(dt):
    """Converts a :class:`~datetime.datetime` to "YYYYMMDDTHHMMSSZ" format.

    Partial seconds are ignored.

    Parameters
    ----------
    dt : :class:`datetime.datetime`
        datetime object to convert to timestring

    Returns
    -------
    time_str : string
        Formated date and time.

    See Also
    --------
    :func:`timestr_to_datetime`
    """
    return dt.strftime("%Y%m%dT%H%M%SZ")


@array_utils.vectorize(otypes=[object])
def timestr_to_datetime(time_str):
    """Converts date "YYYYMMDDTHHMMSS*" to a :class:`~datetime.datetime`.

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


@array_utils.scalarize(dtype=np.int64)
def leap_seconds_between(time_a, time_b):
    """Determine how many leap seconds occurred between two Unix times.

    Parameters
    ----------
    time_a : float
        First Unix/POSIX time.
    time_b : float
        Second Unix/POSIX time.

    Returns
    -------
    int : bool
        The number of leap seconds between *time_a* and *time_b*.
    """
    # Construct the elapse UNIX time
    delta_unix = time_b - time_a

    # Construct the elapsed terrestrial time
    tt_a = unix_to_skyfield_time(time_a).tt
    tt_b = unix_to_skyfield_time(time_b).tt
    delta_tt = (tt_b - tt_a) * 24.0 * 3600

    # Calculate the shift in timescales which should only happen when leap
    # seconds are added/removed
    time_shift = delta_tt - delta_unix
    time_shift_int = np.around(time_shift).astype(int)

    # Check that the shift is an integer number of seconds. I don't know why
    # this wouldn't be true, but if it's not it means things have gone crazy
    if np.any(np.abs(time_shift - time_shift_int) > 0.01):
        raise RuntimeError(
            "Time shifts between TT and UTC does not seem to"
            + " be an integer number of seconds."
        )

    # If the differences are close then there is no leap second
    return time_shift_int


@array_utils.scalarize()
def ensure_unix(time):
    """Convert the input time to Unix time format.

    Parameters
    ----------
    time : float, string, :class:`~datetime.datetime` or :class:`skyfield.timelib.Time`
        Input time, or array of times.

    Returns
    -------
    unix_time : float, or array of
        Output time.
    """
    if isinstance(time[0], datetime):
        return datetime_to_unix(time)

    if isinstance(time[0], str):
        return datetime_to_unix(timestr_to_datetime(time))

    if isinstance(time[0], timelib.Time):
        return datetime_to_unix(time.utc_datetime())

    if isinstance(time, np.ndarray) and np.issubdtype(time.dtype, np.number):
        return time.astype(np.float64)

    raise TypeError(f"Could not convert {type(time)!r} into a UNIX time")


_warned_utc_datetime = False


@array_utils.vectorize(otypes=[object])
def naive_datetime_to_utc(dt):
    """Add UTC timezone info to a naive datetime.

    This only does anything if Skyfield is installed.

    Parameters
    ----------
    dt : datetime
        datetime object without 'tzinfo'

    Returns
    -------
    dt : datetime
        New datetime with `tzinfo` added.
    """
    try:
        from skyfield.api import utc

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=utc)
    except ImportError:
        global _warned_utc_datetime

        if not _warned_utc_datetime:
            warnings.warn(
                "Skyfield not installed. Cannot add UTC timezone to datetime."
            )
            _warned_utc_datetime = True

    return dt


@array_utils.vectorize()
def time_of_day(time):
    """Return the time since the start of the UTC day in seconds.

    Parameters
    ----------
    time : float (UNIX time), or datetime
        Find the start time of the day that `time` is in.

    Returns
    -------
    time : float
        Time since start of UTC day in seconds.
    """
    dt = datetime.fromtimestamp(ensure_unix(time), timezone.utc)
    d = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return (dt - d).total_seconds()
