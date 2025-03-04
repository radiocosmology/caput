r"""Routines for calculation and of solar and sidereal times.

This module can:

- Convert between Python :class:`~datetime.datetime`, UNIX times (as floats),
  and Skyfield time objects.
- Determine the number of leap seconds in an interval (thanks Skyfield!)
- Calculate the Earth Rotation Angle (the successor to Sidereal Time)
- Determine various local time standards (Local Stellar Angle and Day)
- Generate a nice wrapper for Skyfield to ease the loading of required data.

Time Utilities
==============
Time conversion routine which are location independent.

- :py:meth:`unix_to_skyfield_time`
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

Local Time Utilities
====================

Routines which are location specific are grouped into the location aware class :py:class:`Observer`.

This class can be used to calculate Local Stellar Angle (LSA), and the Local
Stellar Day (LSD). LSA is an equivalent to the Local Sidereal Time based around
the Earth Rotation Angle instead of the Greenwich Sidereal Time. This is defined
as:

.. math::
    \mathrm{LSA} = \theta + \lambda

where :math:`\theta` is the Earth Rotation Angle, and :math:`\lambda` is the
longitude. Local Stellar Day counts the stellar days (i.e. the number of cycles
of LSA) that have occured since a given start epoch. This means that the
fractional part is simply the LSA rescaled., with the integer part the number of
stellar days elapsed since that epoch. This requires the specification of the
start epoch, which is determined from a given UNIX time, the code simply picks
the first time :math:`\mathrm{LSA} = 0` after this time.

Note that the quantities LSA and LSD are not really used elsewhere. However, the
concept of a Stellar Day as a length of time is well established (`IERS
constants`_), and the Stellar Angle is an older term for the Earth Rotation
Angle (`NFA Glossary`_).

Skyfield Interface
==================

- :py:class:`SkyfieldWrapper`

This module provides an interface to Skyfield which stores the required datasets
(timescale data and an ephemeris) in a fixed location. The location is
determined by the following (in order):

- As the wrapper is initialised by passing in a ``path=<path>`` option.
- By setting the environment variable ``CAPUT_SKYFIELD_PATH``
- If neither of the above is set, the data is place in ``<path to caput>/caput/data/``

Other skyfield helper functions:

- :py:meth:`skyfield_star_from_ra_dec`
- :py:meth:`skyfield_time_to_unix`

Constants
=========

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

import warnings
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import brentq
from skyfield import timelib
from skyfield.starlib import Star
from skyfield.units import Angle

from . import config
from .misc import listize, scalarize, vectorize

# The approximate length of a UT1 second in SI seconds (i.e. LOD / 86400). This was
# calculated from the IERS EOP C01 IAU2000 data, by calculating the derivative of UT1 -
# TAI from 2019.5 to 2020.5. Note that the variations in this are quite substantial,
# but it's typically 1ms over the source of a day
UT1_S = 1.00000000205

# Approximate number of seconds in a sidereal second.
# The exact value used here is from https://hpiers.obspm.fr/eop-pc/models/constants.html
# but can be derived from USNO Circular 179 Equation 2.12
SIDEREAL_S = 1.0 / 1.002737909350795 * UT1_S

# Approximate length of a stellar second
# This comes from the definition of ERA-UT1 (see IERS Conventions TR Chapter 1) giving
# the first ratio a UT1 and stellar second
STELLAR_S = 1.0 / 1.00273781191135448 * UT1_S


def _fixup_interval_and_step(t0, t1, step):
    # Work routine used by Observer._sr_work and transit_times to sanitise the
    # caller-supplied interval and initial step duration.
    #
    # Returns the interval limits converted to TT JD and the computed
    # initial step size

    # Get the ends of the search interval
    t0 = ensure_unix(t0)
    if t1 is None:
        t1 = t0 + 24 * 3600.0 * STELLAR_S
    else:
        t1 = ensure_unix(t1)
        if t1 <= t0:
            raise ValueError("End of the search interval (t1) is before the start (t0)")

    # Calculate the initial search step
    if step is None:
        if t1 - t0 >= 24 * 3600.0:
            step = 0.2
        else:
            step = (t1 - t0) / (5 * 24 * 3600.0)
    elif step * 24 * 3600 >= t1 - t0:
        raise ValueError("Initial search step is larger than the search interval")

    # Convert the UNIX start and end times into Julian Days
    t0jd, t1jd = unix_to_skyfield_time([t0, t1]).tt

    return t0jd, t1jd, step


class Observer:
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
    sf_wrapper
        (optional) Skyfield wrapper

    Attributes
    ----------
    longitude : float
        Longitude of observer in degrees.
    latitude : float
        Latitude of observer in degrees.
    altitude : float
        Altitude of observer in metres.
    lsd_start_day : float
        UNIX time on the zeroth LSD. The actual zero point is the first time of
        `LSA = 0.0` after the `lsd_start_day`.
    """

    longitude = config.float_in_range(-180.0, 180.0, default=0.0)
    latitude = config.float_in_range(-90.0, 90.0, default=0.0)

    altitude = config.Property(proptype=float, default=0.0)

    lsd_start_day = config.utc_time(default=datetime(2000, 1, 1, 11, 58, 56))

    def __init__(self, lon=0.0, lat=0.0, alt=0.0, lsd_start=None, sf_wrapper=None):
        self.longitude = lon
        self.latitude = lat
        self.altitude = alt

        self.skyfield = skyfield_wrapper if sf_wrapper is None else sf_wrapper

        if lsd_start is not None:
            self.lsd_start_day = lsd_start

    _obs = None

    def get_current_lsd(self) -> float:
        """Get the current LSD."""
        return self.lsd(datetime_to_unix(datetime.utcnow())).astype(np.float64)

    def skyfield_obs(self):
        """Create a Skyfield topos object for the current location.

        Returns
        -------
        obs : :class:`skyfield.toposlib.Topos`
        """
        from skyfield.api import Topos

        earth = self.skyfield.ephemeris["earth"]

        if self._obs is None:
            self._obs = earth + Topos(
                latitude_degrees=self.latitude,
                longitude_degrees=self.longitude,
                elevation_m=self.altitude,
            )

        return self._obs

    @listize()
    def unix_to_lsa(self, time):
        """Calculate the Local Stellar Angle.

        This is the angle between the current meridian and the CIO, i.e. the ERA +
        longitude.

        Parameters
        ----------
        time : float
            Unix time.

        Returns
        -------
        lsa : float
        """
        era = unix_to_era(time)

        return (era + self.longitude) % 360.0

    lsa = unix_to_lsa

    @listize()
    def lsa_to_unix(self, lsa, time0):
        """Convert a Local Stellar Angle (LSA) on a given day to a UNIX time.

        Parameters
        ----------
        lsa : scalar or np.ndarray
            Local Earth Rotation Angle degrees to convert.
        time0 : scalar or np.ndarray
            An earlier time within 24 sidereal hours. For example,
            the start of the solar day of the observation.

        Returns
        -------
        time : scalar or np.ndarray
            Corresponding UNIX time.
        """
        era = (lsa - self.longitude) % 360.0

        return era_to_unix(era, time0)

    def lsd_zero(self):
        """Return the zero point of LSD as a UNIX time.

        Returns
        -------
        lsd_zero : float
        """
        return self.lsa_to_unix(0.0, self.lsd_start_day)

    @listize()
    def unix_to_lsd(self, time):
        """Calculate the Local Stellar Day (LSD) corresponding to the given time.

        The Local Earth Rotation Day is the number of cycles of Earth Rotation
        Angle that have passed since the specified zero epoch (including
        fractional cycles).

        Parameters
        ----------
        time :  float or array of
            UNIX time

        Returns
        -------
        lsd : float or array of
        """
        # Get fractional part from LRA
        frac_part = self.unix_to_lsa(time) / 360.0

        # Calculate the approximate CSD by crudely dividing the time difference by
        # the length of a sidereal day
        approx_lsd = (time - self.lsd_zero()) / (24.0 * 3600.0 * SIDEREAL_S)

        # Subtract the accurate part, and round to the nearest integer to get the
        # number of whole days elapsed (should be accurate for a very long time)
        whole_days = np.rint(approx_lsd - frac_part)

        return whole_days + frac_part

    lsd = unix_to_lsd

    @listize()
    def lsd_to_unix(self, lsd):
        """Calculate the UNIX time corresponding to a given LSD.

        Parameters
        ----------
        lsd : float or array of
            Local Stellar Day to convert to unix

        Returns
        -------
        time :  float or array of
            UNIX time
        """
        # Find the approximate UNIX time
        approx_unix = self.lsd_zero() + lsd * 3600 * 24 * SIDEREAL_S

        # Shift to 12 hours before to give the start of the search period
        start_unix = approx_unix - 12 * 3600

        # Get the LRA from the LSD in degrees
        lsa = 360.0 * (lsd % 1.0)

        # Solve for the next transit of that RA after start_unix
        return self.lsa_to_unix(lsa, start_unix)

    @listize()
    def unix_to_lst(self, unix):
        """Calculate the apparent Local Sidereal Time for the given UNIX time.

        Parameters
        ----------
        unix : float or array of
            UNIX time in floating point seconds since the epoch.

        Returns
        -------
        lst : float or array of
            The apparent LST in degrees.
        """
        st = unix_to_skyfield_time(unix)

        return (st.gast * 15.0 + self.longitude) % 360.0

    lst = unix_to_lst

    @scalarize()
    def transit_RA(self, time):
        """Transiting RA for the observer at given Unix Time.

        Because the RA is defined with respect to the specified epoch (J2000 by
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
        aberration.  See the discussion
        `on stackoverflow <http://stackoverflow.com/questions/11970713>`_.
        Some testing does seem to indicate that these effects are accounted for.

        This calculates the RA in the given epoch which by default is J2000, but
        it might be more appropriate to use an epoch that is closer to the
        observation time. The mismatch in the celestial poles is not
        insignificant (~5 arcmin from J2000 to J2016).

        PyEphem uses all geocentric latitudes, which I don't think affects
        this calculation.
        """
        # Initialize Skyfield location object.
        obs = self.skyfield_obs()

        # Want the RA at the equator, which is much less affected by the celestial
        # pole mismatch between now and J2000 epoch.
        az = 180.0
        el = 90.0 - self.latitude
        obs.pressure = 0

        st = unix_to_skyfield_time(time)
        pos = obs.at(st).from_altaz(az_degrees=az, alt_degrees=el)
        ra, _, _ = pos.radec()  # Fetch ICRS position (effectively J2000)

        return np.degrees(ra.radians)

    def transit_times(
        self, source, t0, t1=None, step=None, lower=False, return_dec=False
    ):
        """Find the transit times of the given source in an interval.

        Parameters
        ----------
        source : skyfield source or float
            The source we are calculating the transit of. This can be any body
            skyfield can observe, such as a star (`skyfield.api.Star`), planet or
            moon (`skyfield.vectorlib.VectorSum` or
            `skyfield.jpllib.ChebyshevPosition`). Additionally if a float is passed,
            this is equivalent to a body with ICRS RA given by the float, and DEC=0.
        t0 : float unix time, or datetime
            The start time to search for. Any type that can be converted to a UNIX
            time by caput.
        t1 : float unix time, or datetime, optional
            The end time of the search interval. If not set, this is 1 day after the
            start time `t0`.
        step : float or None, optional
            The initial search step in days. This is used to find the approximate
            times of transit, and should be set to something less than the spacing
            between events.  If None is passed, an initial search step of 0.2 days,
            or else one fifth of the specified interval, is used, whichever is smaller.
        lower : bool, optional
            By default this only returns the upper (regular) transit. This will cause
            lower transits to be returned instead.
        return_dec : bool, optional
            If set, also return the declination of the source at transit.

        Returns
        -------
        times : np.ndarray
            UNIX times of transits.
        dec : np.ndarray
            Only returned if `return_dec` is set. Declination of source at transit.
        """
        if isinstance(source, float):
            source = skyfield_star_from_ra_dec(source, 0.0)

        # The function to find routes for. For the upper transit we just search for
        # HA=0, for the lower transit we need to rotate the 180 -> -180 transition to
        # be at 0.
        def f(t):
            ha = self._source_ha(source, t)
            return ha if not lower else (ha % 360.0) - 180.0

        # Convert interval to Julian Days and choose the initial search step if not set
        t0jd, t1jd, step = _fixup_interval_and_step(t0, t1, step)

        # Compute transits
        transits, _ = _solve_all(f, t0jd, t1jd, step, skip_decreasing=True, xtol=1e-8)

        # Convert into UNIX times
        t_sf = self.skyfield.timescale.tt_jd(transits)
        t_unix = skyfield_time_to_unix(t_sf)

        if return_dec:
            if len(transits) > 0:
                pos = self.skyfield_obs().at(t_sf).observe(source)
                dec = pos.cirs_radec(epoch=t_sf)[1]._degrees
            else:
                dec = np.array([], dtype=np.float64)

            return t_unix, dec

        return t_unix

    def rise_set_times(self, source, t0, t1=None, step=None, diameter=0.0):
        """Find all times a sources rises or sets in an interval.

        Typical atmospheric refraction at the horizon is 34 arcminutes, but
        this method does _not_ take that into account.

        Parameters
        ----------
        source : skyfield source
            The source we are calculating the rising and setting of.
        t0 : float unix time, or datetime
            The start time to search for. Any type that be converted to a UNIX time
            by caput.
        t1 : float unix time, or datetime, optional
            The end time of the search interval. If not set, this is 1 day after the
            start time `t0`.
        step : float or None, optional
            The initial search step in days. This is used to find the approximate
            times of risings and settings, and should be set to something less than the
            spacing between events.  If None is passed, an initial search step of
            0.2 days, or else one fifth of the specified interval, is used, whichever
            is smaller.
        diameter : float
            The size of the source in degrees. Use this to ensure the whole source is
            below the horizon. Also, if the local horizon is higher (i.e. mountains),
            this can be set to a negative value to account for this.  You may also
            use this parameter to account for atmospheric refraction, if desired,
            by adding 68 arcminutes to the nominal diameter.

        Returns
        -------
        times : np.ndarray
            Source rise/set times as UNIX epoch times.
        rising : np.ndarray
            Boolean array of whether the time corresponds to a rising (True) or
            setting (False).
        """
        return self._sr_work(
            source, t0, t1, step, diameter, skip_rise=False, skip_set=False
        )

    def rise_times(self, source, t0, t1=None, step=None, diameter=0.0):
        """Find all times a sources rises in an interval.

        Typical atmospheric refraction at the horizon is 34 arcminutes, but
        this method does _not_ take that into account.

        Parameters
        ----------
        source : skyfield source
            The source we are calculating the rising of.
        t0 : float unix time, or datetime
            The start time to search for. Any type that be converted to a UNIX time
            by caput.
        t1 : float unix time, or datetime, optional
            The end time of the search interval. If not set, this is 1 day after the
            start time `t0`.
        step : float or None, optional
            The initial search step in days. This is used to find the approximate
            times of rising, and should be set to something less than the spacing
            between events.  If None is passed, an initial search step of 0.2 days,
            or else one fifth of the specified interval, is used, whichever is smaller.
        diameter : float
            The size of the source in degrees. Use this to ensure the whole source is
            below the horizon. Also, if the local horizon is higher (i.e. mountains),
            this can be set to a negative value to account for this.  You may also
            use this parameter to account for atmospheric refraction, if desired,
            by adding 68 arcminutes to the nominal diameter.

        Returns
        -------
        times : np.ndarray
            Source rise times as UNIX epoch times.
        """
        return self._sr_work(
            source, t0, t1, step, diameter, skip_rise=False, skip_set=True
        )[0]

    def set_times(self, source, t0, t1=None, step=None, diameter=0.0):
        """Find all times a sources sets in an interval.

        Typical atmospheric refraction at the horizon is 34 arcminutes, but
        this method does _not_ take that into account.

        Parameters
        ----------
        source : skyfield source
            The source we are calculating the setting of.
        t0 : float unix time, or datetime
            The start time to search for. Any type that be converted to a UNIX time
            by caput.
        t1 : float unix time, or datetime, optional
            The end time of the search interval. If not set, this is 1 day after the
            start time `t0`.
        step : float or None, optional
            The initial search step in days. This is used to find the approximate
            times of setting, and should be set to something less than the spacing
            between events.  If None is passed, an initial search step of 0.2 days,
            or else one fifth of the specified interval, is used, whichever is smaller.
        diameter : float
            The size of the source in degrees. Use this to ensure the whole source is
            below the horizon. Also, if the local horizon is higher (i.e. mountains),
            this can be set to a negative value to account for this.  You may also
            use this parameter to account for atmospheric refraction, if desired,
            by adding 68 arcminutes to the nominal diameter.

        Returns
        -------
        times : np.ndarray
            Source rise times as UNIX epoch times.
        """
        return self._sr_work(
            source, t0, t1, step, diameter, skip_rise=True, skip_set=False
        )[0]

    def solar_transit(self, t0, t1=None, step=None, lower=False, return_dec=False):
        """Find the Solar transits between two times.

        Parameters
        ----------
        t0 : float unix time, or datetime
            The start time to search for. Any type that be converted to a UNIX time
            by caput.
        t1 : float unix time, or datetime, optional
            The end time of the search interval. If not set, this is 1 day after the
            start time `t0`.
        step : float or None, optional
            The initial search step in days. This is used to find the approximate
            times of transit, and should be set to something less than the spacing
            between events.  If None is passed, an initial search step of 0.2 days,
            or else one fifth of the specified interval, is used, whichever is smaller.
        lower : bool, optional
            By default this only returns the upper (regular) transit. This will cause
            lower transits to be returned instead.
        return_dec : bool, optional
            If set, also return the declination of the source at transit.

        Returns
        -------
        times : np.ndarray
            Solar transit times as UNIX epoch times.
        """
        return self.transit_times(
            skyfield_wrapper.ephemeris["sun"], t0, t1, step, lower, return_dec
        )

    def lunar_transit(self, t0, t1=None, step=None, lower=False, return_dec=False):
        """Find the Lunar transits between two times.

        Parameters
        ----------
        t0 : float unix time, or datetime
            The start time to search for. Any type that be converted to a UNIX time
            by caput.
        t1 : float unix time, or datetime, optional
            The end time of the search interval. If not set, this is 1 day after the
            start time `t0`.
        step : float or None, optional
            The initial search step in days. This is used to find the approximate
            times of transit, and should be set to something less than the spacing
            between events.  If None is passed, an initial search step of 0.2 days,
            or else one fifth of the specified interval, is used, whichever is smaller.
        lower : bool, optional
            By default this only returns the upper (regular) transit. This will cause
            lower transits to be returned instead.
        return_dec : bool, optional
            If set, also return the declination of the source at transit.

        Returns
        -------
        times : np.ndarray
            Lunar transit times as UNIX epoch times.
        """
        return self.transit_times(
            skyfield_wrapper.ephemeris["moon"], t0, t1, step, lower, return_dec
        )

    def solar_setting(self, t0, t1=None, step=None):
        """Find the Solar settings between two times.

        This method calculates the conventional astronomical sunset, which
        occurs when the centre of the sun is 50 arcminutes below the horizon.
        This accounts for a solar diameter of 32 arcminutes, plus 34 arcminutes
        of atmospheric refraction at the horizon.

        Parameters
        ----------
        t0 : float unix time, or datetime
            The start time to search for. Any type that be converted to a UNIX time
            by caput.
        t1 : float unix time, or datetime, optional
            The end time of the search interval. If not set, this is 1 day after the
            start time `t0`.
        step : float or None, optional
            The initial search step in days. This is used to find the approximate
            times of setting, and should be set to something less than the spacing
            between events.  If None is passed, an initial search step of 0.2 days,
            or else one fifth of the specified interval, is used, whichever is smaller.

        Returns
        -------
        times : np.ndarray
            Solar setting times as UNIX epoch times.
        """
        return self.set_times(
            skyfield_wrapper.ephemeris["sun"],
            t0,
            t1,
            step,
            diameter=100.0 / 60,
        )

    def lunar_setting(self, t0, t1=None, step=None):
        """Find the Lunar settings between two times.

        This method calculates the conventional astronomical moonset, which
        occurs when the centre of the moon is 50 arcminutes below the horizon.
        This accounts for a lunar diameter of 32 arcminutes, plus 34 arcminutes
        of atmospheric refraction at the horizon.

        Parameters
        ----------
        t0 : float unix time, or datetime
            The start time to search for. Any type that be converted to a UNIX time
            by caput.
        t1 : float unix time, or datetime, optional
            The end time of the search interval. If not set, this is 1 day after the
            start time `t0`.
        step : float or None, optional
            The initial search step in days. This is used to find the approximate
            times of setting, and should be set to something less than the spacing
            between events.  If None is passed, an initial search step of 0.2 days,
            or else one fifth of the specified interval, is used, whichever is smaller.

        Returns
        -------
        times : np.ndarray
            Lunar setting times as UNIX epoch times.
        """
        return self.set_times(
            skyfield_wrapper.ephemeris["moon"],
            t0,
            t1,
            step,
            diameter=100.0 / 60,
        )

    def solar_rising(self, t0, t1=None, step=None):
        """Find the Solar risings between two times.

        This method calculates the conventional astronomical sunrise, which
        occurs when the centre of the sun is 50 arcminutes below the horizon.
        This accounts for a solar diameter of 32 arcminutes, plus 34 arcminutes
        of atmospheric refraction at the horizon.

        Parameters
        ----------
        t0 : float unix time, or datetime
            The start time to search for. Any type that be converted to a UNIX time
            by caput.
        t1 : float unix time, or datetime, optional
            The end time of the search interval. If not set, this is 1 day after the
            start time `t0`.
        step : float or None, optional
            The initial search step in days. This is used to find the approximate
            times of rising, and should be set to something less than the spacing
            between events.  If None is passed, an initial search step of 0.2 days,
            or else one fifth of the specified interval, is used, whichever is smaller.

        Returns
        -------
        times : np.ndarray
            Solar rising times as UNIX epoch times.
        """
        return self.rise_times(
            skyfield_wrapper.ephemeris["sun"],
            t0,
            t1,
            step,
            diameter=100.0 / 60,
        )

    def lunar_rising(self, t0, t1=None, step=None):
        """Find the Lunar risings between two times.

        This method calculates the conventional astronomical moonrise, which
        occurs when the centre of the moon is 50 arcminutes below the horizon.
        This accounts for a lunar diameter of 32 arcminutes, plus 34 arcminutes
        of atmospheric refraction at the horizon.

        Parameters
        ----------
        t0 : float unix time, or datetime
            The start time to search for. Any type that be converted to a UNIX time
            by caput.
        t1 : float unix time, or datetime, optional
            The end time of the search interval. If not set, this is 1 day after the
            start time `t0`.
        step : float or None, optional
            The initial search step in days. This is used to find the approximate
            times of rising, and should be set to something less than the spacing
            between events.  If None is passed, an initial search step of 0.2 days,
            or else one fifth of the specified interval, is used, whichever is smaller.

        Returns
        -------
        times : np.ndarray
            Lunar rising times as UNIX epoch times.
        """
        return self.rise_times(
            skyfield_wrapper.ephemeris["moon"],
            t0,
            t1,
            step,
            diameter=100.0 / 60,
        )

    def object_coords(self, source, date=None, deg=False):
        """Calculates the RA and DEC of the source.

        Gives the ICRS coordinates if no date is given (=J2000), or if a date is
        specified gives the CIRS coordinates at that epoch.

        This also returns the *apparent* position, including abberation and
        deflection by gravitational lensing. This shifts the positions by up to
        20 arcseconds.

        Parameters
        ----------
        source : skyfield source
            skyfield.starlib.Star or skyfield.vectorlib.VectorSum or
            skyfield.jpllib.ChebyshevPosition body representing the source.
        date : float
            Determine coordinates at this unix time.  If None, use Jan 01 2000.
        deg : bool
            Return coordinates in degrees if True, radians if False (default).

        Returns
        -------
        ra, dec: float
            Position of the source.
        """
        if date is None:  # No date, get ICRS coords
            if isinstance(body, Star):
                ra, dec = body.ra.radians, body.dec.radians
            else:
                raise ValueError(
                    "Source is not fixed, cannot calculate coordinates without a date."
                )

        else:  # Calculate CIRS position with all corrections
            t = unix_to_skyfield_time(date)
            radec = self.skyfield_obs().at(t).observe(source).apparent().cirs_radec(t)

            ra, dec = radec[0].radians, radec[1].radians

        # If requested, convert to degrees
        if deg:
            ra = np.degrees(ra)
            dec = np.degrees(dec)

        return ra, dec

    def _sr_work(self, source, t0, t1, step, diameter, skip_rise=False, skip_set=False):
        # A work routine factoring out common functionality

        # The function to find roots for. This is just the altitude of the source with
        # an offset for it's finite size
        def f(t):
            return self._source_alt(source, t) + diameter / 2

        # Convert interval to Julian Days and choose the initial search step if not set
        t0jd, t1jd, step = _fixup_interval_and_step(t0, t1, step)

        times, risings = _solve_all(
            f,
            t0jd,
            t1jd,
            step,
            skip_increasing=skip_rise,
            skip_decreasing=skip_set,
            xtol=1e-6,
        )

        # Convert into UNIX times
        times = skyfield_time_to_unix(self.skyfield.timescale.tt_jd(times))

        return times, risings

    def _source_ha(self, source, t):
        # Calculate the local Hour Angle of a given source
        time = self.skyfield.timescale.tt_jd(t)
        lst = time.gast * 15 + self.longitude

        ra = self.skyfield_obs().at(time).observe(source).radec(epoch=time)[0]._degrees

        return (((lst - ra) + 180) % 360) - 180

    def _source_alt(self, source, t):
        # Calculate the altitude of a given source
        from skyfield.positionlib import _to_altaz

        time = self.skyfield.timescale.tt_jd(t)
        pos = self.skyfield_obs().at(time).observe(source)

        # NOTE: we could have used `.apparent().altz()` here, but the corrections for
        # light travel time are super slow, so we're better off skipping them. To do
        # this we need to use a private Skyfield method. This isn't a great idea given
        # how unstable skyfields internal API seems to be, but it saves reinventing the
        # wheel. We'll see how it goes for now.
        return _to_altaz(pos, None, None)[0].degrees


@scalarize()
def unix_to_skyfield_time(unix_time):
    """Formats the Unix time into a time that can be interpreted by Skyfield.

    Parameters
    ----------
    unix_time : float or array of.
        Unix/POSIX time.

    Returns
    -------
    time : :class:`skyfield.timelib.Time`
    """
    ts = skyfield_wrapper.timescale

    days, seconds = divmod(unix_time, 24 * 3600.0)

    # Construct Julian day and convert to calendar day
    year, month, day = timelib.calendar_date(2440588 + days)

    # Construct Skyfield time. Cheat slightly by putting all of the time of day
    # in the `second` argument.
    return ts.utc(year, month, day, second=seconds)


@scalarize()
def skyfield_time_to_unix(skyfield_time):
    """Formats the Skyfield time into UNIX times.

    Parameters
    ----------
    skyfield_time : `skyfield.timelib.Time`
        Skyfield time.

    Returns
    -------
    time : float or array of
        UNIX time.
    """
    # TODO: I'm surprised there isn't a better way to do this. Needing to convert via a
    # datetime isn't great, but the only other ways I think can do it use private
    # routines
    return ensure_unix(skyfield_time.utc_datetime())


@scalarize()
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


@scalarize()
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
    diff_time = diff_era_deg * 240.0 * STELLAR_S

    # Did if any leap seconds occurred between the search start and the final value
    leap_seconds = leap_seconds_between(time0, time0 + diff_time)

    return time0 + diff_time - leap_seconds


@vectorize(otypes=[object])
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


@vectorize(otypes=[np.float64])
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


@vectorize(otypes=[str])
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


@vectorize(otypes=[object])
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


@scalarize(dtype=np.int64)
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


@scalarize()
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


@vectorize(otypes=[object])
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


@vectorize()
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


class SkyfieldWrapper:
    """A wrapper to help with loading Skyfield and its data.

    Parameters
    ----------
    path : string, optional
        Directory Skyfield should save data in. If not set data will be looked
        for in `$CAPUT_SKYFIELD_PATH` or in `<path to caput>/caput/data`.
    expire : bool, optional
        Deprecated option. Skyfield no longer has a concept of expiring data. To get
        updated data you must force an explicit reload of it which can be done via
        `SkyFieldWrapper.reload`.
    ephemeris : string, optional
        The JPL ephemeris to use. Defaults to `'de421.bsp'`.
    """

    mirror_url = "https://bao.chimenet.ca/skyfield/"

    def __init__(self, path=None, expire=None, ephemeris="de421.bsp"):
        import os

        self._ephemeris_name = ephemeris

        if expire is not None:
            warnings.warn(
                "`expiry` argument deprecated as Skyfield has dropped the idea of "
                "expiring data.",
                DeprecationWarning,
            )

        if path is None:
            if "CAPUT_SKYFIELD_PATH" in os.environ:
                path = os.environ["CAPUT_SKYFIELD_PATH"]
            else:
                path = os.path.join(os.path.dirname(__file__), "data", "")

        # Defer failure if Skyfield is not available until we try to load
        # anything
        try:
            from skyfield import api

            self._load = api.Loader(path)
        except ImportError:
            pass

    _load = None

    @property
    def load(self):
        """A :class:`skyfield.iokit.Loader` object.

        This is to be used in the same way as `skyfield.api.load`,
        in case you want something other than `timescale` or `ephemeris`.
        """
        if self._load is None:
            raise RuntimeError("Skyfield is not installed.")
        return self._load

    @property
    def path(self):
        """The path to the Skyfield data."""
        return self.load.directory

    _timescale = None

    @property
    def timescale(self):
        """A :class:`skyfield.timelib.Timescale` object.

        Loaded at first call and then cached.
        """
        if self._timescale:
            return self._timescale

        try:
            self._timescale = self.load.timescale(builtin=False)
        except OSError:
            warnings.warn(
                "Could not get timescale data from an official source. Trying the "
                "CHIME mirror, but the products are likely out of date."
            )
            timescale_files = ["Leap_Second.dat", "finals2000A.all"]
            try:
                for file in timescale_files:
                    self.load.download(self.mirror_url + file)

                self._timescale = self.load.timescale(builtin=False)
            except OSError as e:
                raise OSError(
                    "Could not get ephemeris either from an existing installation at "
                    f"{self.load.directory}, an official download source or the CHIME "
                    f"mirror. If you can find a working mirror of "
                    f"{timescale_files} try using "
                    '`caput.time.skyfield_wrapper.load.download("<mirror_url>") '
                    "to download directly."
                ) from e

        return self._timescale

    _ephemeris = None

    @property
    def ephemeris(self):
        """A Skyfield ephemeris object (:class:`skyfield.jpllib.SpiceKernel`).

        Loaded at first call, and then cached.
        """
        if self._ephemeris:
            return self._ephemeris

        try:
            self._ephemeris = self.load(self._ephemeris_name)
            return self._ephemeris
        except OSError:
            warnings.warn(
                "Could not get ephemeris data from an official source. Trying the "
                "CHIME mirror, but the products are likely out of date."
            )
            try:
                self.load.download(self.mirror_url + self._ephemeris_name)
                self._ephemeris = self.load(self._ephemeris_name)
            except OSError as e:
                raise OSError(
                    "Could not get ephemeris either from an existing installation at "
                    f"{self.load.directory}, an official download source or the CHIME "
                    f"mirror. If you can find a working mirror of "
                    "{self._ephemeris_name} try using "
                    '`caput.time.skyfield_wrapper.load.download("<mirror_url>") '
                    "to download directly."
                ) from e

    def reload(self):
        """Reload the Skyfield data regardless of the `expire` setting.

        This will only load the data from the official sources.
        """
        # Download the timescale file and ephemeris
        self.load.download("finals2000A.all")
        self.load.download(self._ephemeris_name)


# Set up a module local Skyfield wrapper for time conversion functions in this
# module to use.
skyfield_wrapper = SkyfieldWrapper()


def _solve_all(f, x0, x1, dx, skip_increasing=False, skip_decreasing=False, **kwargs):
    """Find all roots of function f, within the interval [x0, x1].

    To find all the roots we need an estimate of the spacing of the roots. This is
    specified by the parameter `dx`. If this is too large, roots may be missed.

    Parameters
    ----------
    f : callable
        A numpy vectorized function which returns a single value f(x).
    x0, x1 : float
        The start and end of the interval to search.
    dx : float
        An interval known to be less than the spacing between the closest roots.
    skip_increasing, skip_decreasing : bool, optional
        Skip roots where the gradient is positive (or negative).
    **kwargs
        Passed to `scipy.optimize.brentq`. `xtol`, `rtol` and `maxiter` are probably
        the most useful.

    Returns
    -------
    roots : np.ndarray[np.float64]
        The values of the roots found.
    increasing : np.ndarray[bool]
        If the gradient at the root is positive.
    """
    # Form a grid of points to find intervals to search over
    x_init = np.linspace(x0, x1, int(np.ceil((x1 - x0) / dx)), endpoint=True)
    f_init = f(x_init)

    roots = []
    increasing = []

    # Search through intervals
    for xa, xb, fa, fb in zip(x_init[:-1], x_init[1:], f_init[:-1], f_init[1:]):
        # Entries are the same sign, so there is no solution in between.
        # NOTE: we need to deal with the case where one edge might be an exact root,
        # hence the strictly greater than 0.0
        if fa * fb > 0.0:
            continue

        is_increasing = fa < fb

        # Skip positive gradient roots
        if skip_increasing and is_increasing:
            continue

        # Skip negative gradient roots
        if skip_decreasing and not is_increasing:
            continue

        root = brentq(f, xa, xb, **kwargs)

        roots.append(root)
        increasing.append(is_increasing)

    return (np.array(roots, dtype=np.float64), np.array(increasing, dtype=bool))


def skyfield_star_from_ra_dec(ra, dec, name=()):
    """Create a Skyfield star object from an ICRS position.

    Parameters
    ----------
    ra, dec : float
        The ICRS position in degrees.
    name : str or tuple/list of str, optional
        The name(s) of the body.

    Returns
    -------
    body : skyfield.starlib.Star
        An object representing the body.
    """
    if isinstance(name, str):
        name = (name,)

    return Star(
        ra=Angle(degrees=ra, preference="hours"), dec=Angle(degrees=dec), names=name
    )
