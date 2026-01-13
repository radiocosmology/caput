r"""Time and ephemeris calculations for a local observer.

Local Time Utilities
====================

Routines are provided through the location-aware class :py:class:`Observer`.

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

.. _`NFA Glossary`: http://syrte.obspm.fr/iauWGnfa/NFA_Glossary.pdf

.. _`IERS constants`: http://hpiers.obspm.fr/eop-pc/models/constants.html
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, overload

if TYPE_CHECKING:
    import numpy.typing as npt

from datetime import datetime, timezone

import numpy as np
from scipy.optimize import brentq
from skyfield.jpllib import ChebyshevPosition
from skyfield.starlib import Star
from skyfield.units import Angle
from skyfield.vectorlib import VectorSum

from .. import config
from ..util import arraytools
from . import constants
from . import skyfield as csf
from . import time as ctime

_SkySourceLike = TypeVar("_SkySourceLike", bound=Star | VectorSum | ChebyshevPosition)


__all__ = ["Observer"]


class Observer:
    """Time calculations for a local observer.

    Parameters
    ----------
    lon : float
        Longitude of observer in degrees.
    lat : float
        Latitude of observer in degrees.
    alt : float
        Altitude of observer in metres.
    lsd_start : int | float, optional
        The zeroth LSD. If not set use the J2000 epoch start.
    sf_wrapper : :py:class:`~caput.astro.skyfield.SkyfieldWrapper`, optional
        Skyfield wrapper.

    Attributes
    ----------
    longitude : float
        Longitude of observer in degrees.
    latitude : float
        Latitude of observer in degrees.
    altitude : float
        Altitude of observer in metres.
    lsd_start_day : int | float, optional
        UNIX time on the zeroth LSD. The actual zero point is the first instance of
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

        self.skyfield = csf.skyfield_wrapper if sf_wrapper is None else sf_wrapper

        if lsd_start is not None:
            self.lsd_start_day = lsd_start

    _obs = None

    def get_current_lsd(self):
        """Get the current LSD."""
        return self.lsd(ctime.datetime_to_unix(datetime.now(timezone.utc))).astype(
            np.float64
        )

    def skyfield_obs(self):
        """Create a Skyfield topos object for the current location.

        Returns
        -------
        skyfield_topos : skyfield.toposlib.Topos
            Syfield object representing the current location.
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

    @arraytools.listize()
    def unix_to_lsa(self, time):
        """Calculate the Local Stellar Angle.

        This is the angle between the current meridian and the CIRS Celestial
        Intermediate Origin (CIO), i.e. the ERA + longitude.

        Parameters
        ----------
        time : array_like
            Unix time.

        Returns
        -------
        lsa : ndarray
            Local Stellar Angle in degrees.
        """
        era = ctime.unix_to_era(time)

        return (era + self.longitude) % 360.0

    lsa = unix_to_lsa

    @arraytools.listize()
    def lsa_to_unix(self, lsa, time0):
        """Convert a Local Stellar Angle (LSA) on a given day to a UNIX time.

        Parameters
        ----------
        lsa : array_like
            Local Stellar Angle to convert, in degrees.
        time0 : array_like
            An earlier UNIX time within 24 sidereal hours. For example,
            the start of the solar day of the observation.

        Returns
        -------
        time : ndarray
            Corresponding UNIX time.
        """
        era = (lsa - self.longitude) % 360.0

        return ctime.era_to_unix(era, time0)

    def lsd_zero(self):
        """Return the zero point of LSD as a UNIX time.

        Returns
        -------
        lsd_zero : float
            Zero point of LSD as UNIX time.
        """
        return self.lsa_to_unix(0.0, self.lsd_start_day)

    @arraytools.listize()
    def unix_to_lsd(self, time):
        """Calculate the Local Stellar Day (LSD) corresponding to the given time.

        The Local Earth Rotation Day is the number of cycles of Earth Rotation
        Angle that have passed since the specified zero epoch (including
        fractional cycles).

        Parameters
        ----------
        time : array_like
            UNIX time.

        Returns
        -------
        lsd : ndarray
            LSD representation of provided time.
        """
        # Get fractional part from LRA
        frac_part = self.unix_to_lsa(time) / 360.0

        # Calculate the approximate CSD by crudely dividing the time difference by
        # the length of a sidereal day
        approx_lsd = (time - self.lsd_zero()) / (
            24.0 * 3600.0 * constants.sidereal_second
        )

        # Subtract the accurate part, and round to the nearest integer to get the
        # number of whole days elapsed (should be accurate for a very long time)
        whole_days = np.rint(approx_lsd - frac_part)

        return whole_days + frac_part

    lsd = unix_to_lsd

    @arraytools.listize()
    def lsd_to_unix(self, lsd):
        """Calculate the UNIX time corresponding to a given LSD.

        Parameters
        ----------
        lsd : array_like
            Local Stellar Day to convert to unix.

        Returns
        -------
        unix_time : ndarray
            UNIX time.
        """
        # Find the approximate UNIX time
        approx_unix = self.lsd_zero() + lsd * 3600 * 24 * constants.sidereal_second

        # Shift to 12 hours before to give the start of the search period
        start_unix = approx_unix - 12 * 3600

        # Get the LRA from the LSD in degrees
        lsa = 360.0 * (lsd % 1.0)

        # Solve for the next transit of that RA after start_unix
        return self.lsa_to_unix(lsa, start_unix)

    @arraytools.listize()
    def unix_to_lst(self, unix):
        """Calculate the apparent Local Sidereal Time for the given UNIX time.

        Parameters
        ----------
        unix : array_like
            UNIX time in floating point seconds since the epoch.

        Returns
        -------
        lst : ndarray
            The apparent LST in degrees.
        """
        st = csf.unix_to_skyfield_time(unix)

        return (st.gast * 15.0 + self.longitude) % 360.0

    lst = unix_to_lst

    @arraytools.scalarize()
    def transit_RA(self, time):
        """Transiting RA for the observer at given Unix Time.

        Because the RA is defined with respect to the specified epoch (J2000 by
        default), the elevation actually matters here. The elevation of the
        equator is used, which minimizes this effect.

        Parameters
        ----------
        time : array_like
            Time as specified by the Unix/POSIX time.

        Returns
        -------
        RA : float
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

        st = csf.unix_to_skyfield_time(time)
        pos = obs.at(st).from_altaz(az_degrees=az, alt_degrees=el)
        ra, _, _ = pos.radec()  # Fetch ICRS position (effectively J2000)

        return np.degrees(ra.radians)

    def transit_times(
        self, source, t0, t1=None, step=None, lower=False, return_dec=False
    ):
        """Find the transit times of the given source in an interval.

        Parameters
        ----------
        source : _SkySourceLike
            The source we are calculating the transit of. This can be any body
            skyfield can observe, such as a star (`skyfield.api.Star`), planet or
            moon (`skyfield.vectorlib.VectorSum` or
            `skyfield.jpllib.ChebyshevPosition`). Additionally if a float is passed,
            this is equivalent to a body with ICRS RA given by the float, and DEC=0.
        t0 : ctime.TimeLike
            The start time to search for. Any type that can be converted to a UNIX
            time by caput.
        t1 : ctime.TimeLike | None, optional
            The end time of the search interval. If not set, this is 1 day after the
            start time `t0`.
        step : float | None, optional
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
        unix_times : ndarray
            UNIX times of transits.
        dec : bool (if `return_dec`)
            Declination of source at transit, if `return_dec` is set.
        """
        if isinstance(source, float):
            source = csf.skyfield_star_from_ra_dec(source, 0.0)

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
        t_unix = csf.skyfield_time_to_unix(t_sf)

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
        source : _SkySourceLike
            The source we are calculating the rising and setting of.
        t0 : ctime.TimeLike
            The start time to search for. Any type that be converted to a UNIX time
            by caput.
        t1 : ctime.TimeLike | None, optional
            The end time of the search interval. If not set, this is 1 day after the
            start time `t0`.
        step : float | None, optional
            The initial search step in days. This is used to find the approximate
            times of risings and settings, and should be set to something less than the
            spacing between events.  If None is passed, an initial search step of
            0.2 days, or else one fifth of the specified interval, is used, whichever
            is smaller.
        diameter : float, optional
            The size of the source in degrees. Use this to ensure the whole source is
            below the horizon. Also, if the local horizon is higher (i.e. mountains),
            this can be set to a negative value to account for this.  You may also
            use this parameter to account for atmospheric refraction, if desired,
            by adding 68 arcminutes to the nominal diameter.

        Returns
        -------
        unix_times : ndarray
            Source rise/set times as UNIX epoch times.
        rise_or_set : bool ndarray
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
        source : _SkySourceLike
            The source we are calculating the rising of.
        t0 : ctime.TimeLike
            The start time to search for. Any type that be converted to a UNIX time
            by caput.
        t1 : ctime.TimeLike | None, optional
            The end time of the search interval. If not set, this is 1 day after the
            start time `t0`.
        step : float | None, optional
            The initial search step in days. This is used to find the approximate
            times of rising, and should be set to something less than the spacing
            between events.  If None is passed, an initial search step of 0.2 days,
            or else one fifth of the specified interval, is used, whichever is smaller.
        diameter : float, optional
            The size of the source in degrees. Use this to ensure the whole source is
            below the horizon. Also, if the local horizon is higher (i.e. mountains),
            this can be set to a negative value to account for this.  You may also
            use this parameter to account for atmospheric refraction, if desired,
            by adding 68 arcminutes to the nominal diameter.

        Returns
        -------
        unix_times : ndarray
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
        source : _SkySourceLike
            The source we are calculating the setting of.
        t0 : ctime.TimeLike
            The start time to search for. Any type that be converted to a UNIX time
            by caput.
        t1 : ctime.TimeLike | None, optional
            The end time of the search interval. If not set, this is 1 day after the
            start time `t0`.
        step : float | None, optional
            The initial search step in days. This is used to find the approximate
            times of setting, and should be set to something less than the spacing
            between events.  If None is passed, an initial search step of 0.2 days,
            or else one fifth of the specified interval, is used, whichever is smaller.
        diameter : float, optional
            The size of the source in degrees. Use this to ensure the whole source is
            below the horizon. Also, if the local horizon is higher (i.e. mountains),
            this can be set to a negative value to account for this.  You may also
            use this parameter to account for atmospheric refraction, if desired,
            by adding 68 arcminutes to the nominal diameter.

        Returns
        -------
        unix_times : ndarray
            Source rise times as UNIX epoch times.
        """
        return self._sr_work(
            source, t0, t1, step, diameter, skip_rise=True, skip_set=False
        )[0]

    def solar_transit(self, t0, t1=None, step=None, lower=False, return_dec=False):
        """Find the Solar transits between two times.

        Parameters
        ----------
        t0 : ctime.TimeLike
            The start time to search for. Any type that be converted to a UNIX time
            by caput.
        t1 : ctime.TimeLike | None, optional
            The end time of the search interval. If not set, this is 1 day after the
            start time `t0`.
        step : float | None, optional
            The initial search step in days. This is used to find the approximate
            times of transit, and should be set to something less than the spacing
            between events.  If None is passed, an initial search step of 0.2 days,
            or else one fifth of the specified interval, is used, whichever is smaller.
        lower : bool, optional
            By default this only returns the upper (regular) transit. This will cause
            lower transits to be returned instead.
        return_dec : bool, optional
            If set, also return the declination of the source at transit. Default
            is ``False``

        Returns
        -------
        unix_times : ndarray
            Solar transit times as UNIX epoch times.
        """
        return self.transit_times(
            self.skyfield.ephemeris["sun"], t0, t1, step, lower, return_dec
        )

    def lunar_transit(self, t0, t1=None, step=None, lower=False, return_dec=False):
        """Find the Lunar transits between two times.

        Parameters
        ----------
        t0 : ctime.TimeLike
            The start time to search for. Any type that be converted to a UNIX time
            by caput.
        t1 : ctime.TimeLike | None, optional
            The end time of the search interval. If not set, this is 1 day after the
            start time `t0`.
        step : float | None, optional
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
        unix_times : ndarray
            Lunar transit times as UNIX epoch times.
        """
        return self.transit_times(
            self.skyfield.ephemeris["moon"], t0, t1, step, lower, return_dec
        )

    def solar_setting(self, t0, t1=None, step=None):
        """Find the Solar settings between two times.

        This method calculates the conventional astronomical sunset, which
        occurs when the centre of the sun is 50 arcminutes below the horizon.
        This accounts for a solar diameter of 32 arcminutes, plus 34 arcminutes
        of atmospheric refraction at the horizon.

        Parameters
        ----------
        t0 : ctime.TimeLike
            The start time to search for. Any type that be converted to a UNIX time
            by caput.
        t1 : ctime.TimeLike | None, optional
            The end time of the search interval. If not set, this is 1 day after the
            start time `t0`.
        step : float | None, optional
            The initial search step in days. This is used to find the approximate
            times of setting, and should be set to something less than the spacing
            between events.  If None is passed, an initial search step of 0.2 days,
            or else one fifth of the specified interval, is used, whichever is smaller.

        Returns
        -------
        unix_times : ndarray
            Solar setting times as UNIX epoch times.
        """
        return self.set_times(
            self.skyfield.ephemeris["sun"],
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
        t0 : ctime.TimeLike
            The start time to search for. Any type that be converted to a UNIX time
            by caput.
        t1 : ctime.TimeLike | None, optional
            The end time of the search interval. If not set, this is 1 day after the
            start time `t0`.
        step : float | None, optional
            The initial search step in days. This is used to find the approximate
            times of setting, and should be set to something less than the spacing
            between events.  If None is passed, an initial search step of 0.2 days,
            or else one fifth of the specified interval, is used, whichever is smaller.

        Returns
        -------
        unix_times : ndarray
            Lunar setting times as UNIX epoch times.
        """
        return self.set_times(
            self.skyfield.ephemeris["moon"],
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
        t0 : ctime.TimeLike
            The start time to search for. Any type that be converted to a UNIX time
            by caput.
        t1 : ctime.TimeLike | None, optional
            The end time of the search interval. If not set, this is 1 day after the
            start time `t0`.
        step : float | None, optional
            The initial search step in days. This is used to find the approximate
            times of rising, and should be set to something less than the spacing
            between events.  If None is passed, an initial search step of 0.2 days,
            or else one fifth of the specified interval, is used, whichever is smaller.

        Returns
        -------
        unix_times : ndarray
            Solar rising times as UNIX epoch times.
        """
        return self.rise_times(
            self.skyfield.ephemeris["sun"],
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
        t0 : ctime.TimeLike
            The start time to search for. Any type that be converted to a UNIX time
            by caput.
        t1 : ctime.TimeLike | None, optional
            The end time of the search interval. If not set, this is 1 day after the
            start time `t0`.
        step : float | None, optional
            The initial search step in days. This is used to find the approximate
            times of rising, and should be set to something less than the spacing
            between events.  If None is passed, an initial search step of 0.2 days,
            or else one fifth of the specified interval, is used, whichever is smaller.

        Returns
        -------
        unix_times : ndarray
            Lunar rising times as UNIX epoch times.
        """
        return self.rise_times(
            self.skyfield.ephemeris["moon"],
            t0,
            t1,
            step,
            diameter=100.0 / 60,
        )

    def cirs_radec(self, source):
        """Converts a Skyfield body in CIRS coordinates at a given epoch to ICRS.

        Parameters
        ----------
        source : Star
            Skyfield Star object with positions in CIRS coordinates.

        Returns
        -------
        star_icrs : Star
            Skyfield Star object with positions in ICRS coordinates
        """
        from skyfield.functions import to_polar

        ts = self.skyfield.timescale

        epoch = ts.tt_jd(np.median(source.epoch))

        pos = self.skyfield_obs().at(epoch).observe(source)

        # Matrix CT transforms from CIRS to ICRF (https://rhodesmill.org/skyfield/time.html)
        _, dec, ra = to_polar(np.einsum("ij...,j...->i...", epoch.CT, pos.position.au))

        return Star(
            ra=Angle(radians=ra, preference="hours"),
            dec=Angle(radians=dec),
            epoch=epoch,
        )

    def star_cirs(self, ra, dec, epoch):
        """Create a `skyfield.starlib.Star` given the CIRS coordinates of a source.

        Parameters
        ----------
        ra, dec : Angle
            RA and dec of the source in CIRS coordinates
        epoch : skyfield.timelib.Time
            Time of the observation

        Returns
        -------
        icrs_star : Star
            Star object in ICRS coordinates
        """
        return self.cirs_radec(Star(ra=ra, dec=dec, epoch=epoch))

    def object_coords(self, source, date=None, deg=False):
        """Calculate the RA and DEC of a source.

        Gives the ICRS coordinates if no date is given (=J2000), or if a date is
        specified gives the CIRS coordinates at that epoch.

        This also returns the *apparent* position, including abberation and
        deflection by gravitational lensing. This shifts the positions by up to
        20 arcseconds.

        Parameters
        ----------
        source : _SkySourceLike
            skyfield.starlib.Star or skyfield.vectorlib.VectorSum or
            skyfield.jpllib.ChebyshevPosition body representing the source.
        date : float | None, optional
            Determine coordinates at this unix time.  If None, use Jan 01 2000.
        deg : bool, optional
            Return coordinates in degrees if True, radians if False (default).

        Returns
        -------
        coordinates : tuple[float, float]
            Position of the source.
        """
        if date is None:  # No date, get ICRS coords
            if isinstance(source, Star):
                ra, dec = source.ra.radians, source.dec.radians
            else:
                raise ValueError(
                    "Source is not fixed, cannot calculate coordinates without a date."
                )

        else:  # Calculate CIRS position with all corrections
            t = csf.unix_to_skyfield_time(date)
            radec = self.skyfield_obs().at(t).observe(source).apparent().cirs_radec(t)

            ra, dec = radec[0].radians, radec[1].radians

        # If requested, convert to degrees
        if deg:
            ra = np.degrees(ra)
            dec = np.degrees(dec)

        return ra, dec

    def _sr_work(
        self,
        source,
        t0,
        t1=None,
        step=None,
        diameter=100.0 / 60,
        skip_rise=False,
        skip_set=False,
    ):
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
        times = csf.skyfield_time_to_unix(self.skyfield.timescale.tt_jd(times))

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


@overload
def _fixup_interval_and_step(
    t0: ctime.TimeLike, t1: ctime.TimeLike, step
) -> tuple[tuple[ctime.TimeLike, 2], ...]: ...
@overload
def _fixup_interval_and_step(
    t0: npt.ArrayLike[ctime.TimeLike], t1: npt.ArrayLike[ctime.TimeLike], step
) -> tuple[tuple[np.ndarray[ctime.TimeLike], 2], ...]: ...
def _fixup_interval_and_step(t0, t1, step: int | float):
    # Work routine used by Observer._sr_work and transit_times to sanitise the
    # caller-supplied interval and initial step duration.
    #
    # Returns the interval limits converted to TT JD and the computed
    # initial step size

    # Get the ends of the search interval
    t0 = ctime.ensure_unix(t0)
    if t1 is None:
        t1 = t0 + 24 * 3600.0 * constants.stellar_second
    else:
        t1 = ctime.ensure_unix(t1)
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
    t0jd, t1jd = csf.unix_to_skyfield_time([t0, t1]).tt

    return t0jd, t1jd, step


def _solve_all(
    f, x0, x1, dx, skip_increasing=False, skip_decreasing=False, **kwargs: dict
):
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
        Skip roots where the gradient is positive (or negative). Default is False.
    **kwargs : dict
        Passed to `scipy.optimize.brentq`. `xtol`, `rtol` and `maxiter` are probably
        the most useful.

    Returns
    -------
    values : float ndarray
        The values of the roots found.
    root_is_positive : bool ndarray
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

    return np.array(roots, dtype=np.float64), np.array(increasing, dtype=bool)
