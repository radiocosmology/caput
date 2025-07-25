"""Routines for dealing with Skyfield data and conversions.

http://rhodesmill.org/skyfield/


Time Utilities
==============
Time conversion routine which are location independent.

- :py:meth:`skyfield_time_to_unix`
- :py:meth:`unix_to_skyfield_time`


Skyfield Interface
==================

- :py:class:`SkyfieldWrapper`

This module provides an interface to Skyfield which stores the required datasets
(timescale data and an ephemeris) in a fixed location. The location is
determined by the following (in order):

- As the wrapper is initialised by passing in a ``path=<path>`` option.
- By setting the environment variable ``CAPUT_SKYFIELD_PATH``
- If neither of the above is set, the data is placed in ``<path to caput>/caput/astro/_skyfield_data/``

Other skyfield helper functions:

- :py:meth:`skyfield_star_from_ra_dec`
"""

__all__ = [
    "SkyfieldWrapper",
    "skyfield_star_from_ra_dec",
    "skyfield_time_to_unix",
    "unix_to_skyfield_time",
]

import warnings

from skyfield import timelib
from skyfield.starlib import Star
from skyfield.units import Angle

from .. import darray


class SkyfieldWrapper:
    """A wrapper to help with loading Skyfield and its data.

    Parameters
    ----------
    path : string, optional
        Directory Skyfield should save data in. If not set data will be looked
        for in `$CAPUT_SKYFIELD_PATH` or in `<path to caput>/caput/astro/_skyfield_data`.
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
                path = os.path.join(os.path.dirname(__file__), "_skyfield_data", "")

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


@darray.scalarize()
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
    from .time import ensure_unix

    # TODO: I'm surprised there isn't a better way to do this. Needing to convert via a
    # datetime isn't great, but the only other ways I think can do it use private
    # routines
    return ensure_unix(skyfield_time.utc_datetime())


@darray.scalarize()
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
