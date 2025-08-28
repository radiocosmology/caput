r"""Constants and conversions for astronomy and cosmology.

Re-exports all constants in `scipy.constants`_, along with a few extras.

Dynamic prefixes are also supported, using all prefixes defined
in `scipy.constants`. Prefixes are separated from the
unit by an underscore.

Examples
--------
Prefixes are automatically multiplied. In this example, the `kilo`
prefix multiplies the base unit `gram`.

>>> from caput.astro import constants
>>> constants.kilo_gram
1.0

.. _`scipy.constants`: https://docs.scipy.org/doc/scipy/reference/constants.html
"""

from scipy.constants import *

## Include a handful of useful units which are not
## included in `scipy.constants`.

solar_mass: float = 1.98892e30
"""Solar mass in kg."""

second: float = 1.0
"""One second in seconds."""

t_sidereal: float = 23.9344696 * hour
"""One sidereal day in seconds, equal to `23.9344696 * hour`."""

a_rad: float = 4.0 * Stefan_Boltzmann / c
r"""Radiation constant (in J m\ :sup:`-3` K\ :sup:`-4}`, equal to `4 * Stefan_Boltzmann / c`."""

nu21: float = 1420.40575177
"""21cm transition frequency (in MHz)."""

UT1_second: float = 1.00000000205
"""The approximate length of a UT1 second in SI seconds (i.e. LOD / 86400). This was
calculated from the IERS EOP C01 IAU2000 data, by calculating the derivative of UT1 -
TAI from 2019.5 to 2020.5. Note that the variations in this are quite substantial,
but it's typically 1ms over the course of a day."""

sidereal_second: float = 1.0 / 1.002737909350795 * UT1_second
"""Approximate number of seconds in a sidereal second.
The exact value used here is from https://hpiers.obspm.fr/eop-pc/models/constants.html
but can be derived from USNO Circular 179 Equation 2.12."""

stellar_second: float = 1.0 / 1.00273781191135448 * UT1_second
"""Approximate length of a stellar second.
This comes from the definition of ERA-UT1 (see IERS Conventions TR Chapter 1) giving
the first ratio a UT1 and stellar second."""

# Aliases. Included in this dict in case we ever want
# to change/remove them
__aliases = {
    "stefan_boltzmann": Stefan_Boltzmann,
    "k_B": Boltzmann,
    "arc_minute": arcminute,
    "arc_second": arcsecond,
    "UT1_S": UT1_second,
    "SIDEREAL_S": sidereal_second,
    "STELLAR_S": stellar_second,
}


def __split_prefix(name):
    names = [value for value in name.split("_") if value]

    if len(names) > 2:
        raise AttributeError(f"Only single-prefix values are allowed. Got `{name}`.")

    return names


def __getattr__(name):
    if name in globals():
        return globals()[name]
    if name in __aliases:
        return __aliases[name]
    # If this value hasn't been defined, see if it's
    # a combination of a prefix and a unit
    if len(names := __split_prefix(name)) == 2:
        return __getattr__(names[0]) * __getattr__(names[1])

    raise AttributeError(f"Cannot find constant `{name}`.")
