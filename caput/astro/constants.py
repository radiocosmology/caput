"""A set of useful constants and conversions in Astronomy and Cosmology.

Most constants are just imported directly from `scipy.constants`.
A few other units are defined here, as well as some aliases.
"""

from scipy.constants import *

## Include a handful of useful units which are not
## included in `scipy.constants`.
#: Solar masses in kg
solar_mass = 1.98892e30
#: One second in seconds
second = 1.0
#: One sidereal day in seconds
t_sidereal = 23.9344696 * hour
#: Radiation constant (in J m^{-3} K^{-4})
a_rad = 4 * Stefan_Boltzmann / c
#: 21cm transition frequency (in MHz)
nu21 = 1420.40575177
# The approximate length of a UT1 second in SI seconds (i.e. LOD / 86400). This was
# calculated from the IERS EOP C01 IAU2000 data, by calculating the derivative of UT1 -
# TAI from 2019.5 to 2020.5. Note that the variations in this are quite substantial,
# but it's typically 1ms over the course of a day
UT1_second = 1.00000000205
# Approximate number of seconds in a sidereal second.
# The exact value used here is from https://hpiers.obspm.fr/eop-pc/models/constants.html
# but can be derived from USNO Circular 179 Equation 2.12
sidereal_second = 1.0 / 1.002737909350795 * UT1_second
# Approximate length of a stellar second
# This comes from the definition of ERA-UT1 (see IERS Conventions TR Chapter 1) giving
# the first ratio a UT1 and stellar second
stellar_second = 1.0 / 1.00273781191135448 * UT1_second

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
