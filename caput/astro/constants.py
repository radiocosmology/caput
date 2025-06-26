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


__aliases = {
    "stefan_boltzmann": Stefan_Boltzmann,
    "k_B": Boltzmann,
    "arc_minute": arcminute,
    "arc_second": arcsecond,
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
