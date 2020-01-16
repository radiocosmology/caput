import os

from skyfield import api

data_path = os.path.dirname(__file__)

load = api.Loader(data_path if data_path else ".")

# Load the timescale files
load.timescale()

# Load the ephemeris
load("de421.bsp")
