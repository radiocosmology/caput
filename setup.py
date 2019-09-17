# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

from future.utils import bytes_to_native_str

import os
import re
import sysconfig

import numpy
from Cython.Build import cythonize
from setuptools import setup, find_packages
from setuptools.extension import Extension

from caput import __version__


REQUIRES = ["numpy>=1.16", "h5py", "PyYAML", "cython", "future"]

# Don't install requirements if on ReadTheDocs build system.
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if on_rtd:
    requires = []
else:
    requires = REQUIRES


# Try and install Skyfield data
try:
    from caput import time as ctime

    # Force download of data
    sf = ctime.SkyfieldWrapper(ephemeris="http://e-mode.phas.ubc.ca/~jrs65/de421.bsp")
    sf.reload()

    # Set package data to be installed alongside skyfield
    skyfield_data = {
        # TODO: Py3 remove this hack needed to work around a setuptools bug
        bytes_to_native_str(b"caput"): [
            "data/Leap_Second.dat",
            "data/de421.bsp",
            "data/deltat.data",
            "data/deltat.preds",
        ]
    }
    print("Successfully cached skyfield data.")

except Exception as e:
    import warnings

    warnings.warn("Could not install additional Skyfield data: %s" % str(e))
    skyfield_data = {}

# Cython
# Decide whether to use OpenMP or not
if ("CAPUT_NO_OPENMP" in os.environ) or (
    re.search("gcc", sysconfig.get_config_var("CC")) is None
):
    print("Not using OpenMP")
    omp_args = []
else:
    omp_args = ["-fopenmp"]

extensions = [
    Extension(
        name=bytes_to_native_str(b"caput.weighted_median"),
        sources=[bytes_to_native_str(b"caput/weighted_median.pyx")],
        include_dirs=[numpy.get_include()],
        language="c++",
        extra_compile_args=(omp_args + ["-std=c++11", "-g0", "-O3"]),
        extra_link_args=(omp_args + ["-std=c++11"]),
    )
]

setup(
    name="caput",
    version=__version__,
    packages=find_packages(),
    scripts=["scripts/caput-pipeline"],
    install_requires=requires,
    extras_require={"mpi": ["mpi4py>=1.3"]},
    setup_requires=["cython"],
    package_data=skyfield_data,
    # metadata for upload to PyPI
    author="Kiyo Masui, J. Richard Shaw",
    author_email="kiyo@physics.ubc.ca",
    description="Cluster Astronomical Python Utilities.",
    license="GPL v3.0",
    url="http://github.com/radiocosmology/caput",
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
)
