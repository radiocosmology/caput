# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility

from future.utils import bytes_to_native_str

import numpy
import os
from Cython.Build import cythonize
from setuptools import setup, find_packages
from setuptools.extension import Extension

from caput import __version__


REQUIRES = ['numpy', 'h5py', 'PyYAML']

# Don't install requirements if on ReadTheDocs build system.
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    requires = []
else:
    requires = REQUIRES


# Try and install Skyfield data
try:
    from caput import time as ctime

    # Force download of data
    sf = ctime.SkyfieldWrapper(
        ephemeris='http://e-mode.phas.ubc.ca/~jrs65/de421.bsp'
    )
    sf.reload()

    # Set package data to be installed alongside skyfield
    skyfield_data = {
        # TODO: Py3 remove this hack needed to work around a setuptools bug
        bytes_to_native_str(b'caput'): [
            'data/Leap_Second.dat',
            'data/de421.bsp',
            'data/deltat.data',
            'data/deltat.preds'
        ]
    }

except:
    import warnings
    warnings.warn("Could not install additional Skyfield data.")
    skyfield_data = {}

# Cython
extensions = [
    Extension(
        name=bytes_to_native_str(b'caput.weighted_median'),
        sources=[bytes_to_native_str(b'caput/weighted_median.pyx')],
        include_dirs=[numpy.get_include()],
        language="c++",
        extra_compile_args=['-std=c++11','-fopenmp','-g0','-O3',],
        extra_link_args=["-std=c++11","-fopenmp"],
    ),
]

setup(
    name='caput',
    version=__version__,
    packages = find_packages(),
    scripts=['scripts/caput-pipeline'],
    install_requires=requires,
    extras_require={
        'mpi': ['mpi4py>=1.3']
    },
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
