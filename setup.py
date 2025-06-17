"""Build cython extensions.

The full project config can be found in `pyproject.toml`. `setup.py` is still
required to build cython extensions.
"""

import os
import re
import sys
import sysconfig

import numpy
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

# Decide whether to use OpenMP or not
if (
    ("CAPUT_NO_OPENMP" in os.environ)
    or (re.search("gcc", sysconfig.get_config_var("CC")) is None)
    or (sys.platform == "darwin")
):
    print("Not using OpenMP")
    omp_args = []
else:
    omp_args = ["-fopenmp"]

# Set up project extensions
extensions = [
    Extension(
        name="caput.median.weighted",
        sources=["caput/median/weighted.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++",
        extra_compile_args=[*omp_args, "-std=c++11", "-g0", "-O3"],
        extra_link_args=[*omp_args, "-std=c++11"],
    ),
    Extension(
        name="caput.truncate",
        sources=["caput/truncate.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=[*omp_args, "-g0", "-O3"],
        extra_link_args=omp_args,
    ),
    Extension(
        name="caput.coordinates.coord",
        sources=["caput/coordinates/coord.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=[*omp_args, "-g0", "-O3"],
        extra_link_args=omp_args,
    ),
    Extension(
        name="caput._fast_tools",
        sources=["caput/_fast_tools.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=[*omp_args, "-g0", "-O3"],
        extra_link_args=omp_args,
    ),
]

setup(
    name="caput",  # required
    ext_modules=cythonize(extensions),
)
