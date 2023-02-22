import os
import re
import sysconfig

import numpy
from Cython.Build import cythonize
from setuptools import setup, find_packages
from setuptools.extension import Extension

import versioneer


# Don't install requirements if on ReadTheDocs build system.
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if on_rtd:
    requires = []
else:
    # Load the PEP508 formatted requirements from the requirements.txt file. Needs
    # pip version > 19.0
    with open("requirements.txt", "r") as fh:
        requires = fh.readlines()


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
        name="caput.weighted_median",
        sources=["caput/weighted_median.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++",
        extra_compile_args=(omp_args + ["-std=c++11", "-g0", "-O3"]),
        extra_link_args=(omp_args + ["-std=c++11"]),
    ),
    Extension(
        name="caput.truncate",
        sources=["caput/truncate.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=(omp_args + ["-g0", "-O3"]),
        extra_link_args=omp_args,
    ),
    Extension(
        name="caput._fast_tools",
        sources=["caput/_fast_tools.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=(omp_args + ["-g0", "-O3"]),
        extra_link_args=omp_args,
    ),
]


setup(
    name="caput",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    entry_points="""
        [console_scripts]
        caput-pipeline=caput.scripts.runner:cli
    """,
    python_requires=">=3.8",
    install_requires=requires,
    extras_require={
        "mpi": ["mpi4py>=1.3"],
        "compression": [
            "bitshuffle",
            "numcodecs>=0.7.3",
            "zarr>=2.11.0",
        ],
        "profiling": ["psutil", "pyinstrument"],
    },
    setup_requires=["cython"],
    # metadata for upload to PyPI
    author="Kiyo Masui, J. Richard Shaw",
    author_email="kiyo@physics.ubc.ca",
    description="Cluster Astronomical Python Utilities.",
    license="GPL v3.0",
    url="http://github.com/radiocosmology/caput",
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
)
