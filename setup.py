from setuptools import setup
import os

from caput import __version__


REQUIRES = ['numpy', 'h5py', 'PyYAML']

# Don't install requirements if on ReadTheDocs build system.
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    requires = []
else:
    requires = REQUIRES

setup(
    name='caput',
    version=__version__,
    packages=['caput', 'caput.tests'],
    scripts=['scripts/caput-pipeline'],
    install_requires=requires,
    extras_require={
        'mpi': ['mpi4py>=1.3'],
        'skyfield': ['skyfield>=1.0']
    },

    # metadata for upload to PyPI
    author="Kiyo Masui, J. Richard Shaw",
    author_email="kiyo@physics.ubc.ca",
    description="Cluster Astronomical Python Utilities.",
    license="GPL v3.0",
    url="http://github.com/radiocosmology/caput"
)
