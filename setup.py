from setuptools import setup, find_packages
import os


REQUIRES = ['numpy', 'h5py', 'mpi4py', 'PyYAML']

# Don't install requirements if on ReadTheDocs build system.
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    requires = []
else:
    requires = REQUIRES

setup(
    name = 'caput',
    version = 0.1,
    packages = ['caput', 'caput.tests'],
    scripts=['scripts/caput-pipeline'],
    install_requires=requires,

    # metadata for upload to PyPI
    author = "Kiyo Masui, J. Richard Shaw",
    author_email = "kiyo@physics.ubc.ca",
    description = "Cluster Astronomical Python Utilities.",
    license = "GPL v3.0",
    url = "http://github.com/radiocosmology/caput"
)
