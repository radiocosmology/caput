from setuptools import setup, find_packages


setup(
    name = 'caput',
    version = 0.1,
    packages = ['caput', 'caput.tests'],
    scripts=['scripts/caput-pipeline'],
    install_requires = ['numpy', 'h5py', 'mpi4py'],

    # metadata for upload to PyPI
    author = "Kiyo Masui, J. Richard Shaw",
    author_email = "kiyo@physics.ubc.ca",
    description = "Cluster Astronomical Python Utilities.",
    license = "GPL v3.0",
    url = "http://github.com/radiocosmology/caput"
)
