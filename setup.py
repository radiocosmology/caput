from setuptools import setup, find_packages


setup(
    name = 'caput',
    version = 0.1,

    packages = find_packages(),
    scripts=['scripts/caput-pipeline'],
    requires = ['numpy', 'h5py'],  # Probably should change this.

    # metadata for upload to PyPI
    author = "Kiyo Masui, J. Richard Shaw",
    author_email = "kiyo@physics.ubc.ca",
    description = "Cluster Astronomical Python Utilities.",
    license = "GPL v3.0",
    url = "http://github.com/kiyo-masui/caput"
)
