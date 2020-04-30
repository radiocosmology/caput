caput
=====

.. currentmodule:: caput

A collection of utilities for building data analysis pipelines.

Features
--------

- A generic container for holding self-documenting datasets in memory with
  straightforward syncing to h5py_ files (:mod:`~caput.memh5`). Plus some
  specialisation to holding time stream data (:mod:`~caput.tod`).

- Tools to make MPI-parallel analysis a little easier (:mod:`caput.mpiutil` and
  :mod:`~caput.mpiarray`).

- Infrastructure for building, managing and configuring pipelines for data
  processing (:mod:`~caput.pipeline` and :mod:`~caput.config`).

- Routines for converting to between different time representations, dealing
  with leap seconds, and calculating celestial times (:mod:`~caput.time`)


Installation
------------

::

    pip install git+https://github.com/radiocosmology/caput.git

caput depends on h5py_, numpy_ and PyYAML_. For full functionality it also
requires click_, mpi4py_ and Skyfield_.


Configuration
_____________

The caput pipeline runner script accepts a YAML file for configuration. The structure of this file
is documented in :ref:`config`.


Index
-----

.. toctree::
   :maxdepth: 2

   config


Modules
-------

.. autosummary::
    :toctree: generated/

   memh5
   config
   pipeline
   misc
   mpiutil
   mpiarray
   time
   tod

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _GitHub: https://github.com/KeepSafe/aiohttp
.. _h5py: http:/www.h5py.org/
.. _numpy: http://www.numpy.org/
.. _PyYAML: http://pyyaml.org/
.. _mpi4py: http://mpi4py.readthedocs.io/en/stable/
.. _click: http://click.palletsprojects.com/
.. _Skyfield: http://rhodesmill.org/skyfield/
.. _Freenode: http://freenode.net
