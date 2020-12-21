caput
=====

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


Index
-----

.. toctree::
   :maxdepth: 2

   installation
   config
   reference


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _h5py: http:/www.h5py.org/
