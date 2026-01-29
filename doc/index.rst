caput
=====

.. toctree::
   :maxdepth: 1
   :hidden:

   User Guide <self>
   Installation <installation>
   API Reference <autoapi/caput/index.rst>

A collection of utilities for building data analysis pipelines.

Features
--------

.. warning::

  This documentation is under construction.
  The API Reference documents the current state of the code, but other parts of
  the documentation may be out of date or incomplete. Please refer to the API
  Reference for the most accurate information.

- A generic container for holding self-documenting datasets in memory with
  straightforward syncing to h5py_ files (:mod:`~caput.memdata`). Plus some
  specialisation to holding time stream data (:mod:`~caput.containers.tod`).

- Tools to make MPI-parallel analysis a little easier (:mod:`~caput.util.mpitools` and
  :mod:`~caput.mpiarray`).

- Infrastructure for building, managing and configuring pipelines for data
  processing (:mod:`~caput.pipeline` and :mod:`~caput.config`).

- A collection of astronomy-oriented coordinate, ephemeris, and time utilities (:mod:`~caput.astro`).

.. _h5py: https://www.h5py.org/
