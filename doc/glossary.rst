:orphan:

Glossary
--------

.. glossary::

	attributes_like
		An arbitrary representation of an object holding dataset attributes. Typically, this
		is a `dict`.

	dataset_like
		Something that appears like a `h5py`_ dataset.

	file_format
		A subclass of :py:class:`~caput.memdata.fileformats.FileFormat`, used to represent an
		on-disk file format.

	file_like
		Somthing that appears like a `h5py`_ File object, or a path that can be loaded as
		a file.

	file_or_group_like
		Something that looks like either a File or a Group.		


	group_like
		Something that appears like a `h5py`_ Group.

	selection_tuple
		A tuple containing any type that can be used to index a `ndarray`.

	selection_like
		Anythin that can be used to index a `ndarray`.

	sky_source_like
		Any representation of an astronomical source.

	time_like
		Any representation of time.


.. _h5py: https://h5py.org/
