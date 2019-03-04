# Change Log

All notable changes to this project will be documented in this file. This
project adheres to [Semantic Versioning](http://semver.org/), with the exception
that I'm using PEP440 to denote pre-releases.


## Dev

### Added

- Context Manager for all memh5 container types (#60). You can now do:
```python
with memh5.BasicCont.from_file("filename.h5"):
   pass
```
- Support for new CHIME stacked data with `reverse_map` (#58)
- Better handling of missing config information (#56)

### Fixed

- Fixed circular references in memh5 containers (#60)
- Fixed a race condition when creating output directories (#64)
- Fixed bug in `tod.concatenate`.


## [0.6.0] - 2019-01-18

### Changed

- Python 3 support
- `reverse_map` support for containers.

### Fixed

- Made test running more robust by downloading a self-hosted copy of the
  ephemeris.


## [0.4.1] - 2017-07-18

### Fixed

- A bug when using a new version of skyfield.


## [0.4.0] - 2017-06-24

### Added

- Added an enum type for the `caput.config`


## [0.3.0] - 2016-08-13

### Added

- This `CHANGELOG` file.
- A new module (`caput.time`) for converting between various time
  representations (UNIX, `datetime`, and Skyfield format), and for calculating
  celestial times, including Earth Rotation Angle, Local Stellar Angle, and
  Local Stellar Day.
- Added some helper routines for creating specific `Property` attributes to
  `caput.config`.
