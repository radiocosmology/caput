# [20.5.1](https://github.com/radiocosmology/caput/compare/v20.5.0...v20.5.1) (2020-05-07)


### Bug Fixes

* **memh5:** json serialize numpy ndarray ([e8156da](https://github.com/radiocosmology/caput/commit/e8156da8d49dc9534378f845c0f36dd160ea7d3f))



# [20.5.0](https://github.com/radiocosmology/caput/compare/v0.6.0...v20.5.0) (2020-05-05)

Note: we have switched to calendar versioning for this release. 
 

### Bug Fixes

* Fixed circular references in memh5 containers (#60)
* Fixed a race condition when creating output directories (#64)
* Fixed bug in `tod.concatenate`
* **memh5:** explicitly set h5py read mode in routines and tests ([e5c0016](https://github.com/radiocosmology/caput/commit/e5c0016214d60a663f1ef37d9c7761ff0ee39bc9))
* **memh5:** no comm attribute on group in redistribute ([be655b0](https://github.com/radiocosmology/caput/commit/be655b0f5498de5c07ee11b2c941717f1a854ebf))
* **memh5:** allow writing unicode datasets into h5py ([25c5b2d](https://github.com/radiocosmology/caput/commit/25c5b2d952d13300081e31cc9e7830471fe2822c))
* **memh5:** change default `convert_attribute_strings` for backwards compatibility ([bc8261b](https://github.com/radiocosmology/caput/commit/bc8261bed1078355fdab031bdd527521ec58338c))
* **memh5:** comm property was set when distributed=False ([a9acd90](https://github.com/radiocosmology/caput/commit/a9acd90f43289c32aaab52129004d0fc41c88410))
* **memh5:** detect and convert types annotated with metadata ([ef4ebc8](https://github.com/radiocosmology/caput/commit/ef4ebc88f38293c6bec9a46aa2643af62651adef))
* **memh5:** Mapping has moved into collections.abc in Python 3.8 ([631412e](https://github.com/radiocosmology/caput/commit/631412efdc143d45490697b574a1c6d3f74a2d4b))
* **memh5:** serialise datetimes with dict attributes without crashing ([5d30194](https://github.com/radiocosmology/caput/commit/5d301946d06ba3d4d4477f17478d6ed6426cd0a7))
* **memh5:** sort items when copying to/from HDF5 to eliminate MPI errors ([7af91fe](https://github.com/radiocosmology/caput/commit/7af91fee9c44db327f1ba2ad6df961c52c440af2))
* **mpiarray:** partition parallel IO to be under the 2GB HDF5 limit ([bee591e](https://github.com/radiocosmology/caput/commit/bee591e9bf6e9e195ec1467fd4423b4c03feb3ee))
* **mpiarray:** workaround for h5py collective IO bug ([9c68fe3](https://github.com/radiocosmology/caput/commit/9c68fe38ca08d0bf95db77e772db5afd32a3a7af)), closes [#965](https://github.com/radiocosmology/caput/issues/965)
* **mpiutil:** disable the MPI excepthook ([6fba9f6](https://github.com/radiocosmology/caput/commit/6fba9f66af72c19d4bc8afb1b716431e9b46ea59))
* **parallel_map:** fixed crash ([2462a2f](https://github.com/radiocosmology/caput/commit/2462a2f75449f0a4fba398b8922c7814ccd04706))
* **pipeline:** changed incorrect pro_argspec.keywords to pro_argspec.varkw ([906d753](https://github.com/radiocosmology/caput/commit/906d753c6f70fc1db6d6232938a1991f57313dec)), closes [#121](https://github.com/radiocosmology/caput/issues/121)
* **pipeline:** use safe_load to avoid warning when loading pipeline conf ([8f2d7c3](https://github.com/radiocosmology/caput/commit/8f2d7c3af579f6215e72eb581ed682eec8eadacc))
* **runner:** Add call to group at end of file. ([e48ef60](https://github.com/radiocosmology/caput/commit/e48ef6051469057dc1684e910c1ec00dda438e07))
* **runner:** use copyfile instead of copy ([3da9976](https://github.com/radiocosmology/caput/commit/3da997613b94349d2afee2bfb42a81d30d1af9f4)), closes [#102](https://github.com/radiocosmology/caput/issues/102)
* **time:** include required skyfield data in the repo (via Git LFS) ([39f357d](https://github.com/radiocosmology/caput/commit/39f357d281c907391872f66426830b054605a56a))
* **time:** set SkyfieldWrapper default expire=False ([#126](https://github.com/radiocosmology/caput/issues/126)) ([8e84719](https://github.com/radiocosmology/caput/commit/8e8471982139bb8aacf27efadeef4127b53d4cda))
* **tod:** fixes bug where `data` variable overwritten ([0ed8b2c](https://github.com/radiocosmology/caput/commit/0ed8b2cf9bd534b7c61633cfa2255fc14900ade5))


### Features

* Context Manager for all memh5 container types (#60). You can now do:
```python
with memh5.BasicCont.from_file("filename.h5"):
   pass
```
* Support for new CHIME stacked data with `reverse_map` (#58)
* Better handling of missing config information (#56)
* IO for distributed `memh5` containers is now MPI parallel if possible (#66). This should make a
big difference to write times of large containers.
* **config:** add a list Property for validating input lists ([b32ad1d](https://github.com/radiocosmology/caput/commit/b32ad1d238d25a5031eb3035572abd48bbfd4677))
* **BasicCont:** make history an h5 attribute ([fbc5034](https://github.com/radiocosmology/caput/commit/fbc50340907e321cba9682ff6dc47c534f82e122))
* **caput-pipeline:** add option to run job from a temporary dir ([90043f5](https://github.com/radiocosmology/caput/commit/90043f594851713575d778f768baba5ce100187b))
* **Manager:** add metadata options ([47432ee](https://github.com/radiocosmology/caput/commit/47432ee30a43dc8d50d8059f608c7d050e7e4c37)), closes [#104](https://github.com/radiocosmology/caput/issues/104)
* **memh5:** add compression and chunking of memh5 datasets ([c0886c8](https://github.com/radiocosmology/caput/commit/c0886c850cfc80663017a22bafb86d0506cbb915))
* **memh5:** sel_* parameters ([6743098](https://github.com/radiocosmology/caput/commit/67430981a938c05157eaeef2e233532de77715f1)), closes [#108](https://github.com/radiocosmology/caput/issues/108)
* **misc:** context manager for lockfiles ([af67279](https://github.com/radiocosmology/caput/commit/af6727951aa5f13845978d58559bae94d21325ce))
* **MPIArray:** add gather and allgather for collecting the full array ([2da95ae](https://github.com/radiocosmology/caput/commit/2da95ae997f2e3f13a7699ef5323830da5dd8f76))
* **pipeline:** improve log level config ([f418299](https://github.com/radiocosmology/caput/commit/f4182995c252a04b99f3168d6bbec88e12a9d7fd))
* **pipeline:** use locking when writing output ([2ab01ed](https://github.com/radiocosmology/caput/commit/2ab01edce0864c96486eb59b6c1803929034992c))
* **setup.py:** support building without OpenMP ([e478a03](https://github.com/radiocosmology/caput/commit/e478a03d221d4ec71d5f98bc9331d865005d422c))
* **tod:** allow control of string conversion in `tod.concatenate` ([249face](https://github.com/radiocosmology/caput/commit/249face8d4e0ab38e91507c0724606687d8515c8))
* **util:** moving weighted median ([f3e1a2c](https://github.com/radiocosmology/caput/commit/f3e1a2c8c4cbb656a642dc1a304f356f5c7ee311))
* **versioneer:** add versioneer for better version naming ([fa9cccb](https://github.com/radiocosmology/caput/commit/fa9cccbc7443e82cb63a478be908d6923b35e3f1))



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
