"""Configure class attributes using values from a dictionary.

This module to defines strictly typed attributes of a class, that can be loaded
from an input dictionary. This is particularly useful for loading a class from
a YAML document.

Examples
--------
In this example we set up a class to store information about a person.

>>> class Person(Reader):
...
...     name = Property(default='Bill', proptype=str)
...     age = Property(default=26, proptype=float, key='ageinyears')

We then extend it to store information about a person with a pet. The
configuration will be successfully inherited.

>>> class PersonWithPet(Person):
...
...     petname = Property(default='Molly', proptype=str)

Let's create a couple of objects from these classes.

>>> person1 = Person()
>>> person2 = PersonWithPet()

And a dictionary of replacement parameters.

>>> testdict = { 'name' : 'Richard', 'ageinyears' : 40, 'petname' : 'Sooty'}

First let's check what the default parameters are:

>>> print(person1.name, person1.age)
Bill 26.0
>>> print(person2.name, person2.age, person2.petname)
Bill 26.0 Molly

Now let's load the configuration from a dictionary:

>>> person1.read_config(testdict)
>>> person2.read_config(testdict)

Then we'll print the output to see the updated configuration:

>>> print(person1.name, person1.age)
Richard 40.0
>>> print(person2.name, person2.age, person2.petname)
Richard 40.0 Sooty

"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import fileformats

import logging

from yaml.loader import SafeLoader


logger = logging.getLogger(__name__)


class Property:
    """Custom property descriptor that can load values from a given dict."""

    def __init__(self, default=None, proptype=None, key=None):
        """Make a new property type.

        Parameters
        ----------
        default : object
            The initial value for the property.
        proptype : function
            The type of the property. In reality this is just a function which
            gets called whenever we update the value: `val = proptype(newval)`,
            so it can be used for conversion and validation
        key : string
            The name of the dictionary key that we can fetch this value from.
            If None (default), attempt to use the attribute name from the
            class.
        """
        self.proptype = (lambda x: x) if proptype is None else proptype
        self.default = default
        self.key = key
        self.propname = None

    def __get__(self, obj, objtype):
        # Object getter.
        if obj is None:
            return None

        # Ensure the property name has been found and set
        self._set_propname(obj)

        # If the value has not been set, return the default, otherwise return the
        # actual value.
        if self.propname not in obj.__dict__:
            return self.proptype(self.default) if self.default is not None else None
        return obj.__dict__[self.propname]

    def __set__(self, obj, val):
        # Object setter.
        if obj is None:
            return

        # Ensure the property name has been found and set
        self._set_propname(obj)

        # Save the value of this property onto the instance it's a descriptor
        # for.
        obj.__dict__[self.propname] = self.proptype(val)

    def _from_config(self, obj, config):
        """Load the configuration from the supplied dictionary.

        Parameters
        ----------
        obj : object
            The parent object of the Property that we want to update.
        config : dict
            Dictionary of configuration values.

        Raises
        ------
        CaputConfigError
            If there was an error in the config dict.
        """
        self._set_propname(obj)

        if self.key is None:
            self.key = self.propname

        if self.key in config:
            try:
                val = self.proptype(config[self.key])
            except TypeError as e:
                raise CaputConfigError(
                    "Can't read value of '%s' as %s: %s" % (self.key, self.proptype, e),
                    location=config,
                ) from e
            obj.__dict__[self.propname] = val

    def _set_propname(self, obj):
        # As this config.Property instance lives on the class it's in, it
        # doesn't actually know what it's name is. We need to search the class
        # hierarchy for this instance to pull out the name. Once we have it, set
        # it as a local attribute so we have it again.

        import inspect

        if self.propname is None:
            for basecls in inspect.getmro(type(obj))[::-1]:
                for propname, clsprop in basecls.__dict__.items():
                    if clsprop is self:
                        self.propname = propname


class Reader:
    """A class that allows the values of Properties to be assigned from a dictionary."""

    @classmethod
    def from_config(cls, config, *args, **kwargs):
        """Create a new instance with values loaded from config.

        Parameters
        ----------
        config : dict
            Dictionary of configuration values.
        *args : list
            Variable length argument list
        **kwargs : dict
            Arbitrary keyword arguments.
        """
        c = cls(*args, **kwargs)
        c.read_config(config)

        return c

    def read_config(self, config, compare_keys=False, use_defaults=True):
        """Set all properties in this class from the supplied config.

        Parameters
        ----------
        config : dict
            Dictionary of configuration values.
        compare_keys : bool or list[str]
            If True, a CaputConfigError is raised if there are unused keys in the
            config dictionary.
            If a list of strings is given, any unused keys except the ones in the
            list lead to a CaputConfigError.
        use_defaults : bool
            If False, a CaputConfigError is raised if a property is not defined by
            the config dictionary

        Raises
        ------
        CaputConfigError
            If there was an error in the config dict.
        """
        import inspect

        config_keys = list(config.keys())
        prop_keys = []
        for basecls in inspect.getmro(type(self))[::-1]:
            for clsprop in basecls.__dict__.values():
                if isinstance(clsprop, Property):
                    clsprop._from_config(self, config)
                    prop_keys.append(clsprop.key)
        if compare_keys:
            if isinstance(compare_keys, list):
                excluded_keys = set(prop_keys + compare_keys)
            else:
                excluded_keys = set(prop_keys)
            if set(config_keys) - excluded_keys:
                raise CaputConfigError(
                    "Unused configuration keys: [%s]"
                    % ", ".join(set(config_keys) - excluded_keys),
                )
        if not use_defaults:
            if set(prop_keys) - set(config_keys):
                raise CaputConfigError(
                    "Missing configuration keys: [%s]"
                    % ", ".join(set(prop_keys) - set(config_keys)),
                )

        self._finalise_config()

    def _finalise_config(self):
        """Finish up the configuration.

        To be overridden in subclasses if we need to perform some processing
        post configuration.
        """


def utc_time(default=None):
    """Property for representing UTC as UNIX time.

    Parameters
    ----------
    time : `float`, `string` or :class:`~datetime.datetime`
        These are all easy to produce from a YAML file.
    default : `float`, `string` or :class:`~datetime.datetime`, optional
        The optional default time.

    Returns
    -------
    prop : Property
        A property instance setup to parse UTC time.
    """

    def _prop(val):
        # Include import here to get around circular import issues
        # pylint: disable=R0401
        from .time import ensure_unix

        return ensure_unix(val)

    prop = Property(proptype=_prop, default=default)

    return prop


def float_in_range(start, end, default=None):
    """Property type that tests if its input is within the given range.

    Parameters
    ----------
    start, end : float
        Range to test.
    default : `float`, optional
        The optional default time.

    Returns
    -------
    prop : Property
        A property instance setup to validate an input float type.

    Examples
    --------
    Should be used like::

        class Position:

            longitude = config.float_in_range(0.0, 360.0, default=90.0)
    """

    def _prop(val):
        val = float(val)

        if val < start or val > end:
            raise CaputConfigError("Input %f not in range [%f, %f]" % (val, start, end))

        return val

    prop = Property(proptype=_prop, default=default)

    return prop


def enum(options, default=None):
    """Property type that accepts only a set of possible values.

    Parameters
    ----------
    options : list
        List of allowed options.
    default : optional
        The optional default value.

    Returns
    -------
    prop : Property
        A property instance setup to validate an enum type.

    Raises
    ------
    ValueError
        If the default value is not part of the options.

    Examples
    --------
    Should be used like::

        class Project:

            mode = enum(['forward', 'backward'], default='forward')
    """

    def _prop(val):
        if val not in options:
            raise CaputConfigError(f"Input {val} not in {options}")

        return val

    if default is not None and default not in options:
        raise ValueError(f"Default value {default} must be in {options} (or None)")

    prop = Property(proptype=_prop, default=default)

    return prop


def list_type(type_=None, length=None, maxlength=None, default=None):
    """Property type that validates lists against required properties.

    Parameters
    ----------
    type_ : type, optional
        Type to apply. If `None` does not attempt to validate elements against type.
    length : int, optional
        Exact length of the list we expect. If `None` (default) accept any length.
    maxlength : int, optional
        Maximum length of the list. If `None` (default) there is no maximum length.
    default : optional
        The optional default value.

    Returns
    -------
    prop : Property
        A property instance setup to validate the type.

    Raises
    ------
    ValueError
        If the default value fails validation.

    Examples
    --------
    Should be used like::

        class Project:

            mode = list_type(int, length=2, default=[3, 4])
    """

    def _prop(val):
        if not isinstance(val, (list, tuple)):
            raise CaputConfigError(
                "Expected to receive a list, but got '%s.'" % repr(val)
            )

        if type_:
            for ii, item in enumerate(val):
                if not isinstance(item, type_):
                    raise CaputConfigError(
                        "Expected to receive a list with items of type %s, but got "
                        "'%s' of type '%s' at position %i"
                        % (type_, item, type(item), ii)
                    )

        if length and len(val) != length:
            raise CaputConfigError(
                "List expected to be of length %i, but was actually length %i"
                % (length, len(val))
            )

        if maxlength and len(val) > maxlength:
            raise CaputConfigError(
                "Maximum length of list is %i is, but list was actually length %i"
                % (maxlength, len(val))
            )

        return val

    if default:
        try:
            _prop(default)
        except CaputConfigError as e:
            raise ValueError(
                "Default value %s does not satisfy property requirements: %s"
                % (default, repr(e))
            ) from e

    prop = Property(proptype=_prop, default=default)

    return prop


def logging_config(default=None):
    """Property type that validates the caput logging config.

    Allows the type to be either a string (for backward compatibility) or a dict
    setting log levels per module.

    Parameters
    ----------
    default : optional
        The optional default value.

    Returns
    -------
    prop : Property
        A property instance setup to validate the type.

    Examples
    --------
    Should be used like::

        class Project:

            loglevels = logging_config({"root": "INFO", "annoying.module": "WARNING"})
    """
    if default is None:
        default = {}

    def _prop(config):
        if isinstance(config, str):
            config = {"root": config}
        elif not isinstance(config, dict):
            raise ValueError(
                f"Expected a string or YAML block for config value 'logging', got "
                f"'{type(config.__name__)}'."
            )

        # check entries, get module names and warn for duplicates when sorting into new
        # dict
        checked_config = {}
        loglevels = ["DEBUG", "INFO", "WARNING", "ERROR", "NOTSET"]
        for key, level in config.items():
            level = level.upper()
            if level not in loglevels:
                raise ValueError(
                    f"Expected one of {loglevels} for log level of {key} (was {level})."
                )

            already_set_to = checked_config.get(key, None)
            if already_set_to is not None and already_set_to != level:
                logger.warning(
                    f"Setting log level for {key} to {level}, but is already set to "
                    f"{already_set_to}. The old value will get ignored."
                )
            checked_config[key] = level
        return checked_config

    prop = Property(proptype=_prop, default=default)

    return prop


def file_format(default: str | fileformats.FileFormat | None = None) -> Property:
    """Property type that accepts only "zarr", or "hdf5".

    Returns the selected `caput.fileformat.FileFormat` subclass or `caput.fileformats.HDF5` if `value == default`.

    Parameters
    ----------
    default
        A string or type object specifying the fileformat

    Returns
    -------
    prop
        A property instance setup to validate a file format.

    Raises
    ------
    ValueError
        If the default value is not `"hdf5"` or `"zarr"`.

    Examples
    --------
    Should be used like::

        class Project:

            mode = file_format(default='zarr')
    """
    options = ("hdf5", "zarr")

    def _prop(val):
        from . import fileformats

        if val is None:
            return None

        if issubclass(val, fileformats.FileFormat):
            return val

        if not isinstance(val, str):
            raise CaputConfigError(
                f"Input {repr(val)} is of type {type(val).__name__} (expected str or None)."
            )

        val = val.lower()

        if val == "hdf5":
            return fileformats.HDF5
        if val == "zarr":
            return fileformats.Zarr

        raise CaputConfigError(f"Input {repr(val)} needs to be one of {options})")

    if default is not None and (
        (not isinstance(default, str)) or (default.lower() not in options)
    ):
        raise CaputConfigError(f"Default value {repr(default)} must be in {options}")

    return Property(proptype=_prop, default=default)


class _line_dict(dict):
    """A private dict subclass that also stores line numbers for debugging."""

    __line__ = None


class SafeLineLoader(SafeLoader):
    """YAML loader that tracks line numbers.

    Adds the line number information to every YAML block. This is useful for
    debugging and to describe linting errors.
    """

    def construct_mapping(self, node, deep=False):
        """Construct the line mapping."""
        mapping = super().construct_mapping(node, deep=deep)
        mapping = _line_dict(mapping)

        # Add 1 so numbering starts at 1
        mapping.__line__ = node.start_mark.line + 1
        return mapping


class CaputConfigError(Exception):
    """There was an error in the configuration.

    Parameters
    ----------
    message : str
        Message / description of error
    file_ : str
        Configuration file name (optional)
    location : dict
        If using :class:`SafeLineLoader` is used, a dict created by that can be
        passed in here to report the line number where the error occurred.
    """

    def __init__(self, message, file_=None, location=None):
        self.message = message
        self.file = file_
        if isinstance(location, _line_dict):
            self.line = location.__line__
        else:
            self.line = None
        super().__init__(message)

    def __str__(self):
        location = ""
        if self.line is not None:
            location = "\nError in block starting at L{}".format(self.line)
            if self.file is not None:
                location = "{} ({})".format(location, self.file)
        return "{}{}".format(self.message, location)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
