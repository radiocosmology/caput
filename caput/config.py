"""Configure class attributes using values from a dictionary.

This module to defines strictly typed attributes of a class that can be loaded
from an input dictionary. This is particularly useful for loading a class from
a YAML document.

Examples
--------
In this example we set up a class to store information about a person.

>>> class Person(Reader):
...     name = Property(default="Bill", proptype=str)
...     age = Property(default=26, proptype=float, key="ageinyears")

We then extend it to store information about a person with a pet. The
configuration will be successfully inherited.

>>> class PersonWithPet(Person):
...     petname = Property(default="Molly", proptype=str)

Let's create a couple of objects from these classes.

>>> person1 = Person()
>>> person2 = PersonWithPet()

And a dictionary of replacement parameters.

>>> testdict = {"name": "Richard", "ageinyears": 40, "petname": "Sooty"}

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

import logging
from typing import TYPE_CHECKING

from yaml.loader import SafeLoader

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Any

    import numpy.typing as npt
    from yaml.loader import MappingNode

    from .astro.time import TimeLike

logger = logging.getLogger(__name__)


class Property:
    """Custom property descriptor that can load values from a given dict.

    Parameters
    ----------
    default : Any | None, optional
        The initial value for the property.
    proptype : Callable | None, optional
        The type of the property. This is a function which gets called whenever
        we update the value: ``val = proptype(newval)``, so it can be used for
        conversion and validation. This convention also means that we can use the
        standard type functions as prototypes.
    key : str | None, optional
        The name of the dictionary key that we can fetch this value from.
        If None (default), attempt to use the attribute name from the
        class.
    deprecated : bool, optional
        Flag a property as deprecated. This will raise a deprecation warning when
        first accessed, but will not change the behaviour of this `Property` instance.
    """

    def __init__(
        self,
        default: Any | None = None,
        proptype: Callable | None = None,
        key: str | None = None,
        deprecated: bool = False,
    ) -> None:
        """Make a new property type."""
        self.proptype = (lambda x: x) if proptype is None else proptype
        self.default = default
        self.key = key
        self.propname = None
        self._deprecated = deprecated

    def __get__(self, obj: Any, objtype: Any) -> Any | None:
        # Object getter.
        if obj is None:
            return None

        # Ensure the property name has been found and set,
        # and warn if it's been deprecated
        self._set_propname_warn_deprecated(obj)

        # If the value has not been set, return the default, otherwise return the
        # actual value.
        if self.propname not in obj.__dict__:
            return self.proptype(self.default) if self.default is not None else None

        return obj.__dict__[self.propname]

    def __set__(self, obj: Any, val: Any) -> None:
        # Object setter.
        if obj is None:
            return

        # Ensure the property name has been found and set,
        # and warn if it's been deprecated
        self._set_propname_warn_deprecated(obj)

        # Save the value of this property onto the instance it's a descriptor
        # for.
        obj.__dict__[self.propname] = self.proptype(val)

    def _from_config(self, obj: Any, config: dict) -> None:
        """Load the configuration from the supplied dictionary.

        Parameters
        ----------
        obj : Any
            The parent object of the Property that we want to update.
        config : dict
            Dictionary of configuration values.

        Raises
        ------
        CaputConfigError
            If there was an error in the config dict.
        """
        # We don't want to emit a deprecation warning here,
        # since it will get raised regardless of whether or
        # not the user is using the deprecated property
        self._set_propname(obj)

        if self.key is None:
            self.key = self.propname

        if self.key in config:
            try:
                val = self.proptype(config[self.key])
            except TypeError as e:
                raise CaputConfigError(
                    f"Can't read value of '{self.key}' as {self.proptype}: {e}",
                    location=config,
                ) from e
            obj.__dict__[self.propname] = val
            # Only warn about deprecation if the property is
            # actually being set by the config.
            self._set_propname_warn_deprecated(obj)

    def _set_propname(self, obj: Any) -> None:
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

    def _set_propname_warn_deprecated(self, obj: Any) -> None:
        # Set the property name and emit a warning
        # if this property is deprecated
        self._set_propname(obj)

        if self._deprecated:
            # Warn the user that they shouldn't be using this. Set the
            # stacklevel to refer to the class where this property
            # is being used.
            # `FutureWarning` is used based on the description:
            # Base category for warnings about deprecated features when
            # those warnings are intended for end users of applications
            # that are written in Python.
            import warnings

            warnings.warn(
                f"Property `{self.propname}` is deprecated and may behave unpredictably. "
                "Check the documentation of the class where this is being used "
                "to see the recommended solution.",
                category=FutureWarning,
                stacklevel=3,
            )


class Reader:
    """A class that allows the values of Properties to be assigned from a dictionary."""

    @classmethod
    def from_config(cls, config: dict, *args: Any, **kwargs: Any) -> Reader:  # noqa: D417
        r"""Create a new instance with values loaded from config.

        Parameters
        ----------
        config : dict
            Dictionary of configuration values.
        \*args : Any
            Variable length argument list
        \**kwargs : Any
            Arbitrary keyword arguments.

        Returns
        -------
        reader_from_config : Reader
            Class instance with values loaded from config.
        """
        c = cls(*args, **kwargs)
        c.read_config(config)

        return c

    def read_config(
        self,
        config: dict,
        compare_keys: bool | list[str] = False,
        use_defaults: bool = True,
    ) -> None:
        """Set all properties in this class from the supplied config.

        Parameters
        ----------
        config : dict
            Dictionary of configuration values.
        compare_keys : bool | list[str], optional
            If True, a CaputConfigError is raised if there are unused keys in the
            config dictionary. If a list of strings is given, any unused keys except the ones in the
            list lead to a :py:exc:`.CaputConfigError`.
        use_defaults : bool, optional
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
                    "Unused configuration keys: [%s]"  # noqa: UP031
                    % ", ".join(set(config_keys) - excluded_keys),
                )

        if not use_defaults:
            if set(prop_keys) - set(config_keys):
                raise CaputConfigError(
                    "Missing configuration keys: [%s]"  # noqa: UP031
                    % ", ".join(set(prop_keys) - set(config_keys)),
                )

        self._finalise_config()

    def _finalise_config(self):
        """Finish up the configuration.

        To be overridden in subclasses if we need to perform some processing
        post configuration.
        """
        ...


def utc_time(default: TimeLike | None = None) -> Property[TimeLike]:
    """Property type for representing UTC as UNIX time.

    Parameters
    ----------
    default : TimeLike | None, optional
        The optional default time.

    Returns
    -------
    utc_parser : Property
        A property instance setup to parse UTC time.
    """

    def _prop(val):
        # Include import here to get around circular import issues
        from .astro.time import ensure_unix

        return ensure_unix(val)

    return Property(proptype=_prop, default=default)


def float_in_range(
    start: float, end: float, default: float | None = None
) -> Property[float]:
    """Property type that tests if its input is within the given range.

    Parameters
    ----------
    start, end : float
        Range to test.
    default : float | None, optional
        The optional default time.

    Returns
    -------
    float_in_range : Property
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
            raise CaputConfigError(f"Input {val:f} not in range [{start:f}, {end:f}]")

        return val

    return Property(proptype=_prop, default=default)


def enum(options: list[Any], default: Any | None = None) -> Property:
    """Property type that accepts only a set of possible values.

    Parameters
    ----------
    options : list[Any]
        List of allowed options.
    default : Any, optional
        The optional default value.

    Returns
    -------
    enum : Property
        A property instance setup to validate an enum type.

    Raises
    ------
    ValueError
        If the default value is not part of the options.

    Examples
    --------
    Should be used like::

        class Project:
            mode = enum(["forward", "backward"], default="forward")
    """

    def _prop(val):
        if val not in options:
            raise CaputConfigError(f"Input {val} not in {options}")

        return val

    if default is not None and default not in options:
        raise ValueError(f"Default value {default} must be in {options} (or None)")

    return Property(proptype=_prop, default=default)


def list_type(  # noqa: D417
    type_: npt.DtypeLike | None = None,
    length: int | None = None,
    maxlength: int | None = None,
    default: Any | None = None,
) -> Property:
    r"""Property type that validates lists against required properties.

    Parameters
    ----------
    type\_ : dtype | None, optional
        Required type element. If `None`, no type validation is done.
    length : int | None, optional
        Exact length of the list we expect. If `None` (default) accept any length.
    maxlength : int | None, optional
        Maximum length of the list. If `None` (default) there is no maximum length.
    default : Any | None, optional
        The optional default value.

    Returns
    -------
    list_type : Property
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
        if not isinstance(val, list | tuple):
            raise CaputConfigError(f"Expected to receive a list, but got '{val!r}.'")

        if type_:
            for ii, item in enumerate(val):
                if not isinstance(item, type_):
                    raise CaputConfigError(
                        f"Expected to receive a list with items of type {type_!s}, but got "
                        f"'{item!s}' of type '{type(item)!s}' at position {ii}"
                    )

        if length and len(val) != length:
            raise CaputConfigError(
                f"List expected to be of length {length}, but was actually length {len(val)}"
            )

        if maxlength and len(val) > maxlength:
            raise CaputConfigError(
                f"Maximum length of list is {maxlength} is, but list was actually length {len(val)}"
            )

        return val

    if default:
        try:
            _prop(default)
        except CaputConfigError as e:
            raise ValueError(
                f"Default value {default} does not satisfy property requirements: {e!r}"
            ) from e

    return Property(proptype=_prop, default=default)


def logging_config(default: str | dict | None = None) -> Property[str | dict]:
    """Property type that validates the caput logging config.

    Allows the type to be either a string (for backward compatibility) or a dict
    setting log levels per module.

    Parameters
    ----------
    default : str | dict | None, optional
        The optional default logging config. A simple string level is applied
        to the root logger.

    Returns
    -------
    logging_config : Property
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

    return Property(proptype=_prop, default=default)


class _line_dict(dict):
    """A private dict subclass that also stores line numbers for debugging."""

    __line__: int | None = None


class SafeLineLoader(SafeLoader):
    """YAML loader that tracks line numbers.

    Adds the line number information to every YAML block. This is useful for
    debugging and to describe linting errors.
    """

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> Mapping:
        """Construct the line mapping."""
        mapping: Mapping = super().construct_mapping(node, deep=deep)
        mapping: _line_dict = _line_dict(mapping)

        # Add 1 so numbering starts at 1
        mapping.__line__ = node.start_mark.line + 1

        return mapping


class CaputConfigError(RuntimeError):
    r"""There was an error in the configuration.

    Parameters
    ----------
    message : str
        Message / description of error
    file\_ : str | None, optional
        Configuration file name.
    location : \_line\_dict | None, optional
        If using :py:class:`.SafeLineLoader` is used, a dict created by that can be
        passed in here to report the line number where the error occurred.
    """

    def __init__(
        self, message: str, file_: str | None = None, location: _line_dict | None = None
    ) -> None:
        self.message = message
        self.file = file_
        if isinstance(location, _line_dict):
            self.line = location.__line__
        else:
            self.line = None
        super().__init__(message)

    def __str__(self) -> str:
        location = ""
        if self.line is not None:
            location = f"\nError in block starting at L{self.line}"
            if self.file is not None:
                location = f"{location} ({self.file})"
        return f"{self.message}{location}"
