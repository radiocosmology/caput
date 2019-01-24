"""
Configure class attributes using values from a dictionary.

This module to defines strictly typed attributes of a class, that can be loaded
from an input dictionary. This is particularly useful for loading a class from
a YAML document.

Classes
=======

.. autosummary::
   :toctree: generated/

   Property
   Reader

Examples
========

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

>>> print person1.name, person1.age
Bill 26.0
>>> print person2.name, person2.age, person2.petname
Bill 26.0 Molly

Now let's load the configuration from a dictionary:

>>> person1.read_config(testdict)
>>> person2.read_config(testdict)

Then we'll print the output to see the updated configuration:

>>> print person1.name, person1.age
Richard 40.0
>>> print person2.name, person2.age, person2.petname
Richard 40.0 Sooty

"""
# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility

class Property(object):
    """Custom property descriptor that can load values from a given dict.
    """

    def __init__(self, default=None, proptype=None, key=None, required=False):
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
        required : bool
            If True, trying to retrieve this property without first setting
            it will raise AttributError.  If required is True, default must
            be None.
        """

        if default is not None and required:
            raise ValueError("default must be None if required is True")

        self.proptype = (lambda x: x) if proptype is None else proptype
        self.default = default
        self.key = key
        self.propname = None
        self.required = bool(required)

    def __get__(self, obj, objtype):
        # Object getter.
        if obj is None:
            return None

        # Ensure the property name has been found and set
        self._set_propname(obj)

        if self.propname in obj.__dict__:
            # Return the value, if it has been set
            return obj.__dict__[self.propname]
        elif self.required:
            # Raise an exception, if it is required but not set
            raise AttributeError("required Property '{0}' of '{1}' "
                    "object not initialized".format(self.propname,
                        self.__class__.__name__))
        elif self.default is not None:
            # Otherwise, return the default, if provided
            return self.proptype(self.default)

        # In all other cases, return None as a last resort
        return None

    def __set__(self, obj, val):
        # Object setter.
        if obj is None:
            return None

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
        """

        self._set_propname(obj)

        if self.key is None:
            self.key = self.propname

        if self.key in config:
            val = self.proptype(config[self.key])
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


class Reader(object):
    """A class that allows the values of Properties to be assigned from a dictionary.
    """

    @classmethod
    def from_config(cls, config, *args, **kwargs):
        """Create a new instance with values loaded from config.

        Parameters
        ----------
        config : dict
            Dictionary of configuration values.
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
        compare_keys : bool
            If True, an exception is raised if there are unused keys in the
            config dictionary
        use_defaults : bool
            If False, an exception is raised if a property is not defined by
            the config dictionary
        """
        import inspect
        
        config_keys = [x for x in config.keys()]
        prop_keys = []
        for basecls in inspect.getmro(type(self))[::-1]:
            for propname, clsprop in basecls.__dict__.items():
                if isinstance(clsprop, Property):
                    clsprop._from_config(self, config)
                    prop_keys.append(clsprop.key)             

        if compare_keys:
            if set(config_keys) - set(prop_keys):
                raise Exception("Configuration keys [%s] do not have corresponding properties" 
                                % ", ".join(set(config_keys) - set(prop_keys)))
        if not use_defaults:
            if set(prop_keys) - set(config_keys):
                raise Exception("Property keys [%s] are not present in configuration dictionary" 
                                % ", ".join(set(prop_keys) - set(config_keys)))
        
        self._finalise_config()

    def _finalise_config(self):
        """Finish up the configuration.

        To be overriden in subclasses if we need to perform some processing
        post configutation.
        """
        pass


def utc_time(default=None):
    """A property for representing UTC as UNIX time.

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
        from . import time
        return time.ensure_unix(val)

    prop = Property(proptype=_prop, default=default)

    return prop


def float_in_range(start, end, default=None):
    """A property type that tests if its input is within the given range.

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

        class Position(object):

            longitude = config.float_in_range(0.0, 360.0, default=90.0)
    """

    def _prop(val):

        val = float(val)

        if val < start or val > end:
            raise ValueError('Input %f not in range [%f, %f]' % (val, start, end))

        return val

    prop = Property(proptype=_prop, default=default)

    return prop


def enum(options, default=None):
    """A property type that accepts only a set of possible values.

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

    Examples
    --------
    Should be used like::

        class Project(object):

            mode = enum(['forward', 'backward'], default='forward')
    """

    def _prop(val):

        if val not in options:
            raise ValueError('Input %f not in %s' % (repr(val), repr(options)))

        return val

    if default is not None and default not in options:
        raise ValueError('Default value %s must be in %s (or None)' %
                         (repr(default), repr(options)))

    prop = Property(proptype=_prop, default=default)

    return prop


if __name__ == "__main__":
    import doctest
    doctest.testmod()
