"""Tools for caching the results of a long computation to disk.

All data products are assumed to be functions of three things: input_data,
parameters, and optionally the code that generates them (as specified by a git
SHA1, or installed version).

"""

import math
import subprocess
import hashlib
from os import path
import sys


# Determining Code Version
# ------------------------

class UnversionedError(Exception):
    """Raised when code that is expected to be under version control isn't."""
    
    pass


def get_git_dir(module_):
    """Find the .git directory for the code that defines a python module.
    
    If no .git directory can be found - as will be the case for any code not
    under local version control - raises an `UnversionedError`. 
    
    Parameters
    ----------
    module_ : python module

    Returns
    -------
    git_dir : string
        Path to the .git/ directory for the module's source.

    """
    
    # Get the path to the code.
    try:
        file_path = module_.__file__
    except AttributeError:
        msg = "Argument must be a module or package."
        raise TypeError(msg)
    file_path = file_path.split('/')
    # Search the path tree for the .git directory, starting at the deepest
    # directory and proceeding shallower.
    for ii in range(len(file_path) - 2, -1, -1):
        dir_name = '/'.join(file_path[:ii]) + '/.git'
        if path.isdir(dir_name):
            return dir_name
    else:
        msg = "Could not find git repository directory."
        raise UnversionedError(msg)


def get_package_commit(module_):
    """For a module versioned under git, get the SHA1 of the current commit.
    
    Parameters
    ----------
    module_ : python module

    Returns
    -------
    commit : string
        Hex SHA1 hash representing the git commit.

    """

    git_dir = get_git_dir(module_)
    # Run `git show` to get the current commit of caput.
    call = ['git', '--git-dir=' + git_dir, 'show'] 
    proc = subprocess.Popen(call, stdout=subprocess.PIPE)
    ret = proc.wait()
    if not ret == 0:
        raise RuntimeError("Couldn't run git.")
    # Parse stout.
    commit_line = proc.stdout.readline()
    commit = commit_line.split()[1]
    return commit


def get_package_version(module_):
    """Determine the version of top level package from `__version__` attribute.
    
    Pass any module from the package.  If no version can be determined, raise
    an `UnversionedError`.

    Parameters
    ----------
    module_ : python module

    Returns
    -------
    version : string
    
    """
    
    pkg_name = module_.__package__.split('.')[0]
    if pkg_name == module_.__name__:
        # This is the top level.  Get the version.
        try:
            return module_.__version__
        except AttributeError:
            msg = "Package %s has not `__version__` attribute."
            raise UnversionedError(msg % pkg_name)
    else:
        # Get the next level package.
        try:
            pkg = sys.modules[pkg_name]
        except KeyError:
            # Not in names space.  Import it.  As added bonus, this goes
            # straight to the top level, possible skipping several levels of
            # recursion.
            pkg = __import__(pkg_name)
        # Try again.
        return get_package_version(pkg)


def hash_versions(*args):
    """Create a hash representing the versions of passed modules.
    
    The version of each module is identified by it's git commit or if not under
    git version control, the `__version__` attribute of parent package. The
    identifiers for all the modules are then hashed together to produce a
    single identifier for the current state of the collective modules.

    Hash is independent of the argument order.

    Parameters
    ----------
    *args : any number of python modules

    Returns
    -------
    hash : string
        Hex SH1 hash representing the versions of all input modules.

    """

    version_strings = []
    for module in args:
        try:
            version_strings.append(get_package_commit(module))
        except UnversionedError:
            version_strings.append(get_package_version(module))
    version_strings.sort()
    string_to_hash = ' '.join(version_strings)
    hash = hashlib.sha1(string_to_hash).hexdigest()
    return hash


# Hashing Input Parameters
# ------------------------

def hash_obj(obj):
    """Reproducibly compute a hash of any object easily represented in YAML.
    
    Supported types are `dict`, `list`, `set`, `None`, `str`, `int`, `float`,
    `complex`,
    `bool`, and any nesting (but not recursion) there of. Any other types will
    interpreted as one of these types on a best effort basis.  Note in
    particular that tuples and lists hash the same if their contents are the
    same.  This is a consequence of duck typing.

    Hash should be stable across versions of python and platforms (unlike the
    built-in `hash()`). In general, if two values compare as being equal, they
    should hash the same.  That is `1`, `1.00`, and `1.0 + 0j` and `True` all
    have the same hash. It is not easy to engineer a hash collision. `a` hashes
    differently to `repr(a)` and most other representations you might try 
    (unless `a` is a string).

    """
    
    string_to_hash = ''
    # Cases for various types.
    if isinstance(obj, basestring):
        # Any string.
        # This case must come before the sequence case since strings are
        # iterable.
        string_to_hash += obj
    elif hasattr(obj, 'keys') and hasattr(obj, 'items'):
        # Map/dict.
        # This case must come before the sequence case since dicts are
        # iterable.
        string_to_hash += hash_obj('map')
        # Hash the keys, and values then sort by key hashes.
        hashed_pairs = [(hash_obj(k), hash_obj(v)) for k, v in obj.items()]
        hashed_pairs.sort()
        for h_key, h_value in hashed_pairs:
            string_to_hash += h_key + h_value
    elif hasattr(obj, '__iter__'):
        # Sequence/list or Unordered set.
        entry_hashes = []
        for entry in obj:
            entry_hashes.append(hash_obj(entry))
        try: 
            tmp = obj[0]
        except IndexError:
            # Sequence/list of length 0.
            string_to_hash += hash_obj('seq')
        except TypeError:
            # Unordered/set.
            string_to_hash += hash_obj('set')
            entry_hashes.sort()
        else:
            # Sequence/list.
            string_to_hash += hash_obj('seq')
        string_to_hash += ''.join(entry_hashes)
    elif obj is None:
        string_to_hash += hash_obj('null')
    else:
        # Try to interpret as a number.
        try:
            num_repr = _num_hash_parts(obj)
        except TypeError:
            msg = "Argument type %s could not be interpreted."
            msg = msg % repr(type(obj))
            raise TypeError(msg)
        string_to_hash += hash_obj('num')
        string_to_hash += num_repr
    # Compute the hash.
    hash_ = hashlib.sha1(string_to_hash).hexdigest()
    return hash_


def _num_hash_parts(num):
    """Split any number into integer/fractional parts for hashing."""

    if not num.imag == 0:
        im_str = ' ' + _num_hash_parts(num.imag)
    else:
        im_str = ''
    num = num.real
    if math.isnan(num) or math.isinf(num):
        re_str = str(num)
    elif num == int(num):
        # Either an integer (which we must represent at prefect precision),
        # or non fractional in which case this is equivalent to the next case.
        re_str = '%d 1' % int(num)
    else:
        # Unfortunate that I can't think of a way to do this at native
        # precision (without the recasting).
        frac = float(num).as_integer_ratio()
        re_str = "%d %d" % frac
    return re_str + im_str


# Caching
# -------

class Cache(object):
    """A cache on disk for computational products.

    """

    def __init__(self, name, inputs=None, parameters=None, modules=None,
                 root_dir='', ignore_code=True):
        self._name = name
        self._inputs = inputs
        self._parameters = parameters
        
        # First search for an existing cache.  If not found, then make one.
        self._dir = root_dir + '/' + name + stuff

        # Keep a log of cache usage.  Used by scripts to clear out very old and
        # unused caches.  XXX write these scripts.
        with f = file(self._dir + 'cache.usage', 'a'):
            f.write("the date") # XXX

    @property
    def valid(self):
        return path.isfile(self._dir + 'cache.valid')

    @property
    def directory(self):
        return self_dir
    
    def write_info(self):
        fname = self._dir + 'cache.info'
        with f = file(fname, 'w'):
            pass
        # TODO: finish me.
        

    def validate(self):
        self.write_info()
        fname = self._dir + 'cache.valid'
        with file(fname, 'a'):
            os.utime(fname, times)

    def invalidate(self):
        if self.valid:
            path.remove(self._dir + 'cache.valid') 
        if self.code_hash != self._dir.split('_')[-1]:
            # We're ignoring the code versions and modifying a cache created
            # with a different version.
            fname = self._dir + 'cache.code_mixed'
            with file(fname, 'a'):
                os.utime(fname, times)


    
