"""Tools for caching the results of a long computation to disk.

All data products are assumed to be functions of three things: input_data,
parameters, and optionally the code that generates them (as specified by a git
SHA1, or installed version).

"""

import math
import subprocess
import hashlib
from os import path



def get_git_dir(module_):
    """Find the .git directory for the code that defines a python module.
    
    If no .git directory can be found - as will be the case for any code not
    under local version control - raises an `UnversionedError`. 

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
            # Cast as most general type.
            obj = complex(obj)
        except TypeError:
            msg = "Argument type could not be interpreted."
            raise TypeError(msg)
        string_to_hash += hash_obj('num')
        string_to_hash += ' '.join(_float_hash_parts(obj.real)
                                   + _float_hash_parts(obj.imag))
    # Compute the hash.
    hash_ = hashlib.sha1(string_to_hash).hexdigest()
    return hash_

def _float_hash_parts(fl):
    if math.isnan(fl) or math.isinf(fl):
        return (str(fl), )
    n, d = fl.as_integer_ratio()
    return (str(n), str(d))

class UnversionedError(Exception):
    """Raised when code that is expected to be under version control isn't."""
    
    pass

