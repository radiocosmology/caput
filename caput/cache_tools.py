"""Tools for caching the results of a long computation to disk.

All data products are assumed to be functions of three things: input_data,
parameters, and optionally the code that generates them (as specifed by a git
sha1, or installed version).

"""


import subprocess
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
        msg = "Argumment must be a module or package."
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


def get_package_sha1(module_):
    """For a module versioned under git, get the SHA1 of the current commit.

    """

    git_dir = get_git_dir(module_)
    # Run git show to get teh current commit of caput.
    call = ['git', '--git-dir=' + git_dir, 'show'] 
    proc = subprocess.Popen(call, stdout=subprocess.PIPE)
    ret = proc.wait()
    if not ret == 0:
        raise RuntimeError("Couldn't run git.")
    # Parse stout.
    commit_line = proc.stdout.readline()
    commit = commit_line.split()[1]
    return commit
    


class UnversionedError(Exception):
    """Raised when code that is expected to be under version control isn't."""
    
    pass

