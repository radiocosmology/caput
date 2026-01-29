"""Tools for dynamic imports.

Implements helpful tools which are not present in `importlib`.
"""

import importlib
from typing import Any

__all__ = ["import_class"]


def import_class(class_path: str) -> Any:
    """Import class dynamically from a string.

    Parameters
    ----------
    class_path : str
        Fully qualified path to the class. If only a single component, look up in the
        globals.

    Returns
    -------
    imported_class : Any
        The class we want to load.
    """
    path_split = class_path.split(".")
    module_path = ".".join(path_split[:-1])
    class_name = path_split[-1]
    if module_path:
        m = importlib.import_module(module_path)
        task_cls = getattr(m, class_name)
    else:
        task_cls = globals()[class_name]

    return task_cls
