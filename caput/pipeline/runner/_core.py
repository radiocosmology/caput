"""caput.pipeline.runner."""

import logging
import sys

logger = logging.getLogger(__name__)
# Set the logging level and format
logging.basicConfig(level=logging.INFO, format="%(message)s")


def lint_config(configfile):
    """Lint a pipeline config file without running it.

    Parameters
    ----------
    configfile : str or tuple of str
        Path to a pipeline config file, or multiple config files.
    """
    from .._pipeline import Manager

    # nargs=-1 packs multiple arguments (or glob patterns) into tuples
    if not isinstance(configfile, tuple):
        configfile = (configfile,)

    for cfg in configfile:
        _load_venv(cfg)

        Manager.from_yaml_file(cfg, lint=True)


def run_pipeline(configfile, profile, profiler, mpi_abort, psutil):
    """Run a pipeline immediately from the given `configfile`."""
    from ...util import profiler as prfl
    from .._pipeline import Manager

    if mpi_abort:
        from ...util import mpitools

        mpitools.enable_mpi_exception_handler()

    with prfl.Profiler(profile, profiler=profiler.lower()):
        # TODO: it's a bit weird having one profiler at this level and
        # the other one insoide the manager. Clean this up.
        Manager.from_yaml_file(configfile, psutil_profiling=psutil).run()


def template_run(templatefile, var, *args, **kwargs):
    """Run a pipeline from the given `templatefile`.

    Template variable substitutions are specified with `--var <varname>=<val>`
    arguments, with one for each variable. `<val>` may be a comma separated list, in
    which case item represents a separate value that is processed. Values *must* not
    contain a comma themselves. If multiple variables are specified, each with multiple
    substitutions the outer product of all possible values is generated.
    """
    for tfh_name in _from_template(templatefile, var):
        run_pipeline(tfh_name, *args, **kwargs)


def _from_template(templatefile, var):
    """Run a pipeline from the given `templatefile`.

    This is either run immediately (default), or can be placed in the batch
    queue with the --submit flag.

    Template variable substitutions are specified with `--var <varname>=<val>`
    arguments, with one for each variable. `<val>` may be a comma separated list, in
    which case item represents a separate value that is processed. Values *must* not
    contain a comma themselves. If multiple variables are specified, each with multiple
    substitutions the outer product of all possible values is generated.
    """
    import itertools
    import tempfile

    # Parse all the specified variables into a dictionary, with lists for the values
    # taken by each variable
    vardict = {}
    for v in var:
        varname, vals = v.split("=", 1)
        vardict[varname] = vals.split(",")

    template_string = templatefile.read()

    # Loop over the outer product of all the variables
    for vars_single in itertools.product(*vardict.values()):
        # Construct a dict mapping each variable name to its value
        vardict_single = dict(zip(vardict.keys(), vars_single))

        with tempfile.NamedTemporaryFile("w") as tfh:
            # Output the set of variable values used in this iteration
            var_string = " ".join([f"{k}={v}" for k, v in vardict_single.items()])
            logger.info(f"Running script with {var_string}")

            # Expand and save the job script
            expanded_string = template_string.format(**vardict_single)
            tfh.write(expanded_string)
            tfh.flush()

            yield tfh.name


def _load_config(configfile):
    """Load the given configfile, returning the parsed YAML."""
    import yaml

    with open(configfile) as f:
        return yaml.safe_load(f)


def _load_venv(configfile):
    """Load the venv specified under cluster/venv in the given configfile."""
    import site
    from pathlib import Path

    conf = _load_config(configfile)

    try:
        venv_path = conf["cluster"]["venv"]
    except KeyError:
        # Nothing to load
        return

    logger.info(f"Activating '{venv_path}'...")

    base = Path(venv_path).resolve()

    if not base.exists():
        logger.warning(f"Path defined in 'cluster'/'venv' doesn't exist ({base})")
        sys.exit(1)

    site_packages = base / "lib" / f"python{sys.version[:3]}" / "site-packages"
    prev_sys_path = list(sys.path)

    site.addsitedir(site_packages)
    sys.real_prefix = sys.prefix
    sys.prefix = base

    # Move the added items to the front of the path:
    new_sys_path = []
    for item in list(sys.path):
        if item not in prev_sys_path:
            new_sys_path.append(item)
            sys.path.remove(item)

    sys.path[:0] = new_sys_path
