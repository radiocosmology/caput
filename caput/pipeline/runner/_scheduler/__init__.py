"""Special handling for systems with a job scheduler."""

__all__ = ["queue", "register_system", "template_queue"]

import os
import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)
# Set the logging level and format
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Global dictionary of registered queue systems
REGISTERED_SYSTEMS = {}
# Global list of supported mail types across all systems
MAIL_TYPES = []


def _expand_path(path: os.PathLike) -> Path:
    """Expand any environment variables, user directories in path."""
    import os

    return Path(os.path.expandvars(path)).expanduser().resolve()


def _fix_path(path: os.PathLike) -> Path:
    """Turn path to an absolute path."""
    return Path(path).resolve()


def register_system(path: os.PathLike) -> None:
    """Register new queue system(s) from a TOML file."""
    import toml

    global REGISTERED_SYSTEMS
    global MAIL_TYPES

    required_keys = {"script", "command"}

    path = _fix_path(path)

    with open(path) as f:
        data = toml.load(f)

    for system, conf in data.items():
        if not required_keys.issubset(conf.keys()):
            raise ValueError(
                f"System {system} is missing required keys: "
                f"{required_keys - set(conf.keys())}"
            )

        REGISTERED_SYSTEMS[system] = conf

        if "mail" in conf:
            MAIL_TYPES.extend(conf["mail"]["types"])


# Register default queue systems
register_system(Path(__file__).parent / "_default_systems.toml")

# Global dictionary of registered systems and their default parameters.
# This shouldn't be used, and is just a fallback to support legacy config files.
# New config files should specify the queue system directly.
REGISTERED_CLUSTERS = {"gpc": "pbs", "cedar": "slurm", "fir": "slurm"}


def template_queue(
    templatefile: os.PathLike,
    var: str,
    overwrite: Literal["never", "failed"],
    email: str | None = None,
    mailtype: str | None = None,
):
    """Queue a pipeline from the given `templatefile`.

    Template variable substitutions are specified with `--var <varname>=<val>`
    arguments, with one for each variable. `<val>` may be a comma separated list, in
    which case item represents a separate value that is processed. Values *must* not
    contain a comma themselves. If multiple variables are specified, each with multiple
    substitutions the outer product of all possible values is generated.

    Parameters
    ----------
    templatefile : os.PathLike
        Path to a `.yaml` file containing the pipeline template.
    var : str
        Template variable substitutions.
    overwrite : {"never", "failed"}
        How to handle job directories which already exist. If "failed",
        only jobs which have reported `FAILED` will be re-queued.
    email : str | None, optional
        Email address for job status notifications.
    mailtype : str | None, optional
        Types of job events for which to send email notifications. These
        are typically specific to the queue system used.
    """
    from .._core import _from_template

    for tfh_name in _from_template(templatefile, var):
        queue(
            configfile=tfh_name,
            submit=True,
            overwrite=overwrite,
            email=email,
            mailtype=mailtype,
        )


def queue(
    configfile: os.PathLike,
    submit: bool = False,
    lint: bool = True,
    profile: bool = False,
    profiler: str | None = "cProfiler",
    psutil: bool = False,
    overwrite: Literal["never", "failed"] = "never",
    email: str | None = None,
    mailtype: str | None = None,
):
    """Queue a pipeline on a cluster from the given `configfile`.

    This queues the job, using parameters from the `cluster` section of the
    submitted YAML file.

    Parameters
    ----------
    configfile : os.PathLike
        Path to a `.yaml` pipeline config file.
    submit : bool, optional
        If True, the job will be submitted to the scheduler. Otherwise, the
        job directory and files will be created but not submitted. Default
        is False.
    lint : bool, optional
        If True, lint the `configfile` before creating any job files. Default
        is True.
    profile : bool, optional
        If True, use a profiler to monitor the time and resource usage of
        the pipeline job. Default is False.
    profiler : {"cprofile", "pyinstrument"}, optional
        Which profiler to use if `profile` is True. Default is `cprofile`.
    psutil : bool, optional
        If True, use `psutil` to monitor the memory use of the pipeline job.
        Default is False.
    overwrite : {"never", "failed"}, optional
        How to handle job directories which already exist. If "failed",
        only jobs which have reported `FAILED` will be re-queued. Default
        is "never".
    email : str | None, optional
        Email address for job status notifications. Default is None
    mailtype : str | None, optional
        Types of job events for which to send email notifications. These
        are typically specific to the queue system used. Default is None.

    Cluster Config
    ~~~~~~~~~~~~~~
    There are several *required* keys:

    ``nodes``
        The number of nodes to run the job on.
    ``time``
        The time length of the job. Must be a string that the queueing system
        understands.
    ``directory``
        The directory to place the output in.

    There are many *optional* keys that control more functionality:

    ``system``
        The name of the cluster that we are running on. If this is a known system
        (currently ``gpc``, ``cedar``, ``fir``), more relevant defaults are used.
    ``system``
        The queue system to run on. Either ``pbs`` or ``slurm``.
    ``queue``
        The queue to submit to. Only used for *PBS*
    ``ompnum``
        The number of OpenMP threads to use.
    ``pernode``
        Number of processes to run on each node.
    ``mem``
        How much memory to reserve per node.
    ``account``
        The account to submit the job against. Only used on *SLURM*
    ``ppn``
        Only used for PBS. Should typically be equal to the number of
        *processors* on a node.
    ``venv``
        Path to a virtual environment to load before running.
    ``module_list``
        Only used for slurm.
        A list of modules environments to load before running a job.
        If set, a module purge will occur before loading the specified modules.
        Sticky modules like StdEnv/* on Cedar and Fir will not get purged, and
        should not be specified.
        If not set, the current environment is used.
    ``module_path``
        Only used for slurm.
        A list of modules paths to use. May be required to load modules.
    ``temp_directory``
        If set, save the output to a temporary location while running and
        then move to a final location if the job successfully finishes. This
        may be slow, if the temporary and final directories are not on the
        same filesystem.
    """
    import os
    import shutil
    import subprocess

    from .._core import _load_config, lint_config

    if lint:
        lint_config(configfile)

    conf = _load_config(configfile)

    # Resolve the full queue system configuration and
    resolved_config = _resolve_system_config(conf)

    system = resolved_config["system"]
    system_config = REGISTERED_SYSTEMS[system]

    directories = _create_job_directories(resolved_config, overwrite)

    if not directories:
        # The job already exists and we are not overwriting
        return

    # Copy the config file to the job directory
    sfile = _fix_path(configfile)
    dfile = _fix_path(directories["jobdir"] / "config.yaml")

    if sfile != dfile:
        shutil.copyfile(sfile, dfile)

    # Forward profiler configuration
    resolved_config["profile"] = "--profile" if profile else ""
    resolved_config["profiler"] = f"--profiler={profiler}" if profile else ""
    resolved_config["psutil"] = "--psutil" if psutil else ""

    # Resolve the cluster job script based on the configuration
    jobscript = _resolve_job_script(resolved_config, directories)

    # Write the job script to the job directory
    with open(directories["jobdir"] / "jobscript.sh", "w") as f:
        f.write(jobscript)

    # Resolve mail options, if available for the queue system
    extra_options = []

    if email is not None:
        if mailtype is None:
            logger.warning(f"Must specify {system} mailtype for email notifications")

        mail_config = system_config.get("mail")

        if mail_config is None:
            logger.warning(f"Email notifications not supported for {system}")
        else:
            if mailtype not in mail_config["types"]:
                raise ValueError(
                    f"Invalid {system} mailtype specified: ({mailtype}). "
                    f"Valid types are: ({mail_config['types']})."
                )

            extra_options = (
                mail_config["options"].format(email=email, mailtype=mailtype).split()
            )

    if submit:
        # NOTE: explicitly set the environment to what Python thinks it should
        # be. This is because mpi4py will incorrectly modify the environment
        # using lowlevel calls which Python cannot detect.
        subprocess.run(
            [system_config["command"], *extra_options, "jobscript.sh"],
            cwd=directories["jobdir"],
            env=os.environ,
        )


def _resolve_system_config(conf: dict) -> dict:
    """Resolve a cluster configuration dictionary."""
    # Create output directory and copy over params file.
    try:
        cluster_conf = conf["cluster"]
    except KeyError as exc:
        raise KeyError(
            f"Configuration file must have a 'cluster' section. Got {conf.keys()}"
        ) from exc

    # Resolve default system parameters from registered systems
    sysname = cluster_conf["system"]
    try:
        system = REGISTERED_SYSTEMS[sysname]
    except KeyError:
        # As a fallback, see if the user specified a known system
        # using a cluster name
        if sysname in REGISTERED_CLUSTERS:
            sysname = REGISTERED_CLUSTERS[sysname]
            system = REGISTERED_SYSTEMS[sysname]

        else:
            raise ValueError(
                f"Specified system `{system}`: is not known. "
                f"Known systems are: {list(REGISTERED_SYSTEMS.keys())} "
                f"and known clusters are: {list(REGISTERED_CLUSTERS.keys())}"
            )

        cluster_conf["system"] = sysname

    # Certain keys are required
    required_keys = set(system.get("required", set()))

    # Check to see if any required keys are missing
    missing_keys = required_keys - set(cluster_conf.keys())

    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")

    return cluster_conf


def _create_job_directories(conf: dict, overwrite: str) -> dict[str, Path]:
    """Create job directories based on a configuration dictionary."""
    directories = {}

    # If no temporary directory set, just use the final directory
    if "temp_directory" not in conf:
        conf["temp_directory"] = conf["directory"]

    # Construct the working directory
    workdir = _expand_path(conf["temp_directory"])

    if not workdir.is_absolute():
        raise ValueError(f"Working directory path {workdir} must be absolute")

    # Construct the output directory
    finaldir = _expand_path(conf["directory"])

    if not finaldir.is_absolute():
        raise ValueError(f"Final output directory path {finaldir} must be absolute")

    # Create temporary directory if required
    jobdir = workdir / "job/"
    statusfile = jobdir / "STATUS"

    if jobdir.exists():
        if overwrite == "never":
            logger.info(f"Job already exists at {workdir}. Skipping.")
            return None

        if overwrite == "failed" and statusfile.exists():
            with open(statusfile) as fh:
                contents = fh.read()
                if "FINISHED" in contents:
                    logger.info(
                        f"Successful job already exists at {workdir}. Skipping."
                    )
                    return None
    else:
        jobdir.mkdir(parents=True)

    directories["jobdir"] = jobdir
    directories["workdir"] = workdir
    directories["finaldir"] = finaldir

    return directories


def _create_job_environment_strings(conf: dict) -> tuple[str, str]:
    """Load the job environment based on a configuration dictionary.

    Returns a tuple consisting of the module load string and the
    virtual environment path.
    """
    # Get any modules that should be loaded
    modlist = conf.get("module_list")
    modpath = conf.get("module_path")
    modstr = ""

    if modpath:
        if isinstance(modpath, str):
            modpath = (modpath,)
        modstr += "module use " + "\nmodule use ".join(modpath)

    if modlist:
        if isinstance(modlist, str):
            modlist = (modlist,)
        modstr += "\nmodule load " + "\nmodule load ".join(modlist)

    if modstr:
        modstr = "module --force purge\n" + modstr

    # Set up virtualenv
    if "venv" in conf:
        venvpath = _expand_path(conf["venv"] + "/bin/activate")

        if not venvpath.exists():
            raise ValueError(f"Could not find virtualenv at path {conf['venv']}")
    else:
        venvpath = "/dev/null"

    return modstr, venvpath


def _resolve_job_script(conf: dict, directories: dict) -> str:
    """Resolve the job script based on a configuration dictionary."""
    # Parse the module load string and virtualenv path
    modstr, venvpath = _create_job_environment_strings(conf)
    conf["modules"] = modstr
    conf["venv"] = venvpath

    # Derived vars only needed to create script
    conf["mpiproc"] = conf["nodes"] * conf["pernode"]
    conf["workdir"] = directories["workdir"]
    conf["finaldir"] = directories["finaldir"]
    conf["scriptpath"] = _fix_path(__file__)
    conf["logpath"] = directories["jobdir"] / "jobout.log"
    conf["configpath"] = directories["jobdir"] / "config.yaml"
    conf["statuspath"] = directories["jobdir"] / "STATUS"
    conf["usetemp"] = 1 if conf["finaldir"] != conf["workdir"] else 0

    system = conf["system"]

    # Get the default system queue script and fill in the template variables
    return REGISTERED_SYSTEMS[system]["script"] % conf
