"""Special handling for systems with a job scheduler."""

__all__ = ["queue", "template_queue"]

import itertools
import logging
from pathlib import Path

from ._core import _from_template, _load_config, lint_config

logger = logging.getLogger(__name__)
# Set the logging level and format
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Email notification options for slurm and PBS
_SLURM_MAIL_TYPES = ["BEGIN", "END", "FAIL", "REQUEUE", "ALL"]
_PBS_MAIL_TYPES = list(
    itertools.chain.from_iterable(
        ("".join(p) for p in itertools.permutations("abe", n)) for n in [1, 2, 3]
    )
)
MAIL_TYPES = _SLURM_MAIL_TYPES + _PBS_MAIL_TYPES


REGISTERED_SYSTEMS = {
    "pbs": {
        "command": "qsub",
        "mail": {
            "types": _PBS_MAIL_TYPES,
            "options": "-M {email} -m {mailtype}",
        },
        "script": """#!/bin/bash
#PBS -l nodes=%(nodes)i:ppn=%(ppn)i
#PBS -q %(queue)s
#PBS -r n
#PBS -m abe
#PBS -V
#PBS -l walltime=%(time)s
#PBS -N %(name)s

# exit if a command returns non-zero code
set -e

# set status to crashed upon non-zero status of command
function setCrashed {
    echo CRASHED > %(statuspath)s
}

trap setCrashed ERR

%(module)s

source %(venv)s

cd %(workdir)s
export OMP_NUM_THREADS=%(ompnum)i

mpirun -np %(mpiproc)i -npernode %(pernode)i -bind-to none python %(scriptpath)s run %(configpath)s &> %(logpath)s

# Set the status
echo FINISHED > %(statuspath)s

# If the job was successful, then move the output to its final location
if [ %(usetemp)s -eq 1 ]
then
    mkdir -p $(dirname \"%(finaldir)s\")
    mv \"%(workdir)s\" \"%(finaldir)s\"
fi
""",
    },
    "slurm": {
        "command": "sbatch",
        "mail": {
            "types": _SLURM_MAIL_TYPES,
            "options": "--mail-user={email} --mail-type={mailtype}",
        },
        "script": """#!/bin/bash
#SBATCH --account=%(account)s
#SBATCH --nodes=%(nodes)i
#SBATCH --ntasks-per-node=%(pernode)i # number of MPI processes
#SBATCH --cpus-per-task=%(ompnum)i # number of OpenMP processes
#SBATCH --mem=%(mem)s # memory per node
#SBATCH --time=%(time)s
#SBATCH --job-name=%(name)s

# exit if a command returns non-zero code
set -e

# set status to crashed upon non-zero status of command
function setCrashed {
    echo CRASHED > %(statuspath)s
}

trap setCrashed ERR

echo RUNNING > %(statuspath)s

%(modules)s

source %(venv)s

cd %(workdir)s
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python %(scriptpath)s run %(profile)s %(profiler)s %(psutil)s %(configpath)s &> %(logpath)s

# Set the status
echo FINISHED > %(statuspath)s

# If the job was successful, then move the output to its final location
if [ %(usetemp)s -eq 1 ]
then
    mkdir -p $(dirname \"%(finaldir)s\")
    mv \"%(workdir)s\" \"%(finaldir)s\"
fi
""",
    },
}

# Global dictionary of registered systems and their default parameters
REGISTERED_CLUSTERS = {
    "gpc": {"ppn": 8, "mem": "16000M", "queue_sys": "pbs", "account": None},
    "cedar": {"ppn": 32, "mem": "0", "queue_sys": "slurm", "account": "rpp-chime"},
    "fir": {"ppn": 32, "mem": "0", "queue_sys": "slurm", "account": "rpp-chime"},
}


def template_queue(templatefile, var, overwrite, email=None, mailtype=None):
    """Queue a pipeline from the given `templatefile`.

    Template variable substitutions are specified with `--var <varname>=<val>`
    arguments, with one for each variable. `<val>` may be a comma separated list, in
    which case item represents a separate value that is processed. Values *must* not
    contain a comma themselves. If multiple variables are specified, each with multiple
    substitutions the outer product of all possible values is generated.
    """
    for tfh_name in _from_template(templatefile, var):
        queue(
            configfile=tfh_name,
            submit=True,
            overwrite=overwrite,
            email=email,
            mailtype=mailtype,
        )


def queue(
    configfile,
    submit=False,
    lint=True,
    profile=False,
    profiler="cProfiler",
    psutil=False,
    overwrite="never",
    email=None,
    mailtype=None,
):
    r"""Queue a pipeline on a cluster from the given CONFIGFILE.

    This queues the job, using parameters from the `cluster` section of the
    submitted YAML file.

    There are several *required* keys:

    \b
    ``nodes``
        The number of nodes to run the job on.
    ``time``
        The time length of the job. Must be a string that the queueing system
        understands.
    ``directory``
        The directory to place the output in.

    There are many *optional* keys that control more functionality:

    \b
    ``system``
        A name of the cluster that we are running on, if this is supported
        (currently ``gpc``, ``cedar``, ``fir``), this uses more relevant default
        values.
    ``queue_sys``
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

    if lint:
        lint_config(configfile)

    conf = _load_config(configfile)

    # Resolve the full queue system configuration and
    # ensure that all required keys are present
    required_keys = {"nodes", "time", "directory", "ppn", "queue", "ompnum", "pernode"}
    resolved_config = _resolve_system_config(conf, required_keys)

    system = resolved_config["queue_sys"]
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


def _expand_path(path):
    """Expand any variables, user directories in path."""
    import os

    return Path(os.path.expandvars(path)).expanduser().resolve()


def _fix_path(path):
    """Turn path to an absolute path."""
    from pathlib import Path

    return Path(path).resolve()


def _resolve_system_config(conf: dict, required_keys: set) -> dict:
    """Resolve a cluster configuration dictionary."""
    # Create output directory and copy over params file.
    try:
        cluster_conf = conf["cluster"]
    except KeyError as exc:
        raise KeyError(
            "Configuration file must have a 'cluster' section. " f"Got {conf.keys()}"
        ) from exc

    # Base setting if nothing else is set
    resolved_conf = {"name": "job", "queue": "batch", "pernode": 1, "ompnum": 8}

    # Resolve default system parameters from registered systems
    try:
        system = cluster_conf["system"]
    except KeyError as exc:
        raise ValueError(
            "Cluster configuration must specify a queue system. "
            f"Got {cluster_conf.keys()}"
        ) from exc

    try:
        sysconfig = REGISTERED_CLUSTERS[system]
    except KeyError as exc:
        raise ValueError(
            f"Specified system `{system}`: is not known. "
            f"Known systems are: {list(REGISTERED_CLUSTERS.keys())}"
        ) from exc

    # Update the resolved configuration with default system parameters,
    # then overwrite with user specified parameters
    resolved_conf.update(**sysconfig)
    resolved_conf.update(**cluster_conf)

    # Make sure the queue system is valid
    if resolved_conf["queue_sys"] not in REGISTERED_SYSTEMS:
        raise ValueError(
            f"Specified queue system `{resolved_conf['queue_sys']}` is not known. "
            f"Known systems are: {list(REGISTERED_SYSTEMS.keys())}"
        )

    # Check to see if any required keys are missing
    missing_keys = required_keys - set(resolved_conf.keys())

    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")

    return resolved_conf


def _create_job_directories(conf: dict, overwrite: str) -> dict[Path, 3 | 0]:
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


def _create_job_environment_strings(conf: dict) -> tuple[str, 2]:
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

    system = conf["queue_sys"]

    # Get the default system queue script and fill in the template variables
    return REGISTERED_SYSTEMS[system]["script"] % conf
