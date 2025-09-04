"""Special handling for systems with a job scheduler."""

import itertools
import logging

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

    yconf = _load_config(configfile)

    # Global configuration
    # Create output directory and copy over params file.
    try:
        conf = yconf["cluster"]
    except KeyError as exc:
        raise KeyError('Configuration file must have a "cluster" section.') from exc

    # Base setting if nothing else is set
    defaults = {"name": "job", "queue": "batch", "pernode": 1, "ompnum": 8}

    # Per system overrides (i.e. specialisations for Scinet GPC and Westgrid
    # cedar)
    system_defaults = {
        "gpc": {"ppn": 8, "mem": "16000M", "queue_sys": "pbs", "account": None},
        "cedar": {"ppn": 32, "mem": "0", "queue_sys": "slurm", "account": "rpp-chime"},
        "fir": {"ppn": 32, "mem": "0", "queue_sys": "slurm", "account": "rpp-chime"},
    }

    # Start to generate the full resolved config
    rconf = defaults.copy()

    # If the system is specified update the current config with it
    if "system" in conf:
        system = conf["system"]

        if system not in system_defaults:
            raise ValueError(f'Specified system "{system}": is not known.')

        rconf.update(**system_defaults[system])

    # Update the current config with the rest of the users variables
    rconf.update(**conf)

    # Check to see if any required keys are missing
    required_keys = {"nodes", "time", "directory", "ppn", "queue", "ompnum", "pernode"}
    missing_keys = required_keys - set(rconf.keys())
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")

    # If no temporary directory set, just use the final directory
    if "temp_directory" not in rconf:
        rconf["temp_directory"] = rconf["directory"]

    # Construct the working directory
    workdir = _expand_path(rconf["temp_directory"])
    if not workdir.is_absolute():
        raise ValueError(f"Working directory path {workdir} must be absolute")

    # Construct the output directory
    finaldir = _expand_path(rconf["directory"])
    if not finaldir.is_absolute():
        raise ValueError(f"Final output directory path {finaldir} must be absolute")

    # Create temporary directory if required
    jobdir = workdir / "job/"
    statusfile = jobdir / "STATUS"

    if jobdir.exists():
        if overwrite == "never":
            logger.info(f"Job already exists at {workdir}. Skipping.")
            return

        if overwrite == "failed" and statusfile.exists():
            with open(statusfile) as fh:
                contents = fh.read()
                if "FINISHED" in contents:
                    logger.info(
                        f"Successful job already exists at {workdir}. Skipping."
                    )
                    return
    else:
        jobdir.mkdir(parents=True)

    # Copy config file into output directory (check it's not already there first)
    sfile = _fix_path(configfile)
    dfile = _fix_path(jobdir / "config.yaml")
    if sfile != dfile:
        shutil.copyfile(sfile, dfile)

    # Get any modules that should be loaded
    modlist = rconf.get("module_list")
    modpath = rconf.get("module_path")
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

    rconf["modules"] = modstr

    # Set up virtualenv
    if "venv" in rconf:
        venvpath = _expand_path(rconf["venv"] + "/bin/activate")
        if not venvpath.exists():
            raise ValueError(f"Could not find virtualenv at path {rconf['venv']}")
        rconf["venv"] = venvpath
    else:
        rconf["venv"] = "/dev/null"

    # Forward profiler configuration
    rconf["profile"] = "--profile" if profile else ""
    rconf["profiler"] = f"--profiler={profiler}" if profile else ""
    rconf["psutil"] = "--psutil" if psutil else ""

    # Derived vars only needed to create script
    rconf["mpiproc"] = rconf["nodes"] * rconf["pernode"]
    rconf["workdir"] = workdir
    rconf["finaldir"] = finaldir
    rconf["scriptpath"] = _fix_path(__file__)
    rconf["logpath"] = jobdir / "jobout.log"
    rconf["configpath"] = jobdir / "config.yaml"
    rconf["statuspath"] = jobdir / "STATUS"
    rconf["usetemp"] = 1 if rconf["finaldir"] != rconf["workdir"] else 0

    pbs_script = """#!/bin/bash
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
"""
    slurm_script = """#!/bin/bash
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
"""

    if rconf["queue_sys"] == "pbs":
        script = pbs_script
        job_command = "qsub"

        if email is None:
            job_options = []
        else:
            if mailtype is None:
                raise ValueError("Must specify PBS mailtype for email notifications")
            if mailtype not in _PBS_MAIL_TYPES:
                raise ValueError(f"Invalid PBS mailtype specified ({mailtype})")

            job_options = [f"-M {email}", f"-m {mailtype}"]

    elif rconf["queue_sys"] == "slurm":
        script = slurm_script
        job_command = "sbatch"

        if email is None:
            job_options = []
        else:
            if mailtype is None:
                raise ValueError("Must specify slurm mailtype for email notifications")
            if mailtype not in _SLURM_MAIL_TYPES:
                raise ValueError(f"Invalid slurm mailtype specified ({mailtype})")

            job_options = [f"--mail-user={email}", f"--mail-type={mailtype}"]

    else:
        raise ValueError("Specified queueing system not recognized")

    # Fill in the template variables
    script = script % rconf

    # Write and submit the jobscript
    with open(jobdir / "jobscript.sh", "w") as f:
        f.write(script)

    if submit:
        # NOTE: explicitly set the environment to what Python thinks it should
        # be. This is because mpi4py will incorrectly modify the environment
        # using lowlevel calls which Python cannot detect.
        subprocess.run(
            [job_command, *job_options, "jobscript.sh"], cwd=jobdir, env=os.environ
        )


def _expand_path(path):
    """Expand any variables, user directories in path."""
    import os
    from pathlib import Path

    return Path(os.path.expandvars(path)).expanduser().resolve()


def _fix_path(path):
    """Turn path to an absolute path."""
    from pathlib import Path

    return Path(path).resolve()
