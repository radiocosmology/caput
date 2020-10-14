from os.path import (
    normpath,
    expanduser,
    expandvars,
    realpath,
    abspath,
    isabs,
    join,
    exists,
)

import click


products = None


@click.group()
def cli():
    """Executes a data analysis pipeline given a pipeline YAML file.

    This script, when executed on the command line, accepts a single parameter, the
    path to a yaml pipeline file.  For an example of a pipeline file, see
    documentation for caput.pipeline.
    """
    pass


@cli.command()
@click.argument(
    "configfile",
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True),
)
@click.option(
    "--loglevel",
    type=click.Choice(
        ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "CONFIG"],
        case_sensitive=False,
    ),
    default="CONFIG",
    help="Logging level (deprecated, use the config instead).",
)
@click.option(
    "--profile",
    is_flag=True,
    default=False,
    help=(
        "Run the job in a profiler. This will output a `profile_<rank>.prof` file per "
        "MPI rank if using cProfile or `profile_<rank>.txt` file for pyinstrument."
    ),
)
@click.option(
    "--profiler",
    type=click.Choice(["cProfile", "pyinstrument"], case_sensitive=False),
    default="cProfile",
    help="Set the profiler to use. Default is cProfile.",
)
def run(configfile, loglevel, profile, profiler):
    """Run a pipeline immediately from the given CONFIGFILE."""
    from caput.pipeline import Manager

    import warnings

    if loglevel != "CONFIG":
        warnings.warn(
            "--loglevel is deprecated, use the config file instead", DeprecationWarning
        )

    if profile:
        if profiler == "cProfile":
            import cProfile

            pr = cProfile.Profile()
            pr.enable()
        elif profiler == "pyinstrument":
            from pyinstrument import Profiler

            pr = Profiler()
            pr.start()

    P = Manager.from_yaml_file(configfile)
    P.run()

    if profile:

        from caput import mpiutil

        rank = mpiutil.rank

        if profiler == "cProfile":
            pr.disable()
            pr.dump_stats("profile_%i.prof" % mpiutil.rank)
        elif profiler == "pyinstrument":
            pr.stop()
            with open("profile_%i.txt" % rank, "w") as fh:
                fh.write(pr.output_text(unicode=True))


@cli.command()
@click.argument(
    "configfile",
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True),
)
@click.option(
    "--submit/--nosubmit", default=True, help="Submit the job to the queue (or not)"
)
def queue(configfile, submit=False):
    """Queue a pipeline on a cluster from the given CONFIGFILE.

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
        (currently ``gpc`` and ``cedar``), this uses more relevant default
        values.
    ``queue_system``
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
    ``temp_directory``
        If set, save the output to a temporary location while running and
        then move to a final location if the job successfully finishes. This
        may be slow, if the temporary and final directories are not on the
        same filesystem.
    """

    import os.path
    import shutil
    import yaml

    with open(configfile, "r") as f:
        yconf = yaml.safe_load(f)

    ## Global configuration
    ## Create output directory and copy over params file.
    if "cluster" not in yconf:
        raise ValueError('Configuration file must have an "cluster" section.')

    conf = yconf["cluster"]

    # Base setting if nothing else is set
    defaults = {"name": "job", "queue": "batch", "pernode": 1, "ompnum": 8}

    # Per system overrides (i.e. specialisations for Scinet GPC and Westgrid
    # cedar)
    system_defaults = {
        "gpc": {"ppn": 8, "mem": "16000M", "queue_sys": "pbs", "account": None},
        "cedar": {"ppn": 32, "mem": "0", "queue_sys": "slurm", "account": "rpp-krs"},
    }

    # Start to generate the full resolved config
    rconf = defaults.copy()

    # If the system is specified update the current config with it
    if "system" in conf:

        sys = conf["system"]

        if sys not in system_defaults:
            raise ValueError('Specified system "%s": is not known.' % sys)

        rconf.update(**system_defaults[sys])

    # Update the current config with the rest of the users variables
    rconf.update(**conf)

    # Check to see if any required keys are missing
    required_keys = {"nodes", "time", "directory", "ppn", "queue", "ompnum", "pernode"}
    missing_keys = required_keys - set(rconf.keys())
    if missing_keys:
        raise ValueError("Missing required keys: %s" % missing_keys)

    # If no temporary directory set, just use the final directory
    if "temp_directory" not in rconf:
        rconf["temp_directory"] = rconf["directory"]

    # Construct the working directory
    workdir = expandpath(rconf["temp_directory"])
    if not isabs(workdir):
        raise ValueError("Working directory path %s must be absolute" % workdir)

    # Construct the output directory
    finaldir = expandpath(rconf["directory"])
    if not isabs(finaldir):
        raise ValueError("Final output directory path %s must be absolute" % finaldir)

    # Create temporary directory if required
    jobdir = join(workdir, "job/")
    if not exists(jobdir):
        os.makedirs(jobdir)

    # Copy config file into output directory (check it's not already there first)
    sfile = fixpath(configfile)
    dfile = fixpath(join(jobdir, "config.yaml"))
    if sfile != dfile:
        shutil.copyfile(sfile, dfile)

    # Set up virtualenv
    if "venv" in rconf:
        venvpath = rconf["venv"] + "/bin/activate"
        if not exists(venvpath):
            raise ValueError("Could not find virtualenv at path %s" % rconf["venv"])
        rconf["venv"] = venvpath
    else:
        rconf["venv"] = "/dev/null"

    # Derived vars only needed to create script
    rconf["mpiproc"] = rconf["nodes"] * rconf["pernode"]
    rconf["workdir"] = workdir
    rconf["finaldir"] = finaldir
    rconf["scriptpath"] = fixpath(__file__)
    rconf["logpath"] = join(jobdir, "jobout.log")
    rconf["configpath"] = join(jobdir, "config.yaml")
    rconf["statuspath"] = join(jobdir, "STATUS")
    rconf["usetemp"] = 1 if rconf["finaldir"] != rconf["workdir"] else 0

    pbs_script = """#!/bin/bash
#PBS -l nodes=%(nodes)i:ppn=%(ppn)i
#PBS -q %(queue)s
#PBS -r n
#PBS -m abe
#PBS -V
#PBS -l walltime=%(time)s
#PBS -N %(name)s

source %(venv)s

cd %(workdir)s
export OMP_NUM_THREADS=%(ompnum)i

mpirun -np %(mpiproc)i -npernode %(pernode)i -bind-to none python %(scriptpath)s run %(configpath)s &> %(logpath)s

retcode=$?

# Set the status
if [ $retcode -eq 0 ]
then
    echo FINISHED > %(statuspath)s
else
    echo CRASHED > %(statuspath)s
fi

# If the job was successful, then move the output to its final location
if [ %(usetemp)s -eq 1 ] && [ $retcode -eq 0 ]
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

echo RUNNING > %(statuspath)s

source %(venv)s

cd %(workdir)s
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python %(scriptpath)s run %(configpath)s &> %(logpath)s

retcode=$?

# Set the status
if [ $retcode -eq 0 ]
then
    echo FINISHED > %(statuspath)s
else
    echo CRASHED > %(statuspath)s
fi

# If the job was successful, then move the output to its final location
if [ %(usetemp)s -eq 1 ] && [ $retcode -eq 0 ]
then
    mkdir -p $(dirname \"%(finaldir)s\")
    mv \"%(workdir)s\" \"%(finaldir)s\"
fi
"""

    if rconf["queue_sys"] == "pbs":
        script = pbs_script
        job_command = "qsub"
    elif rconf["queue_sys"] == "slurm":
        script = slurm_script
        job_command = "sbatch"
    else:
        raise ValueError("Specified queueing system not recognized")

    # Fill in the template variables
    script = script % rconf

    # Write and submit the jobscript
    with open(jobdir + "/jobscript.sh", "w") as f:
        f.write(script)
    if submit:
        os.system("cd %s; %s jobscript.sh" % (jobdir, job_command))


def expandpath(path):
    """Expand any variables, user directories in path"""
    return normpath(expandvars(expanduser(path)))


def fixpath(path):
    """Turn path to an absolute path"""
    return realpath(abspath(path))


# This is needed because the queue script calls this file directly.
if __name__ == "__main__":
    cli()
