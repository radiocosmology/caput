"""CLI interface to run caput pipelines.

Most imports are done inside functions to avoid loading heavy dependencies
unless needed. This might help reduce latency when using the CLI on certain
systems.
"""

import click

from . import _core, _scheduler


@click.group()
def cli():
    """Executes a data analysis pipeline given a pipeline YAML file.

    This script, when executed on the command line, accepts a single parameter, the
    path to a yaml pipeline file.  For an example of a pipeline file, see
    documentation for caput.pipeline.
    """
    pass


@cli.command("lint")
@click.argument(
    "configfile",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True),
)
def lint_config(configfile):
    """Test a pipeline for errors without running it."""
    _core.lint_config(configfile)


@cli.command()
@click.argument(
    "configfile",
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True),
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
@click.option(
    "--mpi-abort/--no-mpi-abort",
    default=True,
    help=(
        "Enable an MPI aware exception handler such that all ranks will exit when any "
        "one throws an exception."
    ),
)
@click.option(
    "--psutil",
    is_flag=True,
    default=False,
    help=(
        "Run the job with a psutil profiler for each task. The output "
        "can be found in the caput logs, at the INFO level."
    ),
)
def run(configfile, profile, profiler, mpi_abort, psutil):
    """Run a pipeline immediately from the given CONFIGFILE."""
    _core.run_pipeline(configfile, profile, profiler, mpi_abort, psutil)


@cli.command()
@click.argument(
    "templatefile",
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True),
)
@click.option(
    "--var",
    multiple=True,
    help=(
        "The template variable value(s). This option can be given multiple times, once "
        "for each variable to be substituted."
    ),
    metavar="<NAME>=<VALUE>[,<VALUE2>...]",
)
@click.option("--submit/--nosubmit", default=False, help="Submit into the batch queue")
@click.option(
    "--overwrite",
    type=click.Choice(["never", "always", "failed"]),
    default="never",
    help=(
        "Overwrite an existing job: never (default); always; or failed "
        "(if there is no STATUS file with the FINISHED status in it)"
    ),
)
@click.option(
    "--email",
    type=str,
    help=("Email address for notifications specified by mailtype option."),
)
@click.option(
    "--mailtype",
    type=click.Choice(_scheduler.MAIL_TYPES),
    help=(
        "Email notification option, following --mail-type syntax for slurm "
        "or -m syntax for PBS"
    ),
)
def template_run(templatefile, submit, var, overwrite, email=None, mailtype=None):
    """Run a pipeline from the given TEMPLATEFILE.

    This is either run immediately (default), or can be placed in the batch
    queue with the --submit flag.

    Template variable substitutions are specified with `--var <varname>=<val>`
    arguments, with one for each variable. `<val>` may be a comma separated list, in
    which case item represents a separate value that is processed. Values *must* not
    contain a comma themselves. If multiple variables are specified, each with multiple
    substitutions the outer product of all possible values is generated.
    """
    if submit:
        _scheduler.template_queue(templatefile, var, overwrite, email, mailtype)
    else:
        _core.template_run(templatefile, var)


@cli.command()
@click.argument(
    "configfile",
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True),
)
@click.option(
    "--submit/--nosubmit", default=True, help="Submit the job to the queue (or not)"
)
@click.option(
    "--lint/--nolint",
    default=True,
    help="Check the pipeline for errors before submitting it.",
)
@click.option(
    "--profile",
    is_flag=True,
    default=False,
    help=(
        "Run the job in a profiler. This will output a `profile_<rank>.prof` file per "
        "MPI rank if using cProfile or `profile_<rank>.txt` file for pyinstrument,"
    ),
)
@click.option(
    "--profiler",
    type=click.Choice(["cProfile", "pyinstrument"], case_sensitive=False),
    default="cProfile",
    help="Set the profiler to use. Default is cProfile.",
)
@click.option(
    "--psutil",
    is_flag=True,
    default=False,
    help=(
        "Run the job with a psutil profiler for each task. The output "
        "can be found in the caput logs, at the INFO level."
    ),
)
@click.option(
    "--overwrite",
    type=click.Choice(["never", "always", "failed"]),
    default="never",
    help=(
        "Overwrite an existing job: never (default); always; or failed "
        "(if there is no STATUS file with the FINISHED status in it)"
    ),
)
@click.option(
    "--email",
    type=str,
    help=("Email address for notifications specified by mailtype option."),
)
@click.option(
    "--mailtype",
    type=click.Choice(_scheduler.MAIL_TYPES),
    help=(
        "Email notification option, following --mail-type syntax for slurm "
        "or -m syntax for PBS"
    ),
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
    _scheduler.queue(
        configfile,
        submit,
        lint,
        profile,
        profiler,
        psutil,
        overwrite,
        email,
        mailtype,
    )


@cli.command()
@click.argument(
    "file",
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True),
)
def register_system(file):
    """Register a new cluster system from the given FILE to a toml file.

    The toml file should contain a table with the name of the system, and
    contain at least the keys `script` and `command`, which provide a formattable
    job submission script template, and the command to submit jobs, respectively.
    """
    _scheduler.register_system(file)
