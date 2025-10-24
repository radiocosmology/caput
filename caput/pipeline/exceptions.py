"""Pipeline exceptions."""

__all__ = ["PipelineRuntimeError", "PipelineStopIteration"]


class PipelineRuntimeError(RuntimeError):
    """Raised when there is a pipeline related error at runtime."""


class PipelineStopIteration(Exception):
    """Stop the iteration of `next()` in pipeline tasks.

    Pipeline tasks should raise this excetions in the `next()` method to stop
    the iteration of the task and to proceed to `finish()`.

    Note that if `next()` recieves input data as an argument, it is not
    required to ever raise this exception.  The pipeline will proceed to
    `finish()` once the input data has run out.
    """
