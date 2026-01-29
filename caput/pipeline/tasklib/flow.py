"""Simple pipeline tasks used to control how data flows through the pipeline."""

from __future__ import annotations

from ... import config
from .base import MPILoggedTask


class AccumulateList(MPILoggedTask):
    """Accumulate the inputs into a list and return as a group.

    If `group_size` is None, return when the task *finishes*. Otherwise,
    return every time `group_size` inputs have been accumulated.

    Attributes
    ----------
    group_size : int | None
        If this is set, this task will return the list of accumulated
        data whenever it reaches this length. If not set, wait until
        no more input is received and then return everything.
    """

    group_size = config.Property(proptype=int, default=None)

    def __init__(self):
        super().__init__()
        self._items = []

    def next(self, input_):  # noqa: D417
        r"""Append an input to the list of inputs.

        Parameters
        ----------
        input\_ : Any
            Arbitrary input to accumulate.

        Returns
        -------
        output : list[Any] | None
            If :py:attr:`~.AccumulateList.group_size` is ``None``,
            nothing is returned until this task reaches
            :py:meth:`~.AccumulateList.finish`. Otherwise, return
            a list of length `group_size`.
        """
        self._items.append(input_)

        if self.group_size is not None:
            if len(self._items) >= self.group_size:
                output = self._items
                self._items = []

                return output

        return None

    def finish(self):
        """Remove the internal reference.

        Prevents the items from hanging around after the task finishes.

        Returns
        -------
        items : list | None
            Accumulated list of inputs, or ``None`` if `group_size`
            was set, in which case the inputs are returned from
            :py:meth:`~.AccumulateList.next`.
        """
        items = self._items
        del self._items

        # If the group_size was set, then items will either be an empty list
        # or an incomplete list (with the incorrect number of outputs), so
        # in that case return None to prevent the pipeline from crashing.
        return items if self.group_size is None else None


class GatherInputs(MPILoggedTask):
    """Gather an arbitrary number of inputs into a list.

    The relevant number of inputs is determined by the task
    configuration keys.
    """

    def next(self, *inputs):  # noqa: D417
        r"""Gather the inputs into a list.

        Parameters
        ----------
        \*inputs : list[Any]
            Arbitrary number of inputs.

        Returns
        -------
        output : list[Any]
            A list containing all the inputs.
        """
        return list(inputs)


class Delete(MPILoggedTask):
    """Delete pipeline products to free memory."""

    def process(self, x):
        """Delete the input and collect garbage.

        Parameters
        ----------
        x : Any
            The object to be deleted.
        """
        import gc

        self.log.info(f"Deleting {type(x)!s}")
        del x
        gc.collect()


class MakeCopy(MPILoggedTask):
    """Make a copy of the passed object."""

    def process(self, data):
        """Return a deep copy of the given object.

        Parameters
        ----------
        data : Any
            The object to copy.

        Returns
        -------
        copied_object : Any
            A deep copy of the input object.
        """
        import copy

        return copy.deepcopy(data)


class PassOn(MPILoggedTask):
    """Unconditionally forward a tasks input.

    While this seems like a pointless no-op it's useful for connecting tasks in complex
    topologies.
    """

    def next(self, input_):
        """Immediately forward any input."""
        return input_


class WaitUntil(MPILoggedTask):
    """Wait until the ``requires`` key is received before forwarding inputs.

    This simple synchronization task will forward on whatever inputs it gets, however, it won't do
    this until it receives any requirement to it's setup method. This allows certain parts of the
    pipeline to be delayed until a piece of data further up has been generated.
    """

    def setup(self, input_):
        """Accept, but don't save any input."""
        self.log.info("Received the requirement, starting to forward inputs")
        pass

    def next(self, input_):
        """Immediately forward any input."""
        return input_
