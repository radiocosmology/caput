"""Caput container utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..mpiarray import SelectionLike
    from ._core import ContainerPrototype

import numpy as np

from ..memdata import _memh5
from ..mpiarray import _apply_sel
from ._core import ContainerPrototype


def empty_like(cont: ContainerPrototype, **kwargs: dict) -> ContainerPrototype:  # noqa: D417
    r"""Create an empty container with the same properties as `cont`.

    Parameters
    ----------
    cont : ContainerPrototype
        Container to base this one off.
    \**kwargs : dict
        Optional definitions of specific axes we want to override. Works in the
        same way as the :py:class:`ContainerPrototype` constructor, though `axes_from=obj` and
        `attrs_from=obj` are implied.

    Returns
    -------
    container : ContainerPrototype
        New, empty container.
    """
    if isinstance(cont, ContainerPrototype):
        return cont.__class__(axes_from=cont, attrs_from=cont, **kwargs)

    raise RuntimeError(f"Unknown object type `{cont.__class__.__name__}`")


def copy_datasets_filter(
    source: ContainerPrototype,
    dest: ContainerPrototype,
    axis: str | Iterable[str] = [],
    selection: SelectionLike = {},
    exclude_axes: tuple[str] | list[str] | None = None,
    copy_without_selection: bool = False,
) -> None:
    """Copy datasets while filtering a given axis.

    By default, only datasets containing the axis to be filtered will be copied.

    Parameters
    ----------
    source : ContainerPrototype
        Source container
    dest : ContainerPrototype
        Destination container. The axes in this container should reflect the
        selections being made to the source.
    axis : str | tuple[str] | list[str]
        Name of the axes to filter. These must match the axes in `selection`,
        unless selection is a single item. This is partially here for legacy
        reasons, as the selections can be fully specified by `selection`
    selection : dict, optional
        A filtering selection to be applied to each axis.
    exclude_axes : list[str] | tuple[str], optional
        An optional set of axes that if a dataset contains one means it will
        not be copied.
    copy_without_selection : bool, optional
        If set to True, then datasets that do not have an axis appearing in
        selection will still be copied over in full.  Default is False.
    """
    exclude_axes_set = set(exclude_axes) if exclude_axes else set()
    if isinstance(axis, str):
        axis = [axis]
    axis = set(axis)

    # Resolve the selections and axes, removing any that aren't needed
    if not isinstance(selection, dict):
        # Assume we just want to apply this selection to all listed axes
        selection = dict.fromkeys(axis, selection)

    if not axis:
        axis = set(selection.keys())
    # Make sure that all axis keys are present in selection
    elif not all(ax in selection for ax in axis):
        raise ValueError(
            f"Mismatch between axis and selection. Got {axis} "
            f"but selections for {list(selection.keys())}."
        )

    # Try to clean up selections
    for ax in list(selection):
        sel = selection[ax]
        # Remove any unnecessary slices
        if sel == slice(None):
            del selection[ax]
        # Convert any indexed selections to slices where possible
        elif type(sel) in {list, tuple, np.ndarray}:
            if list(sel) == list(range(sel[0], sel[-1])):
                selection[ax] = slice(sel[0], sel[-1])

    stack = [source]

    while stack:
        item = stack.pop()

        if _memh5.is_group(item):
            stack += list(item.values())
            continue

        item_axes = list(item.attrs.get("axis", ()))

        # Do not copy datasets that contain excluded axes
        if not exclude_axes_set.isdisjoint(item_axes):
            continue

        # Unless requested, do not copy datasets that do not contain selected axes
        if not copy_without_selection and not axis.intersection(item_axes):
            continue

        if item.name not in dest:
            dest.add_dataset(item.name)

        dest_dset = dest[item.name]

        # Make sure that both datasets are distributed to the same axis
        if isinstance(item, _memh5.MemDatasetDistributed):
            if not isinstance(dest_dset, _memh5.MemDatasetDistributed):
                raise ValueError(
                    "Cannot filter a distributed dataset into a non-distributed "
                    "dataset using this method."
                )

            # Choose the best possible axis to distribute over. Try
            # to avoid redistributing if possible
            original_ax_id = item.distributed_axis
            # If no selections are being made or the slection is not over the
            # current axis, so no need to redistribute
            if not selection or item_axes[original_ax_id] not in selection:
                new_ax_id = original_ax_id
            else:
                # Find the largest axis available
                ax_priority = [
                    x for _, x in sorted(zip(item.shape, item_axes)) if x not in axis
                ]
                if not ax_priority:
                    raise ValueError(
                        "Could not find a valid axis to redistribute. At least one "
                        "axis must be omitted from filtering."
                    )
                new_ax_id = item_axes.index(ax_priority[-1])

            # Make sure both datasets are distributed to the same axis.
            item.redistribute(new_ax_id)
            dest_dset.redistribute(new_ax_id)

        # Apply the selections
        arr = item[:].view(np.ndarray)

        for ax, sel in selection.items():
            try:
                ax_ind = item_axes.index(ax)
            except ValueError:
                continue
            arr = _apply_sel(arr, sel, ax_ind)

        dest_dset[:] = arr[:]

        # also copy attritutes
        _memh5.copyattrs(item.attrs, dest_dset.attrs)

        if isinstance(dest_dset, _memh5.MemDatasetDistributed):
            # Redistribute back to the original axis
            dest_dset.redistribute(original_ax_id)
