"""Fast weighted median and moving weighted median."""

__all__ = ["moving_weighted_median", "quantile", "weighted_median"]

from .weighted import (
    quantile as quantile,
    moving_weighted_median as moving_weighted_median,
    weighted_median as weighted_median,
)
