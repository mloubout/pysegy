"""Local browser viewer for :mod:`pysegy` datasets.

The viewer is an optional component. Install it with ``pysegy[viewer]`` and
launch it with the ``pysegy-viewer`` command.
"""

from .services import (
    DatasetSummary,
    GatherWindow,
    HeaderTable,
    dataset_summary,
    load_gather,
    load_header_table,
)
from .display import amplitude_limit, horizontal_axis, prepare_wiggles, scaled_amplitudes

__all__ = [
    "DatasetSummary",
    "GatherWindow",
    "HeaderTable",
    "dataset_summary",
    "load_gather",
    "load_header_table",
    "amplitude_limit",
    "horizontal_axis",
    "prepare_wiggles",
    "scaled_amplitudes",
]
