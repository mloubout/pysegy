"""Local browser viewer for :mod:`pysegy` datasets.

The viewer is an optional component. Install it with ``pysegy[viewer]`` and
launch it with the ``pysegy-viewer`` command.
"""

from .services import (
    DatasetSummary,
    GatherWindow,
    HeaderTable,
    FileHeaderInfo,
    dataset_summary,
    file_header_info,
    gather_summary_values,
    load_gather,
    load_header_table,
    receiver_attribute,
)
from .display import (
    amplitude_limit,
    horizontal_axis,
    plotly_colorscale,
    prepare_wiggles,
    scaled_amplitudes,
)

__all__ = [
    "DatasetSummary",
    "GatherWindow",
    "HeaderTable",
    "FileHeaderInfo",
    "dataset_summary",
    "file_header_info",
    "gather_summary_values",
    "load_gather",
    "load_header_table",
    "receiver_attribute",
    "amplitude_limit",
    "horizontal_axis",
    "plotly_colorscale",
    "prepare_wiggles",
    "scaled_amplitudes",
]
