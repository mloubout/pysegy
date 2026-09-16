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

__all__ = [
    "DatasetSummary",
    "GatherWindow",
    "HeaderTable",
    "dataset_summary",
    "load_gather",
    "load_header_table",
]
