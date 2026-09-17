"""Local browser viewer for :mod:`pysegy` datasets.

The viewer is an optional component. Install it with ``pysegy[viewer]`` and
launch it with the ``pysegy-viewer`` command.
"""

from .services import (
    DatasetSummary,
    DatasetDiagnostics,
    DiagnosticCheck,
    GatherWindow,
    HeaderTable,
    FileHeaderInfo,
    dataset_diagnostics,
    dataset_summary,
    file_header_info,
    gather_summary_values,
    load_gather,
    load_header_table,
    open_native_file_dialog,
    receiver_attribute,
)
from .display import (
    amplitude_limit,
    horizontal_axis,
    plotly_colorscale,
    prepare_wiggle_fill,
    prepare_wiggles,
    scaled_amplitudes,
    selected_geometry_record,
    validate_comparison_records,
)

__all__ = [
    "DatasetSummary",
    "DatasetDiagnostics",
    "DiagnosticCheck",
    "GatherWindow",
    "HeaderTable",
    "FileHeaderInfo",
    "dataset_diagnostics",
    "dataset_summary",
    "file_header_info",
    "gather_summary_values",
    "load_gather",
    "load_header_table",
    "open_native_file_dialog",
    "receiver_attribute",
    "amplitude_limit",
    "horizontal_axis",
    "plotly_colorscale",
    "prepare_wiggle_fill",
    "prepare_wiggles",
    "scaled_amplitudes",
    "selected_geometry_record",
    "validate_comparison_records",
]
