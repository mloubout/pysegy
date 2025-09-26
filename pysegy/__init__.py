"""
Minimal Python port of SegyIO.jl.
"""
from importlib.metadata import version, PackageNotFoundError

from .logger import vprint  # noqa

from .types import (
    BinaryFileHeader,
    BinaryTraceHeader,
    FileHeader,
    SeisBlock,
)
from .read import (
    read_fileheader,
    read_traceheader,
    read_file,
    segy_read,
)
from .scan import (
    ShotRecord,
    SegyScan,
    segy_scan,
    save_scan,
    load_scan
)
from .write import (
    write_fileheader,
    write_traceheader,
    write_block,
    segy_write,
)
from .utils import get_header
from .plotting import (
    plot_simage,
    plot_velocity,
    plot_fslice,
    plot_sdata,
    wiggle_plot,
    compare_shots,
)

__all__ = [
    "set_verbose",
    "BinaryFileHeader",
    "BinaryTraceHeader",
    "FileHeader",
    "SeisBlock",
    "SegyScan",
    "ShotRecord",
    "read_fileheader",
    "read_traceheader",
    "read_file",
    "segy_read",
    "segy_scan",
    "save_scan",
    "load_scan",
    "write_fileheader",
    "write_traceheader",
    "write_block",
    "segy_write",
    "get_header",
    "plot_simage",
    "plot_velocity",
    "plot_fslice",
    "plot_sdata",
    "wiggle_plot",
    "compare_shots",
]

try:
    __version__ = version("pysegy")
except PackageNotFoundError:
    # devito is not installed
    __version__ = '0+untagged'
