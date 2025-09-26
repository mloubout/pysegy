"""
Minimal Python port of SegyIO.jl.
"""

# Verbose printing control (defaults to disabled)
VERBOSE: bool = False

def set_verbose(verbose: bool = False) -> None:
    """Enable or disable verbose output for pysegy.

    Parameters
    ----------
    verbose : bool, optional
        When True, print informational messages; defaults to False.
    """
    global VERBOSE
    VERBOSE = bool(verbose)

def vprint(*args, **kwargs) -> None:
    """Print only when verbose mode is enabled."""
    if VERBOSE:  # pragma: no cover - trivial branch
        print(*args, **kwargs)

from importlib.metadata import version, PackageNotFoundError

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
