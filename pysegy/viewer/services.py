"""Framework-independent operations used by the local dataset viewer."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from ..scan import SegyScan, segy_scan
from ..types import TH_FIELDS
from .cache import cache_path, load_cached_scan, store_cached_scan


DEFAULT_HEADER_FIELDS = (
    "TraceNumWithinFile",
    "FieldRecord",
    "TraceNumber",
    "SourceX",
    "SourceY",
    "GroupX",
    "GroupY",
    "Offset",
)


@dataclass(frozen=True)
class DatasetSummary:
    """Small, display-ready summary of a scanned dataset."""

    records: int
    traces: int
    samples_per_trace: int
    sample_interval_us: int
    sample_format: int


@dataclass(frozen=True)
class GatherWindow:
    """A bounded portion of a gather ready for interactive rendering."""

    data: np.ndarray
    trace_indices: np.ndarray
    time_seconds: np.ndarray
    receiver_coordinates: np.ndarray


@dataclass(frozen=True)
class HeaderTable:
    """Selected, scaled trace-header columns for a gather."""

    trace_indices: np.ndarray
    columns: Dict[str, np.ndarray]


def scan_local_dataset(
    path: str,
    *,
    pattern: Optional[str] = None,
    by_receiver: bool = False,
    threads: Optional[int] = None,
) -> SegyScan:
    """Validate and scan a local SEG-Y file or directory."""

    clean_path = str(Path(path).expanduser().resolve())
    source = Path(clean_path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {clean_path}")
    if source.is_dir() and not pattern:
        pattern = "*.segy"
    return segy_scan(
        clean_path,
        file_key=pattern,
        by_receiver=by_receiver,
        threads=threads,
    )


def scan_local_dataset_cached(
    path: str,
    *,
    pattern: Optional[str] = None,
    by_receiver: bool = False,
    threads: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> tuple[SegyScan, bool]:
    """Scan a local dataset, reusing an unchanged metadata scan when possible."""

    clean_path = str(Path(path).expanduser().resolve())
    source = Path(clean_path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {clean_path}")
    if source.is_dir() and not pattern:
        pattern = "*.segy"
    location = cache_path(
        clean_path,
        pattern=pattern,
        by_receiver=by_receiver,
        cache_dir=cache_dir,
    )
    cached = load_cached_scan(location)
    if cached is not None:
        return cached, True
    scan = scan_local_dataset(
        clean_path,
        pattern=pattern,
        by_receiver=by_receiver,
        threads=threads,
    )
    store_cached_scan(location, scan)
    return scan, False


def dataset_summary(scan: SegyScan) -> DatasetSummary:
    """Return the key dimensions and encoding details of ``scan``."""

    header = scan.fileheader.bfh
    return DatasetSummary(
        records=len(scan),
        traces=sum(scan.counts),
        samples_per_trace=header.ns,
        sample_interval_us=header.dt,
        sample_format=header.DataSampleFormat,
    )


def source_geometry(scan: SegyScan) -> np.ndarray:
    """Return source or receiver gather locations as an ``N x 3`` array."""

    if not scan.records:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(scan.shots, dtype=np.float64)


def load_header_table(
    scan: SegyScan,
    record_index: int,
    fields: Iterable[str] = DEFAULT_HEADER_FIELDS,
    *,
    max_rows: int = 5000,
) -> HeaderTable:
    """Read selected header columns with bounded rows for browser display."""

    if not 0 <= record_index < len(scan):
        raise IndexError(f"Record index {record_index} is out of range")
    if max_rows < 1:
        raise ValueError("Header row limit must be positive")
    selected = list(dict.fromkeys(fields))
    unknown = sorted(set(selected) - set(TH_FIELDS))
    if unknown:
        raise ValueError(f"Unknown trace header field(s): {', '.join(unknown)}")
    if not selected:
        return HeaderTable(np.zeros(0, dtype=np.int64), {})

    record = scan[record_index]
    columns = record.read_header_fields(selected)
    step = max(1, int(np.ceil(record.ntraces / max_rows)))
    indices = np.arange(0, record.ntraces, step)
    return HeaderTable(
        trace_indices=indices,
        columns={name: values[::step] for name, values in columns.items()},
    )


def load_gather(
    scan: SegyScan,
    record_index: int,
    *,
    trace_start: int = 0,
    trace_stop: Optional[int] = None,
    sample_start: int = 0,
    sample_stop: Optional[int] = None,
    max_traces: int = 500,
    max_samples: int = 2000,
) -> GatherWindow:
    """Read and downsample a safe display window from one gather.

    Contiguous traces are read once and strides are then selected in memory.
    The returned arrays never exceed ``max_traces`` by ``max_samples``.
    """

    if not 0 <= record_index < len(scan):
        raise IndexError(f"Record index {record_index} is out of range")
    if max_traces < 1 or max_samples < 1:
        raise ValueError("Display limits must be positive")

    record = scan[record_index]
    trace_stop = record.ntraces if trace_stop is None else trace_stop
    sample_stop = record.ns if sample_stop is None else sample_stop
    if not 0 <= trace_start < trace_stop <= record.ntraces:
        raise ValueError("Trace window is outside the selected gather")
    if not 0 <= sample_start < sample_stop <= record.ns:
        raise ValueError("Sample window is outside the selected gather")

    trace_step = max(1, int(np.ceil((trace_stop - trace_start) / max_traces)))
    sample_step = max(1, int(np.ceil((sample_stop - sample_start) / max_samples)))
    contiguous = record.read_data(traces=slice(trace_start, trace_stop))
    data = contiguous[sample_start:sample_stop:sample_step, ::trace_step]
    trace_indices = np.arange(trace_start, trace_stop, trace_step)
    time_seconds = (
        np.arange(sample_start, sample_stop, sample_step, dtype=np.float64)
        * record.dt
        * 1e-6
    )
    receiver_coordinates = record.rec_coordinates[
        trace_start:trace_stop:trace_step
    ]
    return GatherWindow(data, trace_indices, time_seconds, receiver_coordinates)
