"""Framework-independent operations used by the local dataset viewer."""

from dataclasses import dataclass
import platform
from pathlib import Path
import shutil
import subprocess
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
WATER_DEPTH_FIELDS = ("SourceWaterDepth", "GroupWaterDepth")


def open_native_file_dialog() -> Optional[str]:
    """Open the operating system's file chooser and return its selection."""

    system = platform.system()
    if system == "Darwin":
        command = [
            "osascript",
            "-e",
            'POSIX path of (choose file with prompt "Open a SEG-Y file")',
        ]
    elif system == "Windows":
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$dialog = New-Object System.Windows.Forms.OpenFileDialog; "
                "$dialog.Filter = 'SEG-Y files (*.segy;*.sgy)|*.segy;*.sgy|"
                "All files (*.*)|*.*'; "
                "if ($dialog.ShowDialog() -eq 'OK') { $dialog.FileName }"
            ),
        ]
    elif shutil.which("zenity"):
        command = [
            "zenity",
            "--file-selection",
            "--title=Open a SEG-Y file",
            "--file-filter=SEG-Y files | *.segy *.sgy *.SEGY *.SGY",
            "--file-filter=All files | *",
        ]
    elif shutil.which("kdialog"):
        command = [
            "kdialog",
            "--getopenfilename",
            str(Path.home()),
            "SEG-Y files (*.segy *.sgy *.SEGY *.SGY)",
            "--title",
            "Open a SEG-Y file",
        ]
    else:
        raise RuntimeError(
            "No native file picker is available. Install zenity or kdialog, "
            "or enter the dataset path manually."
        )

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    selected = result.stdout.strip()
    return str(Path(selected).expanduser().resolve()) if selected else None


@dataclass(frozen=True)
class DatasetSummary:
    """Small, display-ready summary of a scanned dataset."""

    records: int
    traces: int
    samples_per_trace: int
    sample_interval_us: int
    sample_format: int


@dataclass(frozen=True)
class DiagnosticCheck:
    """One survey quality-control result for display in the viewer."""

    check: str
    status: str
    details: str


@dataclass(frozen=True)
class DatasetDiagnostics:
    """Bounded survey-level geometry and consistency diagnostics."""

    files: int
    coordinate_bounds: tuple[float, float, float, float]
    checks: tuple[DiagnosticCheck, ...]


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


@dataclass(frozen=True)
class FileHeaderInfo:
    """Display-ready textual and binary SEG-Y file headers."""

    textual: Optional[str]
    binary: Dict[str, int]


def _decode_textual_header(raw: bytes) -> Optional[str]:
    """Decode a plausible ASCII or EBCDIC header, or return ``None``."""

    candidates = []
    for encoding in ("ascii", "cp500"):
        decoded = raw.decode(encoding, errors="replace")
        printable = sum(char.isprintable() for char in decoded)
        text = sum(char.isalnum() or char in " .,:;_-/()[]" for char in decoded)
        candidates.append(((printable + text) / (2 * max(1, len(raw))), decoded))
    score, decoded = max(candidates, key=lambda candidate: candidate[0])
    if len(raw) != 3200 or score < 0.85:
        return None
    if sum(char.isalnum() for char in decoded) < 8:
        return None
    return "\n".join(
        decoded[index:index + 80].rstrip()
        for index in range(0, 3200, 80)
    )


def file_header_info(scan: SegyScan) -> FileHeaderInfo:
    """Return decoded textual and binary headers for a scanned dataset."""

    header = scan.fileheader
    return FileHeaderInfo(
        textual=_decode_textual_header(header.th),
        binary={name: int(value) for name, value in header.bfh.values.items()},
    )


def _normalize_dataset_path(path: str) -> str:
    """Normalize a user-supplied dataset path."""

    clean_path = path.strip()
    if (
        len(clean_path) >= 2
        and clean_path[0] == clean_path[-1]
        and clean_path[0] in {'"', "'"}
    ):
        clean_path = clean_path[1:-1].strip()
    if not clean_path:
        raise ValueError("Enter a SEG-Y file or directory path.")
    return str(Path(clean_path).expanduser().resolve())


def scan_local_dataset(
    path: str,
    *,
    pattern: Optional[str] = None,
    by_receiver: bool = False,
    threads: Optional[int] = None,
) -> SegyScan:
    """Validate and scan a local SEG-Y file or directory."""

    clean_path = _normalize_dataset_path(path)
    source = Path(clean_path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {clean_path}")
    if source.is_dir() and not pattern:
        pattern = "*.segy"
    return segy_scan(
        clean_path,
        file_key=pattern,
        keys=WATER_DEPTH_FIELDS,
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

    clean_path = _normalize_dataset_path(path)
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


def dataset_diagnostics(scan: SegyScan) -> DatasetDiagnostics:
    """Inspect scan metadata without reading trace samples or all receivers."""

    geometry = source_geometry(scan)
    locations = geometry[:, :2] if geometry.size else np.empty((0, 2))
    finite = np.all(np.isfinite(locations), axis=1)
    zero = np.all(locations == 0, axis=1)
    valid_locations = locations[finite & ~zero]
    unique_locations = np.unique(valid_locations, axis=0)
    duplicates = len(valid_locations) - len(unique_locations)
    sample_counts = sorted({int(record.ns) for record in scan.records})
    sample_intervals = sorted({int(record.dt) for record in scan.records})
    empty_records = sum(record.ntraces == 0 for record in scan.records)

    water_summary = {}
    for field in WATER_DEPTH_FIELDS:
        values = gather_summary_values(scan, field)
        water_summary[field] = (
            int(np.count_nonzero(~np.isfinite(values))),
            int(np.count_nonzero(np.isfinite(values) & (values != 0))),
        )

    def consistency(values: list[int], label: str) -> DiagnosticCheck:
        status = "Pass" if len(values) <= 1 else "Warning"
        details = ", ".join(map(str, values)) if values else "No values"
        return DiagnosticCheck(label, status, details)

    checks = (
        DiagnosticCheck(
            "Source coordinates",
            "Pass" if np.all(finite) and not np.any(zero) else "Warning",
            f"{np.count_nonzero(~finite):,} non-finite; "
            f"{np.count_nonzero(zero):,} all-zero locations",
        ),
        DiagnosticCheck(
            "Duplicate gather locations",
            "Pass" if duplicates == 0 else "Warning",
            f"{duplicates:,} duplicate locations",
        ),
        DiagnosticCheck(
            "Empty gathers",
            "Pass" if empty_records == 0 else "Warning",
            f"{empty_records:,} gathers without traces",
        ),
        consistency(sample_counts, "Samples per trace"),
        consistency(sample_intervals, "Sample intervals [µs]"),
        DiagnosticCheck(
            "Water-depth summaries",
            "Pass" if all(nonzero for _, nonzero in water_summary.values())
            else "Warning",
            "; ".join(
                f"{field}: {missing:,} missing, {nonzero:,} non-zero"
                for field, (missing, nonzero) in water_summary.items()
            ),
        ),
    )
    if valid_locations.size:
        bounds = (
            float(np.min(valid_locations[:, 0])),
            float(np.max(valid_locations[:, 0])),
            float(np.min(valid_locations[:, 1])),
            float(np.max(valid_locations[:, 1])),
        )
    else:
        bounds = (np.nan, np.nan, np.nan, np.nan)
    return DatasetDiagnostics(
        files=len(set(scan.paths)),
        coordinate_bounds=bounds,
        checks=checks,
    )


def source_geometry(scan: SegyScan) -> np.ndarray:
    """Return source or receiver gather locations as an ``N x 3`` array."""

    if not scan.records:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(scan.shots, dtype=np.float64)


def gather_summary_values(scan: SegyScan, field: str) -> np.ndarray:
    """Return the midpoint of a summarized header range for every gather."""

    if field not in TH_FIELDS:
        raise ValueError(f"Unknown trace header field: {field}")
    values = []
    for record in scan.records:
        limits = record.summary.get(field)
        values.append(np.nan if limits is None else sum(limits) / 2.0)
    return np.asarray(values, dtype=np.float64)


def receiver_attribute(scan: SegyScan, record_index: int, field: str) -> np.ndarray:
    """Return a scaled trace-header field for the selected gather."""

    if not 0 <= record_index < len(scan):
        raise IndexError(f"Record index {record_index} is out of range")
    if field not in TH_FIELDS:
        raise ValueError(f"Unknown trace header field: {field}")
    return scan[record_index].read_header_fields([field])[field]


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
