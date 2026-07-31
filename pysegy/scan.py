"""
Helpers for scanning SEGY files by shot location.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import fnmatch
import os
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cloudpickle
import numpy as np

from . import vprint
from .read import read_fileheader, read_traceheader, read_traces
from .types import (
    BinaryTraceHeader,
    FileHeader,
    SeisBlock,
    TH_BYTE2SAMPLE,
)
from .utils import _check_scale, detect_depth_keys, get_header, open_file

# Trace sorting code (binary file header) for horizontally stacked data, i.e.
# data that holds no gathers, such as a stack or a property model
STACKED_SORTING = 4

# Bounds on the block a scanning thread reads at once, see _read_budget. Only
# the headers of a block are kept, so these also cap what a scan holds in
# memory, no matter how long the traces are or how many threads read the file
DEFAULT_READ_BYTES = 2 * 1024 * 1024
MAX_READ_BYTES = 8 * 1024 * 1024


class TraceData:
    """
    The traces of a :class:`ShotRecord`, read when they are indexed.

    Indexing follows the ``(samples, traces)`` layout of a SEGY block, and only
    the traces that are asked for are read off the file::

        record.data[:, 100:200]     # reads a hundred traces
        record.data[0:600, 100:200]     # and keeps the first 600 samples
        np.asarray(record.data)      # reads them all

    A record too large to hold in memory can therefore be worked through window
    by window, and a distributed application can have every process read only
    the part of it that it owns.
    """

    def __init__(self, record):
        self.record = record
        self.shape = (record.ns, record.ntraces)
        self.dtype = np.float32

    def __array__(self, dtype=None, copy=None):
        values = self[:, :]
        return values if dtype is None else values.astype(dtype)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index):
        samples, traces = index if isinstance(index, tuple) else (index, slice(None))

        # Read the traces that are asked for, as a range of them, and let numpy
        # apply the rest of the indexing to what was read
        if isinstance(traces, slice):
            start, stop, step = traces.indices(self.shape[1])
            block = self.record.read_data(keys=(), traces=slice(start, stop))
            block = block[:, ::step] if step != 1 else block
        else:
            wanted = np.atleast_1d(traces)
            first, last = int(np.min(wanted)), int(np.max(wanted))
            block = self.record.read_data(keys=(), traces=slice(first, last + 1))
            block = block[:, wanted - first]
            if np.isscalar(traces) or np.ndim(traces) == 0:
                block = block[:, 0]

        return block[samples]

    def __repr__(self) -> str:
        return f"TraceData(ns={self.shape[0]}, traces={self.shape[1]})"


@dataclass
class ShotRecord:
    """
    Information about a single shot or receiver gather within a SEGY file.
    """

    path: str
    coordinates: Tuple[float, float, float]
    fileheader: FileHeader
    rec_depth_key: str = "RecGroupElevation"
    depth_key: str = "SourceDepth"
    by_receiver: bool = False
    segments: List[Tuple[int, int]] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    ns: int = 0
    dt: int = 0
    fs: Any = field(default=None, repr=False)
    _data: Optional["TraceData"] = field(default=None, init=False, repr=False)
    _headers: Optional[List[BinaryTraceHeader]] = field(
        default=None, init=False, repr=False
    )
    _rec_coords: Optional[np.ndarray] = field(
        default=None, init=False, repr=False
    )

    def __str__(self) -> str:
        lines = ["ShotRecord:"]
        lines.append(f"    path: {self.path}")
        lines.append(
            "    source: ("
            f"{self.coordinates[0]}, {self.coordinates[1]}, "
            f"{self.coordinates[2]}"
            ")"
        )
        lines.append(f"    traces: {self.ntraces}")
        lines.append(f"    ns: {self.ns}, dt: {self.dt}")
        if self.summary:
            lines.append("    summary:")
            for k, (mn, mx) in self.summary.items():
                lines.append(f"        {k:30s}: {mn}..{mx}")
        return "\n".join(lines)

    __repr__ = __str__

    @property
    def ntraces(self) -> int:
        """
        Number of traces in this record.
        """
        return sum(c for _, c in self.segments)

    def read_data(
        self,
        keys: Optional[Iterable[str]] = None,
        traces: Optional[slice] = None,
    ) -> SeisBlock:
        """
        Load the traces of this shot.

        Parameters
        ----------
        keys : Iterable[str], optional
            Header fields to load with each trace.
        traces : slice, optional
            Range of this record's traces to load. Only the corresponding part
            of the file is read; by default all traces are.

        Returns
        -------
        ndarray
            ``ns`` x number of traces read.
        """
        ns = self.fileheader.bfh.ns
        trace_size = 240 + ns * 4
        start, stop, _ = (traces or slice(None)).indices(self.ntraces)

        data_parts = []
        first = 0
        for offset, count in self.segments:
            # Part of this segment that falls within the requested range
            low, high = max(start, first), min(stop, first + count)
            if high > low:
                with open_file(self.path, "rb", self.fs) as f:
                    f.seek(offset + (low - first) * trace_size)
                    _, d = read_traces(
                        f,
                        ns,
                        high - low,
                        self.fileheader.bfh.DataSampleFormat,
                        keys,
                    )
                    data_parts.append(d)
            first += count
        if not data_parts:
            return np.empty((ns, 0), dtype=np.float32)
        return np.concatenate(data_parts, axis=1)

    def read_headers(
        self, keys: Optional[Iterable[str]] = None
    ) -> List[BinaryTraceHeader]:
        """
        Read only the headers for this shot.
        """
        headers: List[BinaryTraceHeader] = []
        ns = self.fileheader.bfh.ns
        for offset, count in self.segments:
            with open_file(self.path, "rb", self.fs) as f:
                f.seek(offset)
                for _ in range(count):
                    th = read_traceheader(f, keys)
                    headers.append(th)
                    f.seek(ns * 4, os.SEEK_CUR)
        return headers

    @property
    def data(self) -> "TraceData":
        """
        The traces of this record, read when they are indexed.
        """
        if self._data is None:
            self._data = TraceData(self)
        return self._data

    @property
    def traceheaders(self) -> List[List[BinaryTraceHeader]]:
        """
        Load trace headers for all shots on first access.
        """
        if self._headers is None:
            self._headers = self.read_headers()
        return self._headers

    @property
    def rec_coordinates(self) -> np.ndarray:
        """
        Array of receiver coordinates for this gather.
        """
        if self._rec_coords is None:
            if self.by_receiver:
                xname, yname, zname = "SourceX", "SourceY", self.depth_key
            else:
                xname, yname, zname = "GroupX", "GroupY", self.rec_depth_key

            hdrs = self.read_headers(
                keys=[
                    xname,
                    yname,
                    zname,
                    "RecSourceScalar",
                    "ElevationScalar",
                ]
            )
            gx = get_header(hdrs, xname)
            gy = get_header(hdrs, yname)
            dz = get_header(hdrs, zname)
            self._rec_coords = np.column_stack((gx, gy, dz)).astype(np.float32)
        return self._rec_coords


def _is_gathered(fh: FileHeader, records: List[ShotRecord]) -> bool:
    """
    Whether the scanned traces form gathers at all.

    Post-stack data, a velocity model or any other property cube has no shot
    structure: the binary file header says so when its trace sorting is
    ``STACKED_SORTING``, and grouping such a file by source coordinate yields
    single-trace groups, since every trace then carries its own coordinate.
    """
    if fh.bfh.TraceSorting == STACKED_SORTING:
        return False
    return any(r.ntraces > 1 for r in records)


def _merge_records(
    records: List[ShotRecord], fh: FileHeader, fs=None
) -> ShotRecord:
    """
    Merge scanned records into a single one holding every trace, in file order.
    """
    trace_size = 240 + fh.bfh.ns * 4
    first = min(records, key=lambda r: r.segments[0][0])

    segments: List[Tuple[int, int]] = []
    for offset, count in sorted(s for r in records for s in r.segments):
        if segments and offset == segments[-1][0] + segments[-1][1]*trace_size:
            segments[-1] = (segments[-1][0], segments[-1][1] + count)
        else:
            segments.append((offset, count))

    summary: Dict[str, Tuple[float, float]] = {}
    for rec in records:
        for k, (mn, mx) in rec.summary.items():
            if k in summary:
                summary[k] = (min(summary[k][0], mn), max(summary[k][1], mx))
            else:
                summary[k] = (mn, mx)

    return ShotRecord(
        first.path,
        first.coordinates,
        fh,
        first.rec_depth_key,
        first.depth_key,
        first.by_receiver,
        segments,
        summary,
        first.ns,
        first.dt,
        fs,
    )


def _decode_columns(
    buf: bytes, count: int, trace_size: int, keys: Iterable[str]
) -> Dict[str, np.ndarray]:
    """
    Decode ``keys`` for the ``count`` traces packed in ``buf``.

    Scanning only ever looks at a handful of fields, so the headers are decoded
    one field at a time across all traces instead of one trace at a time. The
    result is a column per field rather than a header object per trace, which
    keeps a whole-file scan out of Python's per-trace overhead.

    Parameters
    ----------
    buf : bytes
        Raw bytes holding ``count`` consecutive traces.
    count : int
        Number of traces contained in ``buf``.
    trace_size : int
        Size in bytes of one trace, header included.
    keys : Iterable[str]
        Header fields to decode.

    Returns
    -------
    dict
        Mapping of header name to an integer array with one entry per trace.
    """
    raw = np.frombuffer(buf, dtype=np.uint8, count=count * trace_size)
    raw = raw.reshape(count, trace_size)
    columns: Dict[str, np.ndarray] = {}
    for k in keys:
        offset, size = TH_BYTE2SAMPLE[k]
        dtype = ">i4" if size == 4 else ">i2"
        field_bytes = raw[:, offset:offset + size].copy()
        columns[k] = field_bytes.view(dtype).ravel().astype(np.int64)
    return columns


def _scaled_column(columns: Dict[str, np.ndarray], name: str) -> np.ndarray:
    """
    Return the values of ``name`` with the SEGY scalar applied.

    This is the vectorized counterpart of :func:`pysegy.utils.get_header`: a
    positive scalar multiplies the field, a negative one divides it, and zero
    leaves it untouched.
    """
    values = columns[name]
    scalable, scalar_name = _check_scale(name)
    if not scalable or scalar_name not in columns:
        return values
    factors = columns[scalar_name]
    scaled = values.astype(np.float64)
    positive = factors > 0
    negative = factors < 0
    scaled[positive] *= factors[positive]
    scaled[negative] /= np.abs(factors[negative])
    return scaled


def _update_summary(
    summary: Dict[str, Tuple[float, float]],
    values: Dict[str, np.ndarray],
    span: slice,
) -> None:
    """
    Widen ``summary`` with the ``span`` of traces held in ``values``.

    Parameters
    ----------
    summary : dict
        Mapping of header name to ``(min, max)`` tuple, updated in place.
    values : dict
        Mapping of header name to the scaled column of that field.
    span : slice
        Range of traces contributing to the record being summarised.
    """
    for k, column in values.items():
        block = column[span]
        low, high = block.min().item(), block.max().item()
        if k in summary:
            known_low, known_high = summary[k]
            summary[k] = (min(known_low, low), max(known_high, high))
        else:
            summary[k] = (low, high)


def _read_budget(path: str, fs=None) -> int:
    """
    Return how many bytes a scanning thread should read from ``path`` at once.

    Filesystems advertise the transfer size they are happiest with: APFS and
    NFS report a megabyte, most local Linux filesystems only their block size.
    Reading a couple of those at a time measures fastest, being large enough to
    stream yet small enough that readahead keeps running ahead of the scan
    instead of stalling it on one long request. The hint is clamped so that
    neither a tiny nor an enormous one decides how much memory a scan needs.

    Parameters
    ----------
    path : str
        File about to be scanned.
    fs : filesystem-like object, optional
        Filesystem the file lives on. Non-local ones do their own blocking and
        buffering, so the default is used for them.

    Returns
    -------
    int
        Block size in bytes, between ``DEFAULT_READ_BYTES`` and
        ``MAX_READ_BYTES``.
    """
    if fs is not None or not hasattr(os, "statvfs"):
        return DEFAULT_READ_BYTES
    try:
        hint = max(os.stat(path).st_blksize, os.statvfs(path).f_bsize)
    except OSError:
        return DEFAULT_READ_BYTES
    return min(max(2 * hint, DEFAULT_READ_BYTES), MAX_READ_BYTES)


def _read_exactly(f, size: int) -> bytes:
    """
    Read ``size`` bytes from ``f``, or what is left of it.

    A single ``read`` is allowed to come back short, which remote filesystems
    do, and a block that stopped in the middle of a trace would drop the rest
    of the file from the scan. Reading until the file gives nothing back keeps
    a short read from being mistaken for the end of the file.
    """
    buf = f.read(size)
    if len(buf) == size:
        return buf

    parts = [buf]
    got = len(buf)
    while got < size and buf:
        buf = f.read(size - got)
        parts.append(buf)
        got += len(buf)
    return b"".join(parts)


def _iter_trace_columns(
    f,
    start: int,
    count: int,
    ns: int,
    keys: Iterable[str],
    chunk: int = 1024,
    max_bytes: int = DEFAULT_READ_BYTES,
) -> Iterable[Tuple[int, Dict[str, np.ndarray], int]]:
    """
    Yield decoded header columns for blocks of traces read from ``f``.

    Parameters
    ----------
    f : file-like object
        Opened file positioned at ``start``.
    start : int
        Byte offset of the first trace.
    count : int
        Number of traces to read.
    ns : int
        Samples per trace.
    keys : Iterable[str]
        Header fields to decode.
    chunk : int, optional
        Number of traces to read per block. Blocks are shortened when that many
        traces would not fit in ``max_bytes``.
    max_bytes : int, optional
        Largest block to read at once, as returned by :func:`_read_budget`.

    Yields
    ------
    tuple
        ``(offset, columns, ntraces)`` for each block, where ``offset`` is the
        byte position of its first trace.
    """
    trace_size = 240 + ns * 4
    keys = list(keys)
    # A block holds whole traces, so files with long traces read fewer of them
    # at once: this is what bounds the memory a scan needs, per thread
    chunk = max(1, min(chunk, max_bytes // trace_size))
    pos = start
    remaining = count
    while remaining > 0:
        buf = _read_exactly(f, trace_size * min(chunk, remaining))
        n = len(buf) // trace_size
        if n == 0:
            # The file ends before the traces its size announced
            break
        columns = _decode_columns(buf, n, trace_size, keys)
        # Let the block go before the next one is read, so a scan holds one
        # block per thread rather than two
        buf = None
        yield pos, columns, n
        pos += n * trace_size
        remaining -= n


class SegyScan:
    """
    Representation of SEGY data grouped by shot.

    Parameters
    ----------
    fh : FileHeader
        File header shared by all scanned files.
    records : list of ShotRecord
        Collection of shot metadata describing trace segments.
    """

    def __init__(self, fh: FileHeader, records: List[ShotRecord], fs=None) -> None:
        """
        Create a new :class:`SegyScan` instance.

        Parameters
        ----------
        fh : FileHeader
            File header common to all files being scanned.
        records : list of ShotRecord
            Shot metadata describing trace segments.
        fs : filesystem-like object, optional
            Filesystem providing ``open`` for reading data lazily.
        """
        self.fileheader = fh
        self.records = records
        self.fs = fs
        self._data: Optional[List[SeisBlock]] = None

    def __len__(self) -> int:
        """
        Return the number of distinct shots.
        """
        return len(self.records)

    @property
    def paths(self) -> List[str]:
        """
        List of file paths corresponding to each shot.
        """
        return [r.path for r in self.records]

    @property
    def shots(self) -> List[Tuple[int, int, int]]:
        """
        Source coordinates for each shot including depth.
        """
        return [r.coordinates for r in self.records]

    @property
    def offsets(self) -> List[int]:
        """
        First trace byte offset for every shot.
        """
        return [r.segments[0][0] for r in self.records]

    @property
    def counts(self) -> List[int]:
        """
        Total number of traces for each shot.
        """
        return [r.ntraces for r in self.records]

    def __getitem__(self, idx: int) -> ShotRecord:
        """
        Return the ``idx``-th :class:`ShotRecord`.
        """
        return self.records[idx]

    @property
    def data(self) -> List[SeisBlock]:
        """
        Load data for all shots on first access.
        """
        if self._data is None:
            self._data = [self.read_data(i) for i in range(len(self.records))]
        return self._data

    def summary(self, idx: int) -> dict:
        """
        Header summaries for the ``idx``-th shot.
        """
        return self.records[idx].summary

    def read_data(
        self,
        idx: int,
        keys: Optional[Iterable[str]] = None,
        traces: Optional[slice] = None,
    ) -> SeisBlock:
        """
        Load the traces of a single shot.

        Parameters
        ----------
        idx : int
            Index of the shot to read.
        keys : Iterable[str], optional
            Additional header fields to load with each trace.
        traces : slice, optional
            Range of the shot's traces to load. Only the corresponding part of
            the file is read; by default all traces are.

        Returns
        -------
        SeisBlock
            In-memory representation of the selected traces.
        """
        rec = self.records[idx]
        ns = self.fileheader.bfh.ns
        trace_size = 240 + ns * 4
        start, stop, _ = (traces or slice(None)).indices(rec.ntraces)

        headers: List[BinaryTraceHeader] = []
        data_parts = []
        first = 0
        for offset, count in rec.segments:
            # Part of this segment that falls within the requested range
            low, high = max(start, first), min(stop, first + count)
            if high > low:
                fs_to_use = rec.fs if rec.fs is not None else getattr(self, "fs", None)
                with open_file(rec.path, "rb", fs_to_use) as f:
                    f.seek(offset + (low - first) * trace_size)
                    h, d = read_traces(
                        f,
                        ns,
                        high - low,
                        self.fileheader.bfh.DataSampleFormat,
                        keys,
                    )
                headers.extend(h)
                data_parts.append(d)
            first += count
        if data_parts:
            data = np.concatenate(data_parts, axis=1)
        else:
            data = np.empty((ns, 0), dtype=np.float32)  # pragma: no cover
        return SeisBlock(self.fileheader, headers, data)

    def read_headers(
        self, idx: int, keys: Optional[Iterable[str]] = None
    ) -> List[BinaryTraceHeader]:
        """
        Read only the headers for a single shot.

        Parameters
        ----------
        idx : int
            Shot index to read.
        keys : Iterable[str], optional
            Header fields to populate; by default all are read.

        Returns
        -------
        list of BinaryTraceHeader
            Parsed headers for the requested shot.
        """
        rec = self.records[idx]
        headers: List[BinaryTraceHeader] = []
        ns = self.fileheader.bfh.ns
        for offset, count in rec.segments:
            fs_to_use = rec.fs if rec.fs is not None else getattr(self, "fs", None)
            with open_file(rec.path, "rb", fs_to_use) as f:
                f.seek(offset)
                for _ in range(count):
                    th = read_traceheader(f, keys)
                    headers.append(th)
                    f.seek(ns * 4, os.SEEK_CUR)
        return headers

    def __str__(self) -> str:
        lines = ["SegyScan:"]
        lines.append(f"    shots: {len(self.records)}")
        lines.append(f"    ns: {self.fileheader.bfh.ns}")
        lines.append(f"    dt: {self.fileheader.bfh.dt}")
        return "\n".join(lines)

    __repr__ = __str__


def _split_traces(total: int, chunk: int, blocks: int) -> List[Tuple[int, int]]:
    """
    Split ``total`` traces into at most ``blocks`` contiguous ranges.

    A range is never shorter than ``chunk`` traces: below that the threads spend
    more time opening the file than reading it.

    Returns
    -------
    list of tuple
        ``(first trace, number of traces)`` for each range, in file order.
    """
    if total <= 0:
        return [(0, 0)]
    usable = max(1, min(blocks, total // max(chunk, 1)))
    size = -(-total // usable)
    return [
        (start, min(size, total - start))
        for start in range(0, total, size)
    ]


def _scan_range(
    path: str,
    start: int,
    count: int,
    ns: int,
    trace_keys: List[str],
    summary_keys: List[str],
    coord_keys: Tuple[str, str, str],
    chunk: int,
    fs=None,
) -> Tuple[List[Tuple[tuple, int, int]], Dict[tuple, dict]]:
    """
    Scan ``count`` traces of ``path`` starting at trace ``start``.

    Parameters
    ----------
    path : str
        SEGY file to scan.
    start : int
        Index of the first trace to look at.
    count : int
        Number of traces to scan.
    ns : int
        Samples per trace.
    trace_keys : list of str
        Header fields to decode.
    summary_keys : list of str
        Header fields to summarise.
    coord_keys : tuple of str
        Names of the two coordinates and the depth grouping traces into shots.
    chunk : int
        Number of traces to read per block.
    fs : filesystem-like object, optional
        Filesystem providing ``open`` if reading from non-local storage.

    Returns
    -------
    tuple
        ``(runs, summaries)`` where ``runs`` lists ``(coordinates, offset,
        ntraces)`` for every stretch of traces sharing a position, in file
        order, and ``summaries`` maps coordinates to their header ranges.
    """
    trace_size = 240 + ns * 4
    max_bytes = _read_budget(path, fs)
    runs: List[Tuple[tuple, int, int]] = []
    summaries: Dict[tuple, dict] = {}

    with open_file(path, "rb", fs) as f:
        f.seek(3600 + start * trace_size)
        for base, columns, found in _iter_trace_columns(
            f, 3600 + start * trace_size, count, ns, trace_keys, chunk, max_bytes
        ):
            coords = [
                _scaled_column(columns, k).astype(np.float32) for k in coord_keys
            ]
            summarised = {k: _scaled_column(columns, k) for k in summary_keys}

            # Traces sharing a position belong to one segment, so only the
            # traces where the position moves have to be looked at
            moved = np.ones(found, dtype=bool)
            moved[1:] = (
                (coords[0][1:] != coords[0][:-1])
                | (coords[1][1:] != coords[1][:-1])
                | (coords[2][1:] != coords[2][:-1])
            )
            starts = np.flatnonzero(moved)
            stops = np.append(starts[1:], found)

            for first, last in zip(starts.tolist(), stops.tolist()):
                src = (coords[0][first], coords[1][first], coords[2][first])
                if runs and runs[-1][0] == src:
                    # Same position as the tail of the previous block
                    runs[-1] = (src, runs[-1][1], runs[-1][2] + last - first)
                else:
                    runs.append((src, base + first * trace_size, last - first))
                if summary_keys:
                    _update_summary(
                        summaries.setdefault(src, {}),
                        summarised,
                        slice(first, last),
                    )

    return runs, summaries


def _records_from_runs(
    scanned: List[Tuple[List[Tuple[tuple, int, int]], Dict[tuple, dict]]],
    path: str,
    fh: FileHeader,
    depth_key: str,
    rec_depth_key: str,
    by_receiver: bool,
    fs=None,
) -> Dict[tuple, ShotRecord]:
    """
    Build the records of ``path`` from the runs found in each scanned block.

    ``scanned`` holds the ``(runs, summaries)`` of consecutive trace ranges, so
    a shot split across a block boundary is joined back into one segment here.
    """
    ns = fh.bfh.ns
    trace_size = 240 + ns * 4
    records: Dict[tuple, ShotRecord] = {}

    for runs, summaries in scanned:
        for src, offset, count in runs:
            rec = records.get(src)
            if rec is None:
                rec = ShotRecord(
                    path,
                    src,
                    fh,
                    depth_key if by_receiver else rec_depth_key,
                    rec_depth_key if by_receiver else depth_key,
                    by_receiver,
                    [],
                    {},
                    ns,
                    fh.bfh.dt,
                    fs,
                )
                records[src] = rec

            last_offset, last_count = rec.segments[-1] if rec.segments else (0, 0)
            if rec.segments and last_offset + last_count * trace_size == offset:
                rec.segments[-1] = (last_offset, last_count + count)
            else:
                rec.segments.append((offset, count))

        for src, summary in summaries.items():
            known = records[src].summary
            for name, (low, high) in summary.items():
                if name in known:
                    known[name] = (
                        min(known[name][0], low), max(known[name][1], high)
                    )
                else:
                    known[name] = (low, high)

    return records


def _scan_file(
    path: str,
    keys: Optional[Iterable[str]] = None,
    chunk: int = 1024,
    depth_key: Optional[str] = None,
    rec_depth_key: Optional[str] = None,
    fs=None,
    by_receiver: bool = False,
    threads: int = 1,
) -> SegyScan:
    """
    Scan ``path`` for shot locations.

    Parameters
    ----------
    path : str
        SEGY file to scan.
    keys : Iterable[str], optional
        Additional header fields to summarise.
    chunk : int, optional
        Number of traces to read at once.
    depth_key : str, optional
        Trace header field giving the source depth. When ``None`` the header is
        inferred by examining the initial trace headers in the file.
    rec_depth_key : str, optional
        Header field giving the receiver depth. When ``None`` the header is
        inferred alongside ``depth_key``.
    by_receiver : bool, optional
        Group traces by receiver coordinates instead of source coordinates.
    threads : int, optional
        Number of threads reading this file at once. The traces are split into
        that many contiguous ranges, one per thread.

    fs : filesystem-like object, optional
        Filesystem providing ``open`` if reading from non-local storage.

    Returns
    -------
    SegyScan
        Object describing all shots found in ``path``.
    """
    thread = threading.current_thread().name
    vprint(f"{thread} scanning file {path}")
    if depth_key is None or rec_depth_key is None:
        detected_source, detected_receiver = detect_depth_keys(path, fs=fs)
        if depth_key is None:
            depth_key = detected_source
        if rec_depth_key is None:
            rec_depth_key = detected_receiver

    vprint(
        f"{thread} using depth_key='{depth_key}' rec_depth_key='{rec_depth_key}'"
        f" for {path}"
    )

    trace_keys = [
        "SourceX",
        "SourceY",
        depth_key,
        "GroupX",
        "GroupY",
        rec_depth_key,
        "RecSourceScalar",
        "ElevationScalar",
    ]
    if keys is not None:
        for k in keys:
            if k not in trace_keys:
                trace_keys.append(k)

    with open_file(path, "rb", fs) as f:
        fh = read_fileheader(f)
        vprint(f"Header for {path}: ns={fh.bfh.ns} dt={fh.bfh.dt}")
        ns = fh.bfh.ns
        f.seek(0, os.SEEK_END)
        total = (f.tell() - 3600) // (240 + ns * 4)

    if by_receiver:
        coord_keys = ("GroupX", "GroupY", rec_depth_key)
    else:
        coord_keys = ("SourceX", "SourceY", depth_key)
    summary_keys = list(keys or [])

    blocks = _split_traces(total, chunk, max(threads, 1))
    vprint(f"{thread} reading {total} traces of {path} in {len(blocks)} blocks")
    if len(blocks) == 1:
        scanned = [
            _scan_range(
                path, blocks[0][0], blocks[0][1], ns, trace_keys, summary_keys,
                coord_keys, chunk, fs,
            )
        ]
    else:
        with ThreadPoolExecutor(max_workers=len(blocks)) as pool:
            futures = [
                pool.submit(
                    _scan_range, path, start, count, ns, trace_keys,
                    summary_keys, coord_keys, chunk, fs,
                )
                for start, count in blocks
            ]
            # Kept in block order so the segments stay in file order
            scanned = [fut.result() for fut in futures]

    records = _records_from_runs(
        scanned, path, fh, depth_key, rec_depth_key, by_receiver, fs
    )
    record_list = sorted(records.values(), key=lambda r: r.coordinates)
    if record_list and not _is_gathered(fh, record_list):
        vprint(f"{thread} {path} holds no gathers, scanned as a single record")
        record_list = [_merge_records(record_list, fh, fs)]

    vprint(f"{thread} found {len(record_list)} shots in {path}")
    return SegyScan(fh, record_list, fs=fs)


def segy_scan(
    path: str,
    file_key: Optional[str] = None,
    keys: Optional[Iterable[str]] = None,
    chunk: int = 1024,
    depth_key: Optional[str] = None,
    rec_depth_key: Optional[str] = None,
    threads: Optional[int] = None,
    fs=None,
    by_receiver: bool = False,
) -> SegyScan:
    """
    Scan one or more SEGY files and merge the results.

    Parameters
    ----------
    path : str
        Directory containing SEGY files or a single file path.
    file_key : str, optional
        Glob pattern selecting files within ``path``. When omitted and
        ``path`` points to a file, only that file is scanned.
    keys : Iterable[str], optional
        Additional header fields to summarise while scanning.
    chunk : int, optional
        Number of traces to read per block.
    depth_key : str, optional
        Header name containing the source depth. When ``None`` the field is
        detected automatically.
    rec_depth_key : str, optional
        Header field containing the receiver depth. When ``None`` the field is
        detected automatically.
    by_receiver : bool, optional
        When ``True``, traces are grouped by receiver coordinates rather than
        by source coordinates.
    fs : filesystem-like object, optional
        Filesystem providing ``open`` and ``glob`` if scanning non-local paths.

    Returns
    -------
    SegyScan
        Combined scan object describing all detected shots.
    """

    if threads is None:
        threads = os.cpu_count() or 1

    if file_key is None and (
        (fs is None and os.path.isfile(path)) or (fs and fs.isfile(path))
    ):
        files = [path]
        if fs is None:
            directory = os.path.dirname(path) or "."
        else:
            directory = getattr(fs, "_parent", os.path.dirname)(path)
    else:
        directory = path
        pattern = file_key or "*"
        if fs is None:
            files = [
                os.path.join(directory, fname)
                for fname in os.listdir(directory)
                if fnmatch.fnmatch(fname, pattern)
            ]
        else:
            files = fs.glob(f"{directory.rstrip('/')}/{pattern}")
    files.sort()

    # Threads left over once every file has one are used within the files
    per_file = max(1, threads // len(files)) if files else 1
    vprint(
        f"Scanning {len(files)} files in {directory} with {threads} threads"
        f" ({per_file} per file)"
    )
    records: List[ShotRecord] = []
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = {
            pool.submit(
                _scan_file,
                f,
                keys,
                chunk,
                depth_key,
                rec_depth_key,
                fs,
                by_receiver,
                per_file,
            ): f
            for f in files
        }
        for fut in as_completed(futures):
            scan = fut.result()
            fh = scan.fileheader
            records.extend(scan.records)

    if not records:
        raise FileNotFoundError("No matching SEGY files found")

    records.sort(key=lambda r: r.coordinates)

    vprint(f"Combined scan has {len(records)} shots")
    return SegyScan(fh, records, fs=fs)


def save_scan(path: str, scan: SegyScan, fs=None) -> None:
    """
    Serialize ``scan`` to ``path``.

    Parameters
    ----------
    path : str
        Destination file path. When ``fs`` is provided the path is interpreted
        relative to that filesystem.
    scan : SegyScan
        Object to serialize.
    fs : filesystem-like object, optional
        Filesystem providing ``open`` when writing to non-local storage.
    """
    vprint(f"Saving SegyScan to {path}")
    with open_file(path, "wb", fs) as f:
        cloudpickle.dump(scan, f, protocol=cloudpickle.DEFAULT_PROTOCOL)
    vprint(f"Finished saving {path}")


def load_scan(path: str, fs=None) -> SegyScan:
    """
    Load a :class:`SegyScan` previously saved with :func:`save_scan`.

    Parameters
    ----------
    path : str
        File system path of the saved object. When ``fs`` is provided the path
        is interpreted relative to that filesystem.
    fs : filesystem-like object, optional
        Filesystem providing ``open`` when reading from non-local storage.

    Returns
    -------
    SegyScan
        Deserialized scan object.
    """
    vprint(f"Loading SegyScan from {path}")
    with open_file(path, "rb", fs) as f:
        scan = cloudpickle.load(f)

    # When loading from external storage the filesystem won't be part of the
    # serialized object. Attach it so lazy reads work correctly.
    if fs is not None:
        scan.fs = fs
        for rec in scan.records:
            rec.fs = fs

    vprint(f"Loaded SegyScan with {len(scan.records)} shots")
    return scan
