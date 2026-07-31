"""
Helpers for scanning SEGY files by shot location.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import fnmatch
import os
import struct
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
from .utils import detect_depth_keys, get_header, open_file

# Trace sorting code (binary file header) for horizontally stacked data, i.e.
# data that holds no gathers, such as a stack or a property model
STACKED_SORTING = 4


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


def _parse_header(buf: bytes, keys: Iterable[str]) -> BinaryTraceHeader:
    """
    Return a :class:`BinaryTraceHeader` parsed from ``buf``.

    Parameters
    ----------
    buf : bytes
        240-byte buffer containing the raw trace header.
    keys : Iterable[str]
        Header fields to decode from ``buf``.

    Returns
    -------
    BinaryTraceHeader
        Trace header populated with the requested fields.
    """
    th = BinaryTraceHeader()
    for k in keys:
        offset, size = TH_BYTE2SAMPLE[k]
        fmt = ">i" if size == 4 else ">h"
        val = struct.unpack_from(fmt, buf, offset)[0]
        setattr(th, k, val)
    th.keys_loaded = list(keys)
    return th


def _update_summary(
    summary: Dict[str, Tuple[float, float]],
    th: BinaryTraceHeader,
    keys: Iterable[str],
) -> None:
    """
    Update ``summary`` with values from ``th``.

    Parameters
    ----------
    summary : dict
        Mapping of header name to ``(min, max)`` tuple.
    th : BinaryTraceHeader
        Header providing new values.
    keys : Iterable[str]
        Header fields to include in the summary.
    """
    for k in keys:
        v = get_header([th], k)[0]
        if k in summary:
            mn, mx = summary[k]
            if v < mn:  # pragma: no cover
                mn = v
            if v > mx:
                mx = v
            summary[k] = (mn, mx)
        else:
            summary[k] = (v, v)


def _iter_trace_headers(
    f,
    start: int,
    count: int,
    ns: int,
    keys: Iterable[str],
    chunk: int = 1024,
) -> Iterable[Tuple[int, BinaryTraceHeader]]:
    """
    Yield offsets and headers from ``f`` starting at ``start``.

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
        Number of traces to read per block.

    Yields
    ------
    tuple
        ``(offset, header)`` for each trace encountered.
    """
    trace_size = 240 + ns * 4
    pos = start
    remaining = count
    while remaining > 0:
        n = min(chunk, remaining)
        buf = f.read(trace_size * n)
        for i in range(n):
            base = i * trace_size
            hdr = _parse_header(buf[base:base + 240], keys)
            yield pos + base, hdr
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


def _scan_file(
    path: str,
    keys: Optional[Iterable[str]] = None,
    chunk: int = 1024,
    depth_key: Optional[str] = None,
    rec_depth_key: Optional[str] = None,
    fs=None,
    by_receiver: bool = False,
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
        f.seek(3600)

        records: Dict[Tuple[int, int, int], ShotRecord] = {}

        previous: Optional[Tuple[int, int, int]] = None
        seg_start = 0
        seg_count = 0

        for offset, th in _iter_trace_headers(
            f,
            3600,
            total,
            ns,
            trace_keys,
            chunk,
        ):
            if by_receiver:
                src = (
                    np.float32(get_header([th], "GroupX")[0]),
                    np.float32(get_header([th], "GroupY")[0]),
                    np.float32(get_header([th], rec_depth_key)[0]),
                )
            else:
                src = (
                    np.float32(get_header([th], "SourceX")[0]),
                    np.float32(get_header([th], "SourceY")[0]),
                    np.float32(get_header([th], depth_key)[0]),
                )

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
            _update_summary(rec.summary, th, keys or [])

            # New segment begins when the source position changes
            if previous is None:
                previous = src
                seg_start = offset
                seg_count = 1
            elif src == previous:
                seg_count += 1
            else:
                records[previous].segments.append((seg_start, seg_count))
                previous = src
                seg_start = offset
                seg_count = 1

        if previous is not None:
            # Append the final segment for the last shot
            records[previous].segments.append((seg_start, seg_count))

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

    vprint(
        f"Scanning {len(files)} files in {directory} with {threads} threads"
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
