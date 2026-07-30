"""
Reading utilities for the minimal Python SEGY implementation.
"""

from .types import (
    BinaryFileHeader,
    BinaryTraceHeader,
    FileHeader,
    SeisBlock,
    FH_BYTE2SAMPLE,
    TH_BYTE2SAMPLE,
)
from typing import BinaryIO, Iterable, List, Optional, Tuple
from concurrent.futures import Future, ProcessPoolExecutor
import multiprocessing
import numpy as np
from .utils import unpack_int, open_file
from .ibm import ibm_words_to_ieee
from . import vprint

# Number of traces to read at a time when loading an entire file
TRACE_CHUNKSIZE = 512


def _decode_trace_bytes(
    raw: bytes,
    ns: int,
    ntraces: int,
    datatype: int,
    keys: Optional[Iterable[str]],
    bigendian: bool,
) -> Tuple[List[BinaryTraceHeader], np.ndarray]:
    """Decode one self-contained trace chunk.

    This module-level function is intentionally pickleable so large reads can
    decode independent chunks in worker processes rather than GIL-bound
    threads.  File I/O stays in the parent process, which also supports remote
    and file-like objects that cannot be reopened by a worker.
    """
    if ntraces == 0:
        return [], np.empty((ns, 0), dtype=np.float32)

    trace_size = 240 + ns * 4
    expected = trace_size * ntraces
    if len(raw) != expected:
        raise EOFError(
            f"Expected {expected} bytes for {ntraces} traces, got {len(raw)}"
        )

    key_list = list(TH_BYTE2SAMPLE.keys()) if keys is None else list(keys)
    byteorder = ">" if bigendian else "<"
    header_columns = {}
    for key in key_list:
        offset, size = TH_BYTE2SAMPLE[key]
        dtype = np.dtype(f"{byteorder}i{size}")
        header_columns[key] = np.ndarray(
            (ntraces,), dtype=dtype, buffer=raw, offset=offset,
            strides=(trace_size,),
        )

    headers: List[BinaryTraceHeader] = []
    for idx in range(ntraces):
        hdr = BinaryTraceHeader()
        for key, values in header_columns.items():
            setattr(hdr, key, int(values[idx]))
        hdr.keys_loaded = key_list
        headers.append(hdr)

    sample_dtype = np.dtype(f"{byteorder}{'u4' if datatype == 1 else 'f4'}")
    samples = np.ndarray(
        (ntraces, ns), dtype=sample_dtype, buffer=raw, offset=240,
        strides=(trace_size, 4),
    )
    if datatype == 1:
        samples = ibm_words_to_ieee(samples)
    data = np.asarray(samples, dtype=np.float32).T.copy()
    return headers, data


def read_fileheader(
    f: BinaryIO, keys: Optional[Iterable[str]] = None, bigendian: bool = True
) -> FileHeader:
    """
    Read and parse the binary file header.

    Parameters
    ----------
    f : BinaryIO
        Open binary file handle.
    keys : Iterable[str], optional
        Header fields to read; by default all are loaded.
    bigendian : bool, optional
        ``True`` when the file is big-endian, ``False`` otherwise.

    Returns
    -------
    FileHeader
        Object containing the textual and binary headers.
    """
    if keys is None:
        keys = list(FH_BYTE2SAMPLE.keys())

    start = f.tell()
    f.seek(0)
    text_header = f.read(3600)

    bfh = BinaryFileHeader()
    for k in keys:
        offset = FH_BYTE2SAMPLE[k]
        # All file header fields are integers of 2 or 4 bytes
        size = 4 if k in ("Job", "Line", "Reel") else 2
        val_bytes = text_header[offset:offset + size]
        val = unpack_int(val_bytes, size, bigendian)
        setattr(bfh, k, val)
    bfh.keys_loaded = list(keys)
    f.seek(start)
    return FileHeader(text_header[:3200], bfh)


def read_traceheader(
    f: BinaryIO, keys: Optional[Iterable[str]] = None, bigendian: bool = True
) -> BinaryTraceHeader:
    """
    Read a single binary trace header from ``f``.

    Parameters
    ----------
    f : BinaryIO
        Open binary file handle positioned at a trace header.
    keys : Iterable[str], optional
        Header fields to read; all are loaded when omitted.
    bigendian : bool, optional
        ``True`` for big-endian encoding.

    Returns
    -------
    BinaryTraceHeader
        Parsed header object.
    """
    if keys is None:
        keys = list(TH_BYTE2SAMPLE.keys())

    hdr_bytes = f.read(240)
    th = BinaryTraceHeader()
    for k in keys:
        offset, size = TH_BYTE2SAMPLE[k]
        val = unpack_int(hdr_bytes[offset:offset + size], size, bigendian)
        setattr(th, k, val)
    th.keys_loaded = list(keys)
    return th


def read_traces(
    f: BinaryIO,
    ns: int,
    ntraces: int,
    datatype: int,
    keys: Optional[Iterable[str]] = None,
    bigendian: bool = True,
) -> Tuple[List[BinaryTraceHeader], np.ndarray]:
    """
    Read ``ntraces`` traces and their headers from ``f``.

    Parameters
    ----------
    f : BinaryIO
        Open file handle positioned at the first trace.
    ns : int
        Number of samples per trace.
    ntraces : int
        Number of traces to read.
    datatype : int
        SEGY data sample format code.
    keys : Iterable[str], optional
        Header fields to read for each trace.
    bigendian : bool, optional
        ``True`` for big-endian encoding.

    Returns
    -------
    tuple
        ``(headers, data)`` where ``headers`` is a list of
        :class:`BinaryTraceHeader` and ``data`` is ``ns`` x ``ntraces`` array.
    """
    trace_size = 240 + ns * 4
    raw = f.read(trace_size * ntraces)
    return _decode_trace_bytes(raw, ns, ntraces, datatype, keys, bigendian)


def read_file(
    f: BinaryIO,
    warn_user: bool = True,
    keys: Optional[Iterable[str]] = None,
    bigendian: bool = True,
    workers: int = 5,
) -> SeisBlock:
    """
    Read a complete SEGY file from an open file handle.

    Parameters
    ----------
    f : BinaryIO
        File object to read from.
    warn_user : bool, optional
        Currently unused.
    keys : Iterable[str], optional
        Additional header fields to load with each trace.
    bigendian : bool, optional
        Set ``True`` for big-endian encoding.
    workers : int, optional
        Number of processes used to decode independent trace chunks. Use
        ``1`` to disable multiprocessing. At most one chunk per worker is
        pending, bounding the additional memory used for large datasets.

    Returns
    -------
    SeisBlock
        Entire dataset loaded into memory. The file is read in chunks of
        ``TRACE_CHUNKSIZE`` traces to limit peak memory usage.
    """
    fh = read_fileheader(f, bigendian=bigendian)
    ns = fh.bfh.ns
    dsf = fh.bfh.DataSampleFormat
    trace_size = 240 + ns * 4
    f.seek(0, 2)
    end = f.tell()
    ntraces = (end - 3600) // trace_size
    f.seek(3600)
    headers: List[BinaryTraceHeader] = [BinaryTraceHeader() for _ in range(ntraces)]
    data: np.ndarray = np.zeros((ns, ntraces), dtype=np.float32)
    key_list = None if keys is None else list(keys)

    if workers < 1:
        raise ValueError("workers must be at least 1")

    def store_chunk(start: int, result) -> None:
        h, d = result
        count = len(h)
        for j in range(count):
            headers[start + j] = h[j]
        data[:, start:start + count] = d

    if workers == 1 or ntraces <= TRACE_CHUNKSIZE:
        idx = 0
        while idx < ntraces:
            count = min(TRACE_CHUNKSIZE, ntraces - idx)
            store_chunk(
                idx, read_traces(f, ns, count, dsf, key_list, bigendian)
            )
            idx += count
    else:
        pending: List[Tuple[int, Future]] = []
        # ``fork`` also permits library use from notebooks and ``python -c``
        # on POSIX. Windows only offers spawn and therefore retains Python's
        # usual requirement that multiprocessing entry points are guarded.
        methods = multiprocessing.get_all_start_methods()
        context = multiprocessing.get_context(
            "fork" if "fork" in methods else methods[0]
        )
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=context
        ) as pool:
            idx = 0
            while idx < ntraces:
                count = min(TRACE_CHUNKSIZE, ntraces - idx)
                raw = f.read(trace_size * count)
                future = pool.submit(
                    _decode_trace_bytes,
                    raw,
                    ns,
                    count,
                    dsf,
                    key_list,
                    bigendian,
                )
                pending.append((idx, future))
                idx += count

                # Limit queued bytes while keeping every process occupied.
                if len(pending) >= workers:
                    start, completed = pending.pop(0)
                    store_chunk(start, completed.result())

            for start, completed in pending:
                store_chunk(start, completed.result())

    return SeisBlock(fh, headers, data)


def segy_read(
    path: str,
    keys: Optional[Iterable[str]] = None,
    workers: int = 5,
    fs=None,
) -> SeisBlock:
    """
    Convenience wrapper to read a SEGY file.

    Parameters
    ----------
    path : str
        File system path to the SEGY file. When ``fs`` is provided the
        path is interpreted relative to that filesystem.
    fs : filesystem-like object, optional
        Filesystem providing ``open`` if reading from non-local storage.
    keys : Iterable[str], optional
        Additional header fields to load with each trace.

    Returns
    -------
    SeisBlock
        Loaded dataset.
    """
    vprint(f"Reading SEGY file {path}")

    with open_file(path, "rb", fs) as f:
        block = read_file(f, keys=keys, workers=workers)
    vprint(
        f"Loaded header ns={block.fileheader.bfh.ns} "
        f"dt={block.fileheader.bfh.dt} from {path}"
    )
    return block
