"""
Utility helpers shared across the :mod:`pysegy` package.
"""

from contextlib import contextmanager
from functools import lru_cache
import os
import struct
from typing import BinaryIO, Iterable, List, Tuple, Union

import numpy as np

from .types import BinaryTraceHeader, FH_BYTE2SAMPLE, SeisBlock, TH_BYTE2SAMPLE
from .ibm import ibm_to_ieee_array, ieee_to_ibm

_RECSRC_FIELDS = {
    "SourceX",
    "SourceY",
    "GroupX",
    "GroupY",
    "CDPX",
    "CDPY",
}

_ELEV_FIELDS = {
    "RecGroupElevation",
    "SourceSurfaceElevation",
    "SourceDepth",
    "RecDatumElevation",
    "SourceDatumElevation",
    "SourceWaterDepth",
    "GroupWaterDepth",
}

_SOURCE_DEPTH_FIELDS: Tuple[str, ...] = (
    # Likely-first: true source depth when present
    "SourceDepth",
    # Next prefer datum elevation over water depth; surface elevation after
    "SourceDatumElevation",
    "SourceSurfaceElevation",
    # Least likely candidate
    "SourceWaterDepth",
)

_RECEIVER_DEPTH_FIELDS: Tuple[str, ...] = (
    # Prefer receiver group elevation first
    "RecGroupElevation",
    "RecDatumElevation",
    # Least likely candidate
    "GroupWaterDepth",
)

_DEPTH_SENTINELS = {0, 2147483647, -2147483648, 32767, -32768}


def _check_scale(name: str) -> tuple[bool, str]:
    if name in _RECSRC_FIELDS:
        return True, "RecSourceScalar"
    if name in _ELEV_FIELDS:
        return True, "ElevationScalar"
    return False, ""


@lru_cache(maxsize=None)
def struct_fmt(size: int, bigendian: bool) -> str:
    """
    Return ``struct`` format for ``size`` byte integer.
    """
    return (">" if bigendian else "<") + ("i" if size == 4 else "h")


@lru_cache(maxsize=None)
def struct_obj(size: int, bigendian: bool) -> struct.Struct:
    """
    Return cached :class:`struct.Struct` instance for the given integer size.
    """
    return struct.Struct(struct_fmt(size, bigendian))


def unpack_int(buf: bytes, size: int, bigendian: bool) -> int:
    """
    Decode integer from ``buf`` with ``size`` bytes.
    """
    return struct_obj(size, bigendian).unpack(buf)[0]


def pack_int(value: int, size: int, bigendian: bool) -> bytes:
    """
    Encode ``value`` to bytes according to ``size`` and endianness.
    """
    return struct_obj(size, bigendian).pack(value)


def read_samples(buf: bytes, ns: int, datatype: int, bigendian: bool) -> np.ndarray:
    """
    Return ``ns`` samples from ``buf`` given the SEGY data type.
    """
    if datatype == 1:
        return ibm_to_ieee_array(buf, ns, bigendian)
    dtype = (">" if bigendian else "<") + "f4"
    return np.frombuffer(buf, dtype=dtype, count=ns)


def write_samples(
    f: BinaryIO, trace: Iterable[float], datatype: int, bigendian: bool
) -> None:
    """
    Write ``trace`` values to ``f`` according to the SEGY format.
    """
    if datatype == 1:
        f.write(b"".join(ieee_to_ibm(float(x)) for x in trace))
    else:
        fmt = (">" if bigendian else "<") + f"{len(trace)}f"
        f.write(struct.pack(fmt, *trace))


@contextmanager
def open_file(path: str, mode: str = "rb", fs=None):
    """
    Context manager opening ``path`` using ``fs`` when provided.
    """
    opener = fs.open if fs is not None else open
    with opener(path, mode) as fh:
        yield fh


def get_header(
    src: Union[SeisBlock, Iterable[BinaryTraceHeader]],
    name: str,
    *,
    scale: bool = True,
) -> List[float]:
    """
    Return values for ``name`` from ``src`` optionally applying scaling.
    """
    if isinstance(src, SeisBlock):
        headers = src.traceheaders
    else:
        headers = list(src)

    vals = [getattr(h, name) for h in headers]

    scalable, scale_name = _check_scale(name)
    if scale and scalable:
        scaled: List[float] = []
        for h, v in zip(headers, vals):
            fact = getattr(h, scale_name)
            if fact > 0:
                scaled.append(v * fact)
            elif fact < 0:
                scaled.append(v / abs(fact))
            else:
                scaled.append(v)
        return scaled
    return vals


def _depth_score(values: Iterable[float]) -> Tuple[int, float]:
    """
    Compute a simple quality score for a sequence of depth-like values.

    The score returns a tuple of ``(count, span)`` where ``count`` is the
    number of non-sentinel entries and ``span`` is the range of the data. This
    is retained for backwards compatibility but is no longer used to choose the
    preferred header field. Header selection now follows a fixed priority order
    and simply checks for the presence of valid (non-zero, non-sentinel) data.
    """
    cleaned: List[float] = []
    for val in values:
        fval = float(val)
        if fval in _DEPTH_SENTINELS:
            continue
        cleaned.append(fval)
    if not cleaned:
        return (0, 0.0)
    arr = np.asarray(cleaned, dtype=float)
    span = float(np.max(arr) - np.min(arr))
    if span == 0.0:
        span = float(np.max(np.abs(arr)))
    return (len(cleaned), span)


def detect_depth_keys(
    path: str,
    *,
    fs=None,
    max_traces: int = 32,
) -> Tuple[str, str]:
    """
    Inspect ``path`` and infer source and receiver depth header names.

    Parameters
    ----------
    path : str
        SEGY file to inspect.
    fs : optional
        Optional filesystem object used to open ``path``.
    max_traces : int, optional
        Maximum number of leading trace headers to inspect.

    Returns
    -------
    tuple[str, str]
        ``(source_key, receiver_key)`` header names. Defaults to
        ``("SourceDepth", "GroupWaterDepth")`` when no better match is
        found.
    """

    source_key = _SOURCE_DEPTH_FIELDS[0]
    receiver_key = _RECEIVER_DEPTH_FIELDS[0]

    ordered_keys = list(
        dict.fromkeys(
            list(_SOURCE_DEPTH_FIELDS)
            + list(_RECEIVER_DEPTH_FIELDS)
            + ["ElevationScalar", "RecSourceScalar"],
        )
    )

    with open_file(path, "rb", fs) as f:
        file_header_bytes = f.read(3600)
        ns_offset = FH_BYTE2SAMPLE["ns"]
        ns = max(struct.unpack_from(">h", file_header_bytes, ns_offset)[0], 0)
        step = ns * 4

        headers: List[BinaryTraceHeader] = []
        for _ in range(max_traces):
            hdr_bytes = f.read(240)
            if len(hdr_bytes) < 240:
                break
            th = BinaryTraceHeader()
            for key in ordered_keys:
                offset, size = TH_BYTE2SAMPLE[key]
                fmt = ">i" if size == 4 else ">h"
                val = struct.unpack_from(fmt, hdr_bytes, offset)[0]
                setattr(th, key, val)
            th.keys_loaded = list(ordered_keys)
            headers.append(th)
            if step:
                f.seek(step, os.SEEK_CUR)
        if not headers:
            return source_key, receiver_key

    def _has_valid(values: Iterable[float]) -> bool:
        """Return True when any non-sentinel, non-zero value is present."""
        for v in values:
            fval = float(v)
            if fval in _DEPTH_SENTINELS:
                continue
            if fval != 0.0:
                return True
        return False

    for candidate in _SOURCE_DEPTH_FIELDS:
        vals = get_header(headers, candidate)
        if _has_valid(vals):
            source_key = candidate
            break

    for candidate in _RECEIVER_DEPTH_FIELDS:
        vals = get_header(headers, candidate)
        if _has_valid(vals):
            receiver_key = candidate
            break

    return source_key, receiver_key


__all__ = [
    "get_header",
    "open_file",
    "read_samples",
    "write_samples",
    "struct_fmt",
    "pack_int",
    "unpack_int",
    "detect_depth_keys",
]
