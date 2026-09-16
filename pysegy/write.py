"""
Writing utilities for the minimal Python SEGY implementation.
"""

import struct
from typing import BinaryIO

from .utils import pack_int, struct_fmt, write_samples, open_file
from . import vprint
from .types import (
    SeisBlock,
    FileHeader,
    BinaryTraceHeader,
    FH_BYTE2SAMPLE,
    TH_BYTE2SAMPLE,
)


def write_fileheader(
    f: BinaryIO, fh: FileHeader, bigendian: bool = True
) -> None:
    """
    Write ``fh`` to ``f``.

    Parameters
    ----------
    f : BinaryIO
        Open binary file handle to write to.
    fh : FileHeader
        Header values to write.
    bigendian : bool, optional
        Write numbers in big-endian order when ``True``.
    """
    th = fh.th
    if len(th) < 3200:
        th = th + b" " * (3200 - len(th))
    f.write(th[:3200])
    bfh = fh.bfh
    # Placed at the offsets the fields are declared at, not written one
    # after another: the binary header is not a packed sequence, and the
    # standard leaves gaps in it.  Writing through those gaps put every
    # field past one at the wrong place, so `SegyFormatRevisionNumber`,
    # `FixedLengthTraceFlag` and the count of extended textual headers were
    # written where nothing reads them and came back as zero.
    buffer = bytearray(400)
    for key, offset in FH_BYTE2SAMPLE.items():
        size = 4 if key in ("Job", "Line", "Reel") else 2
        at = offset - 3200
        buffer[at:at + size] = pack_int(getattr(bfh, key), size, bigendian)
    f.write(bytes(buffer))


def write_traceheader(
    f: BinaryIO, th: BinaryTraceHeader, bigendian: bool = True
) -> None:
    """
    Write a single trace header to ``f``.

    Parameters
    ----------
    f : BinaryIO
        Open file handle for output.
    th : BinaryTraceHeader
        Trace header values to write.
    bigendian : bool, optional
        Write numbers in big-endian order when ``True``.
    """
    buf = bytearray(240)
    for key, (offset, size) in TH_BYTE2SAMPLE.items():
        val = getattr(th, key)
        struct.pack_into(struct_fmt(size, bigendian), buf, offset, val)
    f.write(buf)


def write_block(f: BinaryIO, block: SeisBlock, bigendian: bool = True) -> None:
    """
    Write an entire :class:`SeisBlock` to ``f``.

    Parameters
    ----------
    f : BinaryIO
        File handle where the block will be written.
    block : SeisBlock
        Data to serialise.
    bigendian : bool, optional
        Write numbers in big-endian order when ``True``.
    """
    write_fileheader(f, block.fileheader, bigendian)
    dsf = block.fileheader.bfh.DataSampleFormat
    for i, hdr in enumerate(block.traceheaders):
        trace = block.data[:, i]
        write_traceheader(f, hdr, bigendian)
        write_samples(f, trace, dsf, bigendian)


def segy_write(path: str, block: SeisBlock, fs=None) -> None:
    """
    Convenience wrapper to write ``block`` to ``path``.

    Parameters
    ----------
    path : str
        Destination file path. When ``fs`` is provided the path is
        interpreted relative to that filesystem.
    block : SeisBlock
        Dataset to write to disk.
    """
    vprint(f"Writing SEGY file {path}")

    with open_file(path, "wb", fs) as f:
        write_block(f, block)
    vprint(f"Finished writing {path}")
