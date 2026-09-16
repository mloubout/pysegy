import gzip
import io
import os
import shutil
from io import BytesIO
import urllib.error
import urllib.request

import fsspec
import numpy as np
import pytest

import pysegy as seg  # noqa: E402
from pysegy.ibm import ibm_to_ieee  # noqa: E402
from pysegy.types import FileHeader, BinaryTraceHeader, SeisBlock  # noqa: E402

DATAFILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "data",
    "overthrust_2D_shot_1_20.segy",
)


def test_read():
    block = seg.segy_read(DATAFILE)
    assert block.fileheader.bfh.ns == 751
    assert len(block.traceheaders) == 3300
    assert block.traceheaders[0].SourceX == 400
    assert block.traceheaders[0].GroupX == 100


def test_write_roundtrip(tmp_path):
    fh = FileHeader()
    fh.bfh.ns = 4
    fh.bfh.DataSampleFormat = 5
    headers = [BinaryTraceHeader() for _ in range(2)]
    for th in headers:
        th.ns = 4
        th.SourceX = 1234
    data = np.array([[float(i*j) for j in range(2)] for i in range(4)])
    block = SeisBlock(fh, headers, data)
    tmp = tmp_path / 'temp.segy'
    seg.segy_write(str(tmp), block)
    out = seg.segy_read(str(tmp))
    assert out.fileheader.bfh.ns == 4
    assert out.traceheaders[0].SourceX == 1234
    assert np.all(out.data == data)


def test_ibm_conversion():
    """
    Ensure IBM -> IEEE conversion works for known constant.
    """
    assert ibm_to_ieee(b"\x41\x10\x00\x00") == 1.0


def test_fileheader_io():
    """
    Round-trip a file header using in-memory bytes.
    """
    fh = FileHeader()
    fh.bfh.Job = 99
    fh.bfh.Line = 123
    buf = BytesIO()
    seg.write.write_fileheader(buf, fh)
    buf.seek(0)
    out = seg.read.read_fileheader(buf)
    assert out.bfh.Job == 99
    assert out.bfh.Line == 123
    assert len(buf.getvalue()) == 3600


def test_write_read_block_bytesio():
    """
    Write and read a simple block using BytesIO.
    """
    fh = FileHeader()
    fh.bfh.ns = 2
    fh.bfh.DataSampleFormat = 5
    headers = [BinaryTraceHeader() for _ in range(1)]
    headers[0].ns = 2
    data = np.array([[1.0], [2.0]])
    block = SeisBlock(fh, headers, data)
    bio = BytesIO()
    seg.write.write_block(bio, block)
    bio.seek(0)
    out = seg.read.read_file(bio)
    assert np.all(out.data == data)


def test_read_traces_rejects_truncated_input():
    with pytest.raises(EOFError, match="Expected 248 bytes"):
        seg.read.read_traces(BytesIO(bytes(247)), 2, 1, 5)


def test_read_traces_empty_input():
    headers, data = seg.read.read_traces(BytesIO(), 2, 0, 5)
    assert headers == []
    assert data.shape == (2, 0)
    assert data.dtype == np.float32


def test_read_traces_little_endian():
    header = bytearray(240)
    header[0:4] = (123).to_bytes(4, "little", signed=True)
    raw = bytes(header) + np.array([1.5, -2.0], dtype="<f4").tobytes()
    headers, data = seg.read.read_traces(
        BytesIO(raw), 2, 1, 5, keys=["TraceNumWithinLine"], bigendian=False
    )
    assert headers[0].TraceNumWithinLine == 123
    assert np.array_equal(data[:, 0], [1.5, -2.0])


def test_read_file_parallel_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(seg.read, "TRACE_CHUNKSIZE", 1)
    fh = FileHeader()
    fh.bfh.ns = 2
    fh.bfh.DataSampleFormat = 5
    headers = [BinaryTraceHeader() for _ in range(3)]
    for index, header in enumerate(headers):
        header.ns = 2
        header.SourceX = index + 10
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    path = tmp_path / "parallel.segy"
    seg.segy_write(str(path), SeisBlock(fh, headers, expected))

    with path.open("rb") as stream:
        actual = seg.read.read_file(stream, workers=2)

    assert [header.SourceX for header in actual.traceheaders] == [10, 11, 12]
    assert np.array_equal(actual.data, expected)


def test_read_file_rejects_invalid_workers():
    with pytest.raises(ValueError, match="workers must be at least 1"):
        seg.read.read_file(BytesIO(bytes(3600)), workers=0)


def test_read_with_filesystem():
    fs = fsspec.filesystem("file")
    block = seg.segy_read(DATAFILE, fs=fs)
    assert block.fileheader.bfh.ns == 751


def test_write_with_filesystem(tmp_path):
    fs = fsspec.filesystem("file")
    fh = FileHeader()
    fh.bfh.ns = 2
    fh.bfh.DataSampleFormat = 5
    hdr = BinaryTraceHeader()
    hdr.ns = 2
    hdr.SourceX = 111
    data = np.array([[1.0], [2.0]])
    block = SeisBlock(fh, [hdr], data)
    dest = tmp_path / "fsout.segy"
    seg.segy_write(str(dest), block, fs=fs)
    out = seg.segy_read(str(dest), fs=fs)
    assert out.fileheader.bfh.ns == 2
    assert out.traceheaders[0].SourceX == 111


BP_URL = (
    "http://s3.amazonaws.com/open.source.geoscience/"
    "open_data/bpmodel94/Model94_shots.segy.gz"
)

#: The bucket behind :data:`BP_URL` stopped serving it anonymously: the
#: object and a listing of its prefix both answer ``403 AccessDenied``, with
#: or without credentials, while the bucket itself still resolves.  Whether
#: the object was removed or only made private cannot be told from outside,
#: since S3 answers 403 rather than 404 to a caller that may not list.
#:
#: Marked expected-to-fail on that error alone rather than on any failure:
#: these tests check what the reader makes of a real survey, and a blanket
#: xfail would swallow a regression in that as readily as the download.  If
#: the data comes back they pass again and nothing has to be undone.
_BP_UNREACHABLE = pytest.mark.xfail(
    raises=urllib.error.HTTPError,
    reason=f"{BP_URL} answers 403 AccessDenied since 2026-09-16",
)


@_BP_UNREACHABLE
def test_bp_model_headers():
    """
    Download a portion of the BP model data and verify header values.
    """
    response = urllib.request.urlopen(BP_URL)
    with gzip.GzipFile(fileobj=response) as gz:
        data = gz.read(40000)
    fh = seg.read.read_fileheader(BytesIO(data))
    assert fh.bfh.dt == 4000
    assert fh.bfh.ns == 2000
    assert fh.bfh.DataSampleFormat == 1
    bio = BytesIO(data)
    bio.seek(3600)
    th = seg.read.read_traceheader(bio)
    assert th.ns == 2000
    assert th.SourceX == 0
    assert th.GroupX == 15


@_BP_UNREACHABLE
def test_bp_model_scan(tmp_path):
    """Download the full BP Model dataset and verify shot statistics.

    The dataset contains 278 distinct shot locations. Receiver counts vary
    between 240 and 480 per shot. This test ensures the reader can process the
    entire file and that these counts match the known reference values.
    """
    dest = tmp_path / "Model94_shots.segy"
    response = urllib.request.urlopen(BP_URL)
    with gzip.GzipFile(fileobj=response) as gz, open(dest, "wb") as f:
        shutil.copyfileobj(gz, f)

    scan = seg.segy_scan(str(dest))

    fh = scan.fileheader
    shots = scan.shots
    counts = scan.counts

    ns = fh.bfh.ns
    trace_size = 240 + ns * 4
    with open(dest, "rb") as f:
        f.seek(0, os.SEEK_END)
        total = (f.tell() - 3600) // trace_size

    assert total == int(sum(counts))
    assert len(shots) == 278
    assert int(min(counts)) == 240
    assert int(max(counts)) == 480

    hdrs = scan.read_headers(0, keys=["GroupX"])
    assert hdrs[0].GroupX == 15
    assert len(hdrs) == counts[0]


def test_scan_directory_pattern():
    data_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "data"
    )
    scan = seg.segy_scan(
        data_dir, "overthrust_2D_shot_*.segy", keys=["GroupX"]
    )
    assert isinstance(scan, seg.SegyScan)
    assert len(scan.shots) == 97
    assert len(set(scan.paths)) == 5
    idx = 0  # first shot across all files
    assert scan.paths[idx].endswith("overthrust_2D_shot_1_20.segy")
    assert scan.counts[idx] == 127
    assert scan.summary(idx)["GroupX"] == (100, 6400)


def test_scan_with_filesystem():
    fs = fsspec.filesystem("file")
    data_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "data"
    )
    scan = seg.segy_scan(
        data_dir,
        "overthrust_2D_shot_*.segy",
        keys=["GroupX"],
        fs=fs,
    )
    assert isinstance(scan, seg.SegyScan)
    assert len(scan.shots) == 97


def test_scan_unsorted_traces(tmp_path):
    """
    Ensure scanning handles files with interleaved shots.
    """
    fh = FileHeader()
    fh.bfh.ns = 1
    fh.bfh.DataSampleFormat = 5

    hdr1 = BinaryTraceHeader()
    hdr1.ns = 1
    hdr1.SourceX = 1
    hdr1.SourceY = 1
    hdr1.GroupX = 1

    hdr2 = BinaryTraceHeader()
    hdr2.ns = 1
    hdr2.SourceX = 2
    hdr2.SourceY = 2
    hdr2.GroupX = 2

    hdr3 = BinaryTraceHeader()
    hdr3.ns = 1
    hdr3.SourceX = 1
    hdr3.SourceY = 1
    hdr3.GroupX = 3

    headers = [hdr1, hdr2, hdr3]
    data = np.zeros((1, 3), dtype=np.float32)
    block = SeisBlock(fh, headers, data)
    tmp = tmp_path / "unsorted.segy"
    with open(tmp, "wb") as f:
        seg.write.write_block(f, block)

    scan = seg.segy_scan(str(tmp), keys=["GroupX"])
    assert len(scan.shots) == 2
    assert scan.shots[0] == (1.0, 1.0, 0.0)
    assert scan.counts == [2, 1]
    assert scan.summary(0)["GroupX"] == (1, 3)


def test_scan_by_receiver_gather(tmp_path):
    """
    Group traces by receiver location instead of source.
    """
    fh = FileHeader()
    fh.bfh.ns = 1
    fh.bfh.DataSampleFormat = 5

    h1 = BinaryTraceHeader()
    h1.ns = 1
    h1.SourceX = 1
    h1.SourceY = 1
    h1.GroupX = 5
    h1.GroupY = 0

    h2 = BinaryTraceHeader()
    h2.ns = 1
    h2.SourceX = 2
    h2.SourceY = 2
    h2.GroupX = 5
    h2.GroupY = 0

    h3 = BinaryTraceHeader()
    h3.ns = 1
    h3.SourceX = 3
    h3.SourceY = 3
    h3.GroupX = 10
    h3.GroupY = 0

    headers = [h1, h2, h3]
    data = np.zeros((1, 3), dtype=np.float32)
    block = SeisBlock(fh, headers, data)
    tmp = tmp_path / "rec_gather.segy"
    with open(tmp, "wb") as f:
        seg.write.write_block(f, block)

    scan = seg.segy_scan(str(tmp), by_receiver=True)
    assert len(scan.shots) == 2
    assert scan.counts == [2, 1]
    assert scan.shots[0] == (5.0, 0.0, 0.0)
    coords = scan[0].rec_coordinates
    assert coords.shape == (2, 3)
    assert tuple(coords[0]) == (1.0, 1.0, 0.0)


def _write_model(path, ntraces=8, ns=4):
    """
    Write a stacked file, e.g. a velocity model: no gathers, and every trace
    carrying its own coordinate.
    """
    fh = FileHeader()
    fh.bfh.ns = ns
    fh.bfh.DataSampleFormat = 5
    fh.bfh.TraceSorting = seg.scan.STACKED_SORTING

    headers = []
    for i in range(ntraces):
        hdr = BinaryTraceHeader()
        hdr.ns = ns
        hdr.SourceX = hdr.GroupX = hdr.CDPX = 10 * i
        headers.append(hdr)

    data = np.arange(ns * ntraces, dtype=np.float32).reshape(ns, ntraces)
    with open(path, "wb") as f:
        seg.write.write_block(f, SeisBlock(fh, headers, data))
    return data


def test_scan_stacked_file(tmp_path):
    """
    A file without gathers scans as a single record holding every trace.
    """
    tmp = tmp_path / "model.segy"
    data = _write_model(tmp, ntraces=8, ns=4)

    scan = seg.segy_scan(str(tmp), keys=["GroupX"])
    assert len(scan) == 1
    assert scan.counts == [8]
    assert scan[0].segments == [(3600, 8)]
    assert scan.summary(0)["GroupX"] == (0, 70)
    assert np.array_equal(scan[0].data, data)


def test_read_trace_range(tmp_path):
    """
    Reading a range of traces only reads that range, and matches a full read.
    """
    tmp = tmp_path / "model.segy"
    data = _write_model(tmp, ntraces=8, ns=4)
    record = seg.segy_scan(str(tmp))[0]

    assert np.array_equal(record.read_data(traces=slice(2, 5)), data[:, 2:5])
    assert np.array_equal(record.read_data(traces=slice(0, 1)), data[:, :1])
    assert np.array_equal(record.read_data(traces=slice(7, 8)), data[:, 7:])
    assert np.array_equal(record.read_data(), data)

    # Same through the scan, which also returns the matching headers
    block = seg.segy_scan(str(tmp)).read_data(0, traces=slice(2, 5))
    assert np.array_equal(block.data, data[:, 2:5])
    assert len(block.traceheaders) == 3
    assert block.traceheaders[0].GroupX == 20


def test_data_indexing(tmp_path):
    """
    Indexing `data` reads the traces it asks for, and nothing else.
    """
    tmp = tmp_path / "model.segy"
    data = _write_model(tmp, ntraces=8, ns=4)
    record = seg.segy_scan(str(tmp))[0]

    assert isinstance(record.data, seg.TraceData)
    assert record.data.shape == data.shape
    assert len(record.data) == data.shape[0]
    assert repr(record.data) == "TraceData(ns=4, traces=8)"
    assert np.array_equal(np.asarray(record.data), data)

    # A window of traces, of samples, or of both
    assert np.array_equal(record.data[:, 2:5], data[:, 2:5])
    assert np.array_equal(record.data[1:3], data[1:3])
    assert np.array_equal(record.data[1:3, 2:5], data[1:3, 2:5])

    # A single trace, a few of them, and a strided range
    assert np.array_equal(record.data[:, 3], data[:, 3])
    assert np.array_equal(record.data[:, [1, 5, 6]], data[:, [1, 5, 6]])
    assert np.array_equal(record.data[:, 0:8:2], data[:, 0:8:2])

    # Only the traces of the window are read
    reads = []
    record.read_data = lambda keys=None, traces=None: reads.append(traces) or data
    record.data[:, 2:5]
    assert reads == [slice(2, 5)]


def test_scan_threads_match_sequential(tmp_path):
    """
    Splitting one file across threads finds the same shots and segments.

    The chunk is small enough that gathers straddle the block boundaries, which
    is where the threaded blocks have to be stitched back together.
    """
    fh = FileHeader()
    fh.bfh.ns = 2
    fh.bfh.DataSampleFormat = 5

    headers = []
    for shot in range(16):
        for _ in range(5):
            hdr = BinaryTraceHeader()
            hdr.ns = 2
            hdr.SourceX = 100 * (shot + 1)
            hdr.SourceY = 7
            hdr.GroupX = 10 * shot
            headers.append(hdr)

    data = np.arange(2 * len(headers), dtype=np.float32).reshape(2, len(headers))
    tmp = tmp_path / "gathers.segy"
    with open(tmp, "wb") as f:
        seg.write.write_block(f, SeisBlock(fh, headers, data))

    reference = seg.segy_scan(str(tmp), keys=["GroupX"], chunk=3, threads=1)
    assert len(reference) == 16
    assert reference.counts == [5] * 16

    for threads in (2, 4, 8):
        scan = seg.segy_scan(str(tmp), keys=["GroupX"], chunk=3, threads=threads)
        assert scan.shots == reference.shots
        assert scan.counts == reference.counts
        assert [r.segments for r in scan.records] == [
            r.segments for r in reference.records
        ]
        assert [r.summary for r in scan.records] == [
            r.summary for r in reference.records
        ]
        assert np.array_equal(scan[3].data, reference[3].data)


def test_read_budget(tmp_path):
    """
    The block size follows what the filesystem reports, within bounds.
    """
    tmp = tmp_path / "model.segy"
    _write_model(tmp, ntraces=4, ns=4)

    budget = seg.scan._read_budget(str(tmp))
    assert seg.scan.DEFAULT_READ_BYTES <= budget <= seg.scan.MAX_READ_BYTES

    # Filesystems doing their own buffering, and paths that cannot be queried,
    # fall back to the default
    assert seg.scan._read_budget(str(tmp), fs=fsspec.filesystem("file")) == \
        seg.scan.DEFAULT_READ_BYTES
    assert seg.scan._read_budget(str(tmp_path / "absent.segy")) == \
        seg.scan.DEFAULT_READ_BYTES


def test_scan_read_blocks_are_bounded(tmp_path):
    """
    A chunk larger than the read budget is split, so long traces and many
    threads cannot grow what a scan holds in memory.
    """
    ns, ntraces = 8, 40
    tmp = tmp_path / "model.segy"
    _write_model(tmp, ntraces=ntraces, ns=ns)
    trace_size = 240 + ns * 4

    with open(tmp, "rb") as f:
        f.seek(3600)
        blocks = list(
            seg.scan._iter_trace_columns(
                f, 3600, ntraces, ns, ["SourceX"], chunk=1000,
                max_bytes=4 * trace_size,
            )
        )

    assert [found for _, _, found in blocks] == [4] * 10
    assert [base for base, _, _ in blocks] == [
        3600 + 4 * trace_size * i for i in range(10)
    ]
    # The traces are all still accounted for once the blocks are put together
    assert seg.segy_scan(str(tmp), chunk=1000).counts == [ntraces]


def test_record_traceheaders(tmp_path):
    """
    The headers of a record are read on first access and then kept.
    """
    tmp = tmp_path / "model.segy"
    _write_model(tmp, ntraces=4, ns=4)
    record = seg.segy_scan(str(tmp))[0]

    headers = record.traceheaders
    assert [h.GroupX for h in headers] == [0, 10, 20, 30]
    assert record.traceheaders is headers


def test_scan_stops_at_end_of_file(tmp_path):
    """
    Asking for more traces than the file holds stops at the last whole one.
    """
    ns, ntraces = 8, 4
    tmp = tmp_path / "model.segy"
    _write_model(tmp, ntraces=ntraces, ns=ns)
    trace_size = 240 + ns * 4

    # A file cut in the middle of its last trace keeps the whole ones only
    with open(tmp, "r+b") as f:
        f.truncate(3600 + (ntraces - 1) * trace_size + 100)

    with open(tmp, "rb") as f:
        f.seek(3600)
        blocks = list(
            seg.scan._iter_trace_columns(
                f, 3600, ntraces + 5, ns, ["SourceX"], chunk=2
            )
        )

    assert sum(found for _, _, found in blocks) == ntraces - 1
    assert seg.segy_scan(str(tmp)).counts == [ntraces - 1]


def test_read_exactly_handles_short_reads():
    """
    A file-like object answering in dribs still yields whole traces.
    """
    class Trickle(io.BytesIO):
        """Return at most seven bytes per read, as a slow stream would."""

        def read(self, size=-1):
            return super().read(min(size, 7) if size > 0 else size)

    stream = Trickle(b"abcdefghij")
    assert seg.scan._read_exactly(stream, 10) == b"abcdefghij"
    assert seg.scan._read_exactly(stream, 4) == b""

    ns, ntraces = 2, 3
    trace_size = 240 + ns * 4
    raw = bytes(range(256)) * ((trace_size * ntraces) // 256 + 1)
    blocks = list(
        seg.scan._iter_trace_columns(
            Trickle(raw[:trace_size * ntraces]), 0, ntraces, ns, ["SourceX"],
            chunk=2,
        )
    )
    assert sum(found for _, _, found in blocks) == ntraces


def test_split_traces():
    """
    Trace ranges cover every trace and stay at least one chunk long.
    """
    assert seg.scan._split_traces(10, 4, 1) == [(0, 10)]
    assert seg.scan._split_traces(10, 4, 4) == [(0, 5), (5, 5)]
    assert seg.scan._split_traces(3, 4, 4) == [(0, 3)]
    assert seg.scan._split_traces(0, 4, 4) == [(0, 0)]

    blocks = seg.scan._split_traces(1000, 16, 8)
    assert len(blocks) == 8
    assert sum(count for _, count in blocks) == 1000
    assert [start for start, _ in blocks] == [125 * i for i in range(8)]


def test_read_trace_range_segments(tmp_path):
    """
    Trace ranges spanning several segments of a gather.
    """
    scan = seg.segy_scan(DATAFILE)
    record = scan[0]
    full = record.read_data()

    assert np.array_equal(record.read_data(traces=slice(3, 20)), full[:, 3:20])
    assert record.read_data(traces=slice(0, 0)).shape == (record.ns, 0)


def test_save_and_load_scan(tmp_path):
    scan = seg.segy_scan(DATAFILE)
    dest = tmp_path / "scan.pkl"
    seg.save_scan(str(dest), scan)
    out = seg.load_scan(str(dest))
    assert isinstance(out, seg.SegyScan)
    assert out.shots == scan.shots
    assert out.counts == scan.counts


def test_save_and_load_scan_fs(tmp_path):
    fs = fsspec.filesystem("file")
    scan = seg.segy_scan(DATAFILE)
    dest = tmp_path / "scan_fs.pkl"
    seg.save_scan(str(dest), scan, fs=fs)
    out = seg.load_scan(str(dest), fs=fs)
    assert out.shots == scan.shots


def test_index_and_lazy_data():
    scan = seg.segy_scan(DATAFILE)
    rec = scan[0]
    assert rec.coordinates == scan.shots[0]
    assert rec._data is None
    rec.data
    assert rec.data is not None
    assert rec._data is not None
    assert rec.fileheader.bfh.ns == scan.fileheader.bfh.ns
    all_blocks = scan.data
    assert len(all_blocks) == len(scan.shots)


def test_rec_coordinates():
    scan = seg.segy_scan(DATAFILE)
    rec = scan[0]
    coords = rec.rec_coordinates
    assert coords.shape[0] == scan.counts[0]
    assert coords[0, 0] == pytest.approx(100.0)
    assert coords[0, 1] == pytest.approx(0.0)
    assert coords[0, 2] == pytest.approx(500.0)


def test_get_header_scaling():
    fh = FileHeader()
    fh.bfh.ns = 1
    fh.bfh.DataSampleFormat = 5

    h1 = BinaryTraceHeader()
    h1.ns = 1
    h1.SourceX = 10
    h1.RecSourceScalar = 2

    h2 = BinaryTraceHeader()
    h2.ns = 1
    h2.SourceX = 20
    h2.RecSourceScalar = -2

    h3 = BinaryTraceHeader()
    h3.ns = 1
    h3.SourceX = 5
    h3.RecSourceScalar = 1

    h4 = BinaryTraceHeader()
    h4.ns = 1
    h4.SourceX = 7
    h4.RecSourceScalar = 0

    headers = [h1, h2, h3, h4]
    block = SeisBlock(fh, headers, np.zeros((1, 4), dtype=np.float32))

    vals = seg.get_header(block, "SourceX")
    assert vals[:2] == [20, 10]
    assert vals[2:] == [5, 7]

    raw = seg.get_header(block, "SourceX", scale=False)
    assert raw == [10, 20, 5, 7]


def _coords_fixture(tmp_path, ntraces=12, ns=3):
    """A file whose receiver coordinates need scaling to read back."""
    fh = FileHeader()
    fh.bfh.ns = ns
    fh.bfh.DataSampleFormat = 5
    headers = []
    for index in range(ntraces):
        header = BinaryTraceHeader()
        header.ns = ns
        header.SourceX = 1000
        header.SourceY = 2000
        header.GroupX = 3000 + index
        header.GroupY = -4000 - index
        header.GroupWaterDepth = 10 + index
        # Exercise all three branches of the scalar rule.
        header.RecSourceScalar = (10, -100, 0)[index % 3]
        header.ElevationScalar = (-10, 0, 100)[index % 3]
        headers.append(header)
    data = np.arange(ns * ntraces, dtype=np.float32).reshape(ns, ntraces)
    path = tmp_path / "coords.segy"
    seg.segy_write(str(path), SeisBlock(fh, headers, data))
    return path, ntraces


def test_read_header_fields_matches_per_trace_read(tmp_path):
    """The column reader must agree with building a header per trace.

    ``rec_coordinates`` reads a handful of words for every trace of a shot;
    decoding a field at a time across the block is how the scan already does
    it, and the two must not disagree.
    """
    from pysegy.utils import get_header

    path, ntraces = _coords_fixture(tmp_path)
    scan = seg.segy_scan(str(path))
    record = scan[0]
    keys = ["GroupX", "GroupY", "GroupWaterDepth"]

    bulk = record.read_header_fields(keys)
    hdrs = record.read_headers(
        keys=keys + ["RecSourceScalar", "ElevationScalar"]
    )
    for key in keys:
        np.testing.assert_allclose(bulk[key], get_header(hdrs, key))


def test_rec_coordinates_apply_the_coordinate_scalar(tmp_path):
    """Scaled coordinates must match what get_header would have produced."""
    from pysegy.utils import get_header

    path, ntraces = _coords_fixture(tmp_path)
    scan = seg.segy_scan(str(path))
    record = scan[0]

    hdrs = record.read_headers(
        keys=["GroupX", "GroupY", record.rec_depth_key,
              "RecSourceScalar", "ElevationScalar"]
    )
    expected = np.column_stack((
        get_header(hdrs, "GroupX"),
        get_header(hdrs, "GroupY"),
        get_header(hdrs, record.rec_depth_key),
    )).astype(np.float32)

    np.testing.assert_allclose(record.rec_coordinates, expected)


def test_scan_keeps_the_receiver_coordinates(tmp_path):
    """The scan decodes them already; it must not throw them away.

    Reading them back from the file means transferring the samples they are
    interleaved with, which on a field survey is ~125 MB of file per shot to
    obtain ~150 KB of coordinates.
    """
    path, _ = _coords_fixture(tmp_path, ntraces=12)
    scan = seg.segy_scan(str(path))
    for record in scan.records:
        assert record._rec_coords is not None


def test_rec_coordinates_survive_the_file_going_away(tmp_path):
    """Nothing is read back: the scan already holds them.

    Deleting the data is the honest way to ask whether the file is touched.
    """
    path, ntraces = _coords_fixture(tmp_path, ntraces=12)
    scan = seg.segy_scan(str(path))
    expected = [np.array(rec.rec_coordinates) for rec in scan.records]
    os.remove(path)
    for rec, want in zip(scan.records, expected):
        np.testing.assert_allclose(rec.rec_coordinates, want)
    assert sum(len(v) for v in expected) == ntraces


def test_scanned_coordinates_match_the_headers(tmp_path):
    """What the scan kept must equal what reading the headers would give."""
    from pysegy.utils import get_header

    path, _ = _coords_fixture(tmp_path, ntraces=12)
    scan = seg.segy_scan(str(path))
    for record in scan.records:
        hdrs = record.read_headers(
            keys=["GroupX", "GroupY", record.rec_depth_key,
                  "RecSourceScalar", "ElevationScalar"]
        )
        expected = np.column_stack((
            get_header(hdrs, "GroupX"),
            get_header(hdrs, "GroupY"),
            get_header(hdrs, record.rec_depth_key),
        )).astype(np.float32)
        np.testing.assert_allclose(record.rec_coordinates, expected)


def test_scanned_coordinates_span_blocks_in_order(tmp_path):
    """A shot read over several blocks must come back whole and in order.

    The rows have to line up with the traces they came from, or every
    receiver of a large shot is attributed to the wrong position.
    """
    from pysegy.utils import get_header

    # One gather: a scalar that varies per trace would scale the source
    # position differently per trace and split them into separate records.
    ns, ntraces = 3, 40
    fh = FileHeader()
    fh.bfh.ns = ns
    fh.bfh.DataSampleFormat = 5
    headers = []
    for index in range(ntraces):
        header = BinaryTraceHeader()
        header.ns = ns
        header.SourceX, header.SourceY = 1000, 2000
        header.GroupX, header.GroupY = 3000 + index, -4000 - index
        header.GroupWaterDepth = 10 + index
        header.RecSourceScalar = -100
        header.ElevationScalar = -100
        headers.append(header)
    path = tmp_path / "span.segy"
    seg.segy_write(str(path), SeisBlock(
        fh, headers,
        np.arange(ns * ntraces, dtype=np.float32).reshape(ns, ntraces),
    ))

    scan = seg.segy_scan(str(path), chunk=3)
    record = scan.records[0]
    assert record.ntraces == ntraces

    hdrs = record.read_headers(
        keys=["GroupX", "RecSourceScalar", "ElevationScalar"]
    )
    np.testing.assert_allclose(
        record.rec_coordinates[:, 0], get_header(hdrs, "GroupX")
    )
