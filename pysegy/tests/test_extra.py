import importlib
import importlib.metadata
import os
from io import BytesIO
from pathlib import Path
import tempfile

import fsspec
import numpy as np
import pytest

import pysegy as seg
from pysegy.types import FileHeader, BinaryTraceHeader, SeisBlock, BinaryFileHeader
from pysegy.utils import _SOURCE_DEPTH_FIELDS, _RECEIVER_DEPTH_FIELDS

DATAFILE = os.path.join(os.path.dirname(__file__),
                        "..", "..", "data", "overthrust_2D_shot_1_20.segy")


def test_package_not_found(monkeypatch):
    importerror = importlib.metadata.PackageNotFoundError()
    monkeypatch.setattr(importlib.metadata, "version",
                        lambda name: (_ for _ in ()).throw(importerror))
    importlib.reload(seg)
    assert seg.__version__ == "0+untagged"


def test_ibm_conversions():
    arr = np.array([1.0, -2.0, 0.0], dtype=np.float32)
    buf = b"".join(seg.ibm.ieee_to_ibm(float(x)) for x in arr)
    out = seg.ibm.ibm_to_ieee_array(buf, 3)
    assert out.shape == arr.shape
    assert seg.ibm.ieee_to_ibm(0.0) == b"\x00\x00\x00\x00"
    with pytest.raises(ValueError):
        seg.ibm.ibm_to_ieee(b"\x00\x00")
    assert seg.ibm.ibm_to_ieee(b"\x00\x00\x00\x00") == 0.0
    assert seg.ibm.ibm_to_ieee(0x41000000) == 0.0
    assert seg.ibm.ibm_to_ieee(0) == 0.0
    seg.ibm.ieee_to_ibm(-0.5)
    seg.ibm.ieee_to_ibm(32.0)
    seg.ibm.ieee_to_ibm(-0.5)


def test_get_header_no_scale():
    fh = FileHeader()
    fh.bfh.ns = 1
    th = BinaryTraceHeader()
    th.ns = 1
    block = SeisBlock(fh, [th], np.zeros((1, 1), dtype=np.float32))
    vals = seg.get_header(block, "ns")
    assert vals == [1]


def test_type_methods_roundtrip():
    bfh = BinaryFileHeader()
    bfh.keys_loaded = []
    bfh.Job = 5
    fh = FileHeader(bfh=bfh)
    th = BinaryTraceHeader()
    th.ns = 1
    block = SeisBlock(fh, [th], np.zeros((1, 1), dtype=np.float32))
    assert len(block) == 1
    assert "SeisBlock" in str(block)
    assert "BinaryFileHeader" in str(fh)
    assert "BinaryFileHeader" in str(bfh)
    assert "BinaryTraceHeader" in str(th)
    state = bfh.__getstate__()
    dup = BinaryFileHeader()
    dup.__setstate__(state)
    assert dup.values == bfh.values
    th_state = th.__getstate__()
    th2 = BinaryTraceHeader()
    th2.__setstate__(th_state)
    assert th2.values == th.values
    assert repr(th).startswith("BinaryTraceHeader")
    with pytest.raises(AttributeError):
        _ = bfh.nope
    with pytest.raises(AttributeError):
        BinaryTraceHeader().nope


def test_read_write_little_endian():
    fh = FileHeader()
    fh.th = b"HDR"
    fh.bfh.ns = 2
    fh.bfh.DataSampleFormat = 5
    th = BinaryTraceHeader()
    th.ns = 2
    th.SourceX = 42
    data = np.array([[1.0], [2.0]], dtype=np.float32)
    block = SeisBlock(fh, [th], data)
    bio = BytesIO()
    seg.write.write_block(bio, block, bigendian=False)
    bio.seek(0)
    out = seg.read.read_file(bio, bigendian=False)
    assert out.traceheaders[0].SourceX == 42
    np.testing.assert_allclose(out.data, data)


def test_read_write_ibm():
    fh = FileHeader()
    fh.bfh.ns = 1
    fh.bfh.DataSampleFormat = 1
    th = BinaryTraceHeader()
    th.ns = 1
    data = np.array([[3.0]], dtype=np.float32)
    block = SeisBlock(fh, [th], data)
    bio = BytesIO()
    seg.write.write_block(bio, block)
    bio.seek(0)
    out = seg.read.read_file(bio)
    assert out.data.shape == data.shape


def test_scan_utilities(tmp_path):
    fs = fsspec.filesystem("file")
    scan = seg.segy_scan(DATAFILE, keys=["GroupX", "Offset"], fs=fs)
    assert "SegyScan" in str(scan)
    rec = scan[0]
    assert "ShotRecord" in str(rec)
    assert len(scan) == len(scan.records)
    assert scan.offsets[0] >= 3600
    assert "summary" in str(rec)
    path = tmp_path / "scan.pkl"
    seg.save_scan(str(path), scan)
    loaded = seg.load_scan(str(path))
    assert len(loaded.records) == len(scan.records)


def test_scan_errors(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    fs = fsspec.filesystem("file")
    with pytest.raises(FileNotFoundError):
        seg.segy_scan(str(empty), fs=fs)


class TestDetectDepthKeys:
    @classmethod
    def setup_class(cls):
        cls._tmpdir = tempfile.TemporaryDirectory(prefix="pysegy-depth-")
        cls.tmpdir = Path(cls._tmpdir.name)

    @classmethod
    def teardown_class(cls):
        cls._tmpdir.cleanup()

    def _write_block(self, name: str, headers):
        fh = FileHeader()
        fh.bfh.ns = 1
        fh.bfh.DataSampleFormat = 5
        data = np.zeros((1, len(headers)), dtype=np.float32)
        block = SeisBlock(fh, headers, data)
        dest = self.tmpdir / name
        seg.segy_write(str(dest), block)
        return dest

    @pytest.mark.parametrize("source_key", _SOURCE_DEPTH_FIELDS)
    def test_source_candidates(self, source_key):
        headers = []
        for i in range(3):
            th = BinaryTraceHeader()
            th.ns = 1
            th.ElevationScalar = 1
            th.RecSourceScalar = 1
            for key in _SOURCE_DEPTH_FIELDS:
                setattr(th, key, 0)
            for key in _RECEIVER_DEPTH_FIELDS:
                setattr(th, key, 0)
            setattr(th, source_key, 100 + i)
            headers.append(th)

        dest = self._write_block(f"depth_source_{source_key}.segy", headers)
        detected_source, detected_receiver = seg.detect_depth_keys(str(dest))
        assert detected_source == source_key
        assert detected_receiver == "GroupWaterDepth"

    @pytest.mark.parametrize("receiver_key", _RECEIVER_DEPTH_FIELDS)
    def test_receiver_candidates(self, receiver_key):
        headers = []
        for i in range(3):
            th = BinaryTraceHeader()
            th.ns = 1
            th.ElevationScalar = 1
            th.RecSourceScalar = 1
            for key in _SOURCE_DEPTH_FIELDS:
                setattr(th, key, 0)
            th.SourceDepth = 50 + i
            for key in _RECEIVER_DEPTH_FIELDS:
                setattr(th, key, 0)
            setattr(th, receiver_key, 200 + i)
            headers.append(th)

        dest = self._write_block(f"depth_receiver_{receiver_key}.segy", headers)
        detected_source, detected_receiver = seg.detect_depth_keys(str(dest))
        assert detected_source == "SourceDepth"
        assert detected_receiver == receiver_key

    def test_defaults_when_empty(self):
        th = BinaryTraceHeader()
        th.ns = 1
        th.ElevationScalar = 1
        th.RecSourceScalar = 1
        dest = self._write_block("depth_default.segy", [th])
        source_key, receiver_key = seg.detect_depth_keys(str(dest))
        assert source_key == "SourceDepth"
        assert receiver_key == "GroupWaterDepth"

    def test_detect_depth_keys_with_no_traces(self):
        fh = FileHeader()
        dest = self.tmpdir / "depth_empty.segy"
        with open(dest, "wb") as f:
            seg.write.write_fileheader(f, fh)
        source_key, receiver_key = seg.detect_depth_keys(str(dest))
        assert source_key == "SourceDepth"
        assert receiver_key == "GroupWaterDepth"

    def test_segy_scan_uses_detected_keys(self):
        headers = []
        for i in range(4):
            th = BinaryTraceHeader()
            th.ns = 1
            th.SourceX = 1000
            th.SourceY = 2000
            th.GroupX = 3000 + i
            th.GroupY = 4000 + i
            for key in _SOURCE_DEPTH_FIELDS:
                setattr(th, key, 0)
            for key in _RECEIVER_DEPTH_FIELDS:
                setattr(th, key, 0)
            th.SourceWaterDepth = 150.0
            th.RecGroupElevation = 600.0
            th.ElevationScalar = 1
            th.RecSourceScalar = 1
            headers.append(th)

        dest = self._write_block("scan_auto_depth.segy", headers)
        scan = seg.segy_scan(str(dest))
        assert len(scan.records) == 1
        coords = scan.records[0].coordinates
        assert coords[2] == pytest.approx(150.0)
        rec_depths = scan.records[0].rec_coordinates
        assert rec_depths[0, 2] == pytest.approx(600.0)
