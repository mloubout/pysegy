import os

import numpy as np
import pytest

from pysegy.viewer.services import (
    dataset_summary,
    load_gather,
    load_header_table,
    scan_local_dataset,
    scan_local_dataset_cached,
    source_geometry,
)
from pysegy.viewer.cache import clear_cache, dataset_fingerprint, load_cached_scan


DATAFILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "overthrust_2D_shot_1_20.segy"
)


@pytest.fixture(scope="module")
def scan():
    return scan_local_dataset(DATAFILE, threads=1)


def test_scan_and_summarize_dataset(scan):
    summary = dataset_summary(scan)
    assert summary.records == 20
    assert summary.traces == 3300
    assert summary.samples_per_trace == 751
    assert summary.sample_interval_us == 4000
    assert source_geometry(scan).shape == (20, 3)


def test_load_gather_is_bounded(scan):
    window = load_gather(scan, 0, max_traces=20, max_samples=100)
    assert window.data.shape[0] <= 100
    assert window.data.shape[1] <= 20
    assert window.trace_indices.shape == (window.data.shape[1],)
    assert window.time_seconds.shape == (window.data.shape[0],)
    assert window.receiver_coordinates.shape == (window.data.shape[1], 3)
    assert np.all(np.diff(window.trace_indices) > 0)


def test_load_gather_validates_selection(scan):
    with pytest.raises(IndexError, match="out of range"):
        load_gather(scan, len(scan))
    with pytest.raises(ValueError, match="Trace window"):
        load_gather(scan, 0, trace_start=1, trace_stop=1)
    with pytest.raises(ValueError, match="Display limits"):
        load_gather(scan, 0, max_traces=0)


def test_load_header_table_is_scaled_and_bounded(scan):
    table = load_header_table(
        scan,
        0,
        ["SourceX", "GroupX", "Offset"],
        max_rows=10,
    )
    assert 0 < len(table.trace_indices) <= 10
    assert set(table.columns) == {"SourceX", "GroupX", "Offset"}
    assert all(
        len(values) == len(table.trace_indices)
        for values in table.columns.values()
    )


def test_load_header_table_validates_fields(scan):
    with pytest.raises(ValueError, match="Unknown trace header"):
        load_header_table(scan, 0, ["NotAHeader"])
    with pytest.raises(ValueError, match="row limit"):
        load_header_table(scan, 0, max_rows=0)


def test_scan_cache_roundtrip(tmp_path):
    first, first_hit = scan_local_dataset_cached(
        DATAFILE, threads=1, cache_dir=tmp_path
    )
    second, second_hit = scan_local_dataset_cached(
        DATAFILE, threads=1, cache_dir=tmp_path
    )
    assert not first_hit
    assert second_hit
    assert second.counts == first.counts
    assert clear_cache(tmp_path) == 1


def test_dataset_fingerprint_changes_with_file(tmp_path):
    path = tmp_path / "small.segy"
    path.write_bytes(b"one")
    before = dataset_fingerprint(str(path), pattern=None, by_receiver=False)
    path.write_bytes(b"different contents")
    after = dataset_fingerprint(str(path), pattern=None, by_receiver=False)
    assert before != after


def test_corrupt_cache_is_discarded(tmp_path):
    path = tmp_path / "broken.scan"
    path.write_bytes(b"not a pickled scan")
    assert load_cached_scan(path) is None
    assert not path.exists()


def test_scan_local_dataset_rejects_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        scan_local_dataset(str(tmp_path / "missing.segy"))
