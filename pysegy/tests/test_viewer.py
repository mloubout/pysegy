import os

import numpy as np
import pytest

from pysegy.viewer.services import (
    dataset_summary,
    load_gather,
    scan_local_dataset,
    source_geometry,
)


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


def test_scan_local_dataset_rejects_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        scan_local_dataset(str(tmp_path / "missing.segy"))
