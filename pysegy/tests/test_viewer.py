import os

import numpy as np
import pytest

from pysegy.viewer.services import (
    _decode_textual_header,
    dataset_diagnostics,
    dataset_summary,
    file_header_info,
    gather_summary_values,
    load_gather,
    load_header_table,
    receiver_attribute,
    scan_local_dataset,
    scan_local_dataset_cached,
    source_geometry,
)
from pysegy.viewer.cache import clear_cache, dataset_fingerprint, load_cached_scan
from pysegy.viewer.display import (
    ORANGE_BLACK_COLORSCALE,
    RTM_COLORSCALE,
    SEISMIC_COLORSCALE,
    anchored_colorscale,
    amplitude_limit,
    horizontal_axis,
    plotly_colorscale,
    prepare_wiggle_fill,
    prepare_wiggles,
    scaled_amplitudes,
    selected_geometry_record,
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


def test_dataset_diagnostics_are_bounded_and_actionable(scan):
    diagnostics = dataset_diagnostics(scan)
    assert diagnostics.files == 1
    assert diagnostics.coordinate_bounds == (400.0, 4200.0, 0.0, 0.0)
    checks = {check.check: check for check in diagnostics.checks}
    assert checks["Source coordinates"].status == "Pass"
    assert checks["Samples per trace"].status == "Pass"
    assert checks["Sample intervals [µs]"].details == "4000"
    assert checks["Water-depth summaries"].status == "Warning"


def test_file_header_information(scan):
    info = file_header_info(scan)
    assert len(info.textual.splitlines()) == 40
    assert info.binary["ns"] == 751
    assert info.binary["dt"] == 4000


def test_textual_header_decodes_ebcdic():
    cards = "".join(f"C{line:2d} EBCDIC HEADER".ljust(80) for line in range(1, 41))
    decoded = _decode_textual_header(cards.encode("cp500"))
    assert decoded.splitlines()[0].startswith("C 1 EBCDIC HEADER")
    assert len(decoded.splitlines()) == 40


def test_unreadable_textual_header_is_skipped():
    raw = bytes(range(256)) * 12 + bytes(128)
    assert _decode_textual_header(raw) is None
    assert _decode_textual_header(b"short") is None


def test_water_depth_geometry_values(scan):
    source_depth = gather_summary_values(scan, "SourceWaterDepth")
    group_depth = receiver_attribute(scan, 0, "GroupWaterDepth")
    assert source_depth.shape == (len(scan),)
    assert group_depth.shape == (scan[0].ntraces,)
    assert np.all(np.isfinite(source_depth))
    with pytest.raises(ValueError, match="Unknown trace header"):
        gather_summary_values(scan, "MissingWaterDepth")


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


def test_display_transformations_are_non_destructive(scan):
    window = load_gather(scan, 0, max_traces=12, max_samples=40)
    original = window.data.copy()
    transformed = scaled_amplitudes(
        window, time_gain=1.0, reverse_polarity=True
    )
    assert transformed.shape == original.shape
    assert np.array_equal(window.data, original)
    assert amplitude_limit(transformed, 95) > 0
    with pytest.raises(ValueError, match="Time gain"):
        scaled_amplitudes(window, time_gain=4)


def test_amplitude_limit_handles_empty_and_nonfinite_data():
    assert amplitude_limit(np.array([])) == 1.0
    assert amplitude_limit(np.array([0.0, np.nan, np.inf])) == 1.0
    with pytest.raises(ValueError, match="percentile"):
        amplitude_limit(np.array([1.0]), 0)


def test_plotly_colorscale_samples_rgba_colormap():
    def grayscale(values):
        return np.column_stack((values, values, values, np.ones_like(values)))

    scale = plotly_colorscale(grayscale, samples=3)
    assert scale == [
        [0.0, "rgb(0,0,0)"],
        [0.5, "rgb(128,128,128)"],
        [1.0, "rgb(255,255,255)"],
    ]
    with pytest.raises(ValueError, match="at least two"):
        plotly_colorscale(grayscale, samples=1)


def test_seismic_color_scales_are_complete_and_saturate_extremes():
    for scale in (SEISMIC_COLORSCALE, RTM_COLORSCALE, ORANGE_BLACK_COLORSCALE):
        assert scale[0][0] == 0.0
        assert scale[-1][0] == 1.0

    assert SEISMIC_COLORSCALE[0][1] == SEISMIC_COLORSCALE[1][1]
    assert ORANGE_BLACK_COLORSCALE[0][1] == ORANGE_BLACK_COLORSCALE[1][1]

    assert SEISMIC_COLORSCALE[4][0] == 0.5
    assert [0.5, "#f7f7f5"] in RTM_COLORSCALE
    assert all(color != "#050505" for _, color in RTM_COLORSCALE)
    assert RTM_COLORSCALE != SEISMIC_COLORSCALE
    rtm_positions = np.asarray([stop[0] for stop in RTM_COLORSCALE])
    assert np.allclose(rtm_positions, 1.0 - rtm_positions[::-1])
    colors = [color for _, color in RTM_COLORSCALE]
    channels = np.asarray([
        [int(color[index:index + 2], 16) for index in (1, 3, 5)]
        for color in colors
    ])
    assert np.all(np.ptp(channels[2:9], axis=1) <= 32)
    assert np.all(np.ptp(channels[[0, 1, 9, 10]], axis=1) > 60)
    luminances = []
    for color in colors:
        srgb = np.asarray([
            int(color[index:index + 2], 16) / 255.0
            for index in (1, 3, 5)
        ])
        linear = np.where(
            srgb <= 0.04045,
            srgb / 12.92,
            ((srgb + 0.055) / 1.055) ** 2.4,
        )
        luminances.append(float(linear @ [0.2126, 0.7152, 0.0722]))
    assert np.all(np.diff(luminances[:6]) > 0)
    assert np.all(np.diff(luminances[5:]) < 0)
    with pytest.raises(ValueError, match="span zero to one"):
        anchored_colorscale([(0.1, "black"), (1.0, "white")])
    with pytest.raises(ValueError, match="strictly increasing"):
        anchored_colorscale([
            (0.0, "black"), (0.0, "gray"), (1.0, "white")
        ])


def test_horizontal_axes_and_wiggle_limit(scan):
    window = load_gather(scan, 0, max_traces=30, max_samples=50)
    trace_axis, trace_label = horizontal_axis(window, "Trace number")
    receiver_axis, receiver_label = horizontal_axis(window, "Receiver X")
    assert trace_label == "Trace"
    assert receiver_label == "Receiver X"
    assert np.array_equal(trace_axis, window.trace_indices)
    assert np.array_equal(receiver_axis, window.receiver_coordinates[:, 0])

    wiggles = prepare_wiggles(
        window.data, trace_axis, window.time_seconds, max_traces=7
    )
    assert wiggles.traces.shape[1] <= 7
    assert wiggles.traces.shape[0] == window.data.shape[0]
    assert len(wiggles.positions) == wiggles.traces.shape[1]
    with pytest.raises(ValueError, match="Unknown horizontal axis"):
        horizontal_axis(window, "Offset")


def test_wiggle_fill_separates_positive_and_negative_amplitudes():
    position = 10.0
    trace = np.asarray([9.0, 10.0, 12.0, 8.0])
    time = np.asarray([0.0, 0.1, 0.2, 0.3])
    fill = prepare_wiggle_fill(trace, position, time)
    assert np.all(fill.positive_x >= position)
    assert np.all(fill.negative_x <= position)
    assert fill.positive_x[0] == fill.positive_x[-1] == position
    assert fill.negative_x[0] == fill.negative_x[-1] == position
    assert len(fill.time_seconds) == len(trace) + 2
    with pytest.raises(ValueError, match="matching vectors"):
        prepare_wiggle_fill(trace, position, time[:-1])


def test_geometry_selection_uses_only_gather_points():
    points = [
        {"curve_number": 0, "customdata": [4, 100]},
        {"curve_number": 1, "customdata": [999]},
    ]
    assert selected_geometry_record(points) == 4
    assert selected_geometry_record([points[1]]) is None
    assert selected_geometry_record([
        {"curve_number": 0, "customdata": ["not-an-index"]}
    ]) is None


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
