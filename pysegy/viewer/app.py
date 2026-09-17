"""Streamlit application for local SEG-Y exploration."""

import os

import colorcet as cc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pysegy.types import TH_FIELDS
from pysegy.viewer.cache import clear_cache
from pysegy.viewer.display import (
    ORANGE_BLACK_COLORSCALE,
    RTM_COLORSCALE,
    SEISMIC_COLORSCALE,
    amplitude_limit,
    horizontal_axis,
    plotly_colorscale,
    prepare_wiggles,
    scaled_amplitudes,
    selected_geometry_record,
)
from pysegy.viewer.services import (
    DEFAULT_HEADER_FIELDS,
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


st.set_page_config(page_title="pysegy Viewer", layout="wide")
st.title("pysegy Viewer")
st.caption("Explore local SEG-Y geometry and gathers without uploading your data.")

SEISMIC_COLOR_SCALES = {
    "Perceptual gray": plotly_colorscale(cc.cm.CET_L1),
    "Seismic": SEISMIC_COLORSCALE,
    "RTM": RTM_COLORSCALE,
    "Orange–black": ORANGE_BLACK_COLORSCALE,
}
WATER_DEPTH_COLOR_SCALE = plotly_colorscale(cc.cm.bgy)
TRACE_COUNT_COLOR_SCALE = plotly_colorscale(cc.cm.fire)
SIGNED_DEPTH_COLOR_SCALE = SEISMIC_COLOR_SCALES["Seismic"]

with st.sidebar:
    st.header("Dataset")
    path = st.text_input("SEG-Y file or directory", os.getenv("PYSEGY_VIEWER_PATH", ""))
    pattern = st.text_input("Directory pattern", "*.segy")
    by_receiver = st.toggle("Group by receiver", value=False)
    use_cache = st.toggle("Cache scan metadata", value=True)
    scan_clicked = st.button("Scan dataset", type="primary", width="stretch")
    if st.button("Clear scan cache", width="stretch"):
        removed = clear_cache()
        st.success(f"Removed {removed} cached scan file(s).")

if scan_clicked:
    try:
        with st.spinner("Scanning trace headers…"):
            file_pattern = (
                pattern if os.path.isdir(os.path.expanduser(path)) else None
            )
            if use_cache:
                scan, cache_hit = scan_local_dataset_cached(
                    path,
                    pattern=file_pattern,
                    by_receiver=by_receiver,
                )
            else:
                scan = scan_local_dataset(
                    path,
                    pattern=file_pattern,
                    by_receiver=by_receiver,
                )
                cache_hit = False
            st.session_state.scan = scan
            st.session_state.cache_hit = cache_hit
        st.session_state.record_index = 0
    except Exception as exc:
        st.error(f"Could not scan the dataset: {exc}")

scan = st.session_state.get("scan")
if scan is None:
    st.info("Enter a local SEG-Y path and select **Scan dataset** to begin.")
    st.stop()

summary = dataset_summary(scan)
cols = st.columns(5)
cols[0].metric("Gathers", f"{summary.records:,}")
cols[1].metric("Traces", f"{summary.traces:,}")
cols[2].metric("Samples / trace", f"{summary.samples_per_trace:,}")
cols[3].metric("Sample interval", f"{summary.sample_interval_us:,} µs")
cols[4].metric("Sample format", summary.sample_format)
if st.session_state.get("cache_hit"):
    st.caption("Loaded unchanged scan metadata from the local cache.")

pending_record = st.session_state.pop("pending_record_index", None)
if pending_record is not None:
    st.session_state.record_index = int(pending_record)
current_record = int(st.session_state.get("record_index", 0))
st.session_state.record_index = min(max(current_record, 0), len(scan) - 1)


def step_record(delta: int) -> None:
    """Move the shared gather selection from a widget callback."""

    selected = int(st.session_state.get("record_index", 0)) + delta
    st.session_state.record_index = min(max(selected, 0), len(scan) - 1)


previous_col, index_col, next_col = st.columns([1, 2, 1])
previous_col.button(
    "← Previous gather",
    disabled=st.session_state.record_index == 0,
    on_click=step_record,
    args=(-1,),
    width="stretch",
)
record_index = index_col.number_input(
    "Gather index",
    min_value=0,
    max_value=max(0, len(scan) - 1),
    step=1,
    key="record_index",
)
next_col.button(
    "Next gather →",
    disabled=st.session_state.record_index == len(scan) - 1,
    on_click=step_record,
    args=(1,),
    width="stretch",
)
record = scan[int(record_index)]

overview_tab, geometry_tab, gather_tab, headers_tab, diagnostics_tab = st.tabs(
    ["File headers", "Geometry", "Gather", "Trace headers", "Diagnostics"]
)

with overview_tab:
    header_info = file_header_info(scan)
    if header_info.textual:
        st.subheader("Textual file header")
        with st.container(border=True):
            st.text(header_info.textual)
    else:
        st.info("No readable ASCII or EBCDIC textual file header was found.")
    st.subheader("Binary file header")
    binary_frame = pd.DataFrame(
        header_info.binary.items(), columns=["Field", "Value"]
    )
    st.dataframe(binary_frame, width="stretch", hide_index=True)
    unique_paths = list(dict.fromkeys(scan.paths))
    st.caption(
        f"This combined scan contains {len(unique_paths):,} file(s). "
        "The displayed file header is the header retained by the scan."
    )

with geometry_tab:
    geometry = source_geometry(scan)
    frame = pd.DataFrame(geometry, columns=["x", "y", "depth"])
    frame["record"] = range(len(frame))
    frame["traces"] = scan.counts
    gather_color_label = st.selectbox(
        "Color gather locations by",
        ["Gather depth", "Trace count", "Source water depth", "Group water depth"],
    )
    gather_colors = {
        "Gather depth": frame["depth"],
        "Trace count": frame["traces"],
        "Source water depth": gather_summary_values(scan, "SourceWaterDepth"),
        "Group water depth": gather_summary_values(scan, "GroupWaterDepth"),
    }
    frame["color"] = gather_colors[gather_color_label]
    figure = px.scatter(
        frame,
        x="x",
        y="y",
        color="color",
        custom_data=["record", "traces"],
        hover_data=["record", "traces"],
        title="Gather locations",
        labels={"color": gather_color_label},
        color_continuous_scale=(
            TRACE_COUNT_COLOR_SCALE
            if gather_color_label == "Trace count"
            else (
                SIGNED_DEPTH_COLOR_SCALE
                if gather_color_label == "Gather depth"
                else WATER_DEPTH_COLOR_SCALE
            )
        ),
        color_continuous_midpoint=(
            0 if gather_color_label == "Gather depth" else None
        ),
    )
    receivers = record.rec_coordinates
    receiver_step = max(1, int(np.ceil(len(receivers) / 5000)))
    receivers = receivers[::receiver_step]
    receiver_color_label = st.selectbox(
        "Color selected receivers by",
        ["Receiver depth", "Source water depth", "Group water depth"],
    )
    if receiver_color_label == "Receiver depth":
        receiver_colors = receivers[:, 2]
    else:
        receiver_field = {
            "Source water depth": "SourceWaterDepth",
            "Group water depth": "GroupWaterDepth",
        }[receiver_color_label]
        receiver_colors = receiver_attribute(
            scan, int(record_index), receiver_field
        )[::receiver_step]
    figure.add_trace(
        go.Scattergl(
            x=receivers[:, 0],
            y=receivers[:, 1],
            mode="markers",
            name="Selected gather receivers",
            marker={
                "size": 6,
                "color": receiver_colors,
                "colorscale": WATER_DEPTH_COLOR_SCALE,
                "showscale": True,
                "colorbar": {"title": receiver_color_label, "x": 1.15},
            },
            customdata=receivers[:, 2],
            hovertemplate=(
                "x=%{x}<br>y=%{y}<br>depth=%{customdata}<extra></extra>"
            ),
        )
    )
    selected_source = geometry[int(record_index)]
    figure.add_trace(
        go.Scattergl(
            x=[selected_source[0]],
            y=[selected_source[1]],
            mode="markers",
            name="Selected gather",
            marker={
                "size": 16,
                "symbol": "circle-open",
                "color": "#202020",
                "line": {"width": 3},
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    geometry_event = st.plotly_chart(
        figure,
        width="stretch",
        key="geometry_selection",
        on_select="rerun",
        selection_mode="points",
    )
    clicked_record = selected_geometry_record(geometry_event.selection.points)
    if clicked_record is not None and clicked_record != int(record_index):
        st.session_state.pending_record_index = clicked_record
        st.rerun()
    if not geometry.size or not geometry[:, :2].any():
        st.warning("No non-zero source or receiver coordinates were detected.")

    st.subheader("Water-depth profile")
    water_field = st.selectbox(
        "Water-depth field",
        ["SourceWaterDepth", "GroupWaterDepth"],
    )
    water_depth = receiver_attribute(scan, int(record_index), water_field)
    profile = pd.DataFrame({
        "Receiver X": record.rec_coordinates[:, 0],
        "Water depth": water_depth,
        "Trace": np.arange(record.ntraces),
    })
    profile_figure = px.scatter(
        profile,
        x="Receiver X",
        y="Water depth",
        color="Water depth",
        hover_data=["Trace"],
        title=f"Selected gather — {water_field}",
        color_continuous_scale=WATER_DEPTH_COLOR_SCALE,
    )
    profile_figure.update_yaxes(autorange="reversed")
    st.plotly_chart(profile_figure, width="stretch")
    if not np.any(water_depth):
        st.info(f"The selected gather contains no non-zero {water_field} values.")

with gather_tab:
    control_a, control_b = st.columns(2)
    trace_range = control_a.slider(
        "Trace range",
        0,
        record.ntraces,
        (0, record.ntraces),
    )
    sample_range = control_b.slider(
        "Sample range",
        0,
        record.ns,
        (0, record.ns),
    )
    display_a, display_b, display_c = st.columns(3)
    display_mode = display_a.selectbox("Display", ["Image", "Wiggle"])
    axis_mode = display_b.selectbox(
        "Horizontal axis", ["Trace number", "Receiver X"]
    )
    color_scale = display_c.selectbox(
        "Perceptual color scale", list(SEISMIC_COLOR_SCALES),
        disabled=display_mode == "Wiggle",
    )
    gain_a, gain_b, gain_c, size_control = st.columns(4)
    percentile = gain_a.slider("Amplitude percentile", 80, 100, 99)
    time_gain = gain_b.slider("Time gain", 0.0, 3.0, 0.0, 0.25)
    reverse_polarity = gain_c.toggle("Reverse polarity", value=False)
    plot_height = size_control.slider(
        "Plot height",
        min_value=600,
        max_value=1200,
        value=850,
        step=50,
        help="Increase the vertical canvas for long seismic records.",
    )
    try:
        window = load_gather(
            scan,
            int(record_index),
            trace_start=trace_range[0],
            trace_stop=trace_range[1],
            sample_start=sample_range[0],
            sample_stop=sample_range[1],
        )
        amplitudes = scaled_amplitudes(
            window,
            time_gain=time_gain,
            reverse_polarity=reverse_polarity,
        )
        x_values, x_title = horizontal_axis(window, axis_mode)
        if display_mode == "Image":
            limit = amplitude_limit(amplitudes, percentile)
            figure = go.Figure(
                go.Heatmap(
                    z=amplitudes,
                    x=x_values,
                    y=window.time_seconds,
                    colorscale=SEISMIC_COLOR_SCALES[color_scale],
                    zmin=-limit,
                    zmax=limit,
                    colorbar={"title": "Amplitude"},
                )
            )
        else:
            wiggles = prepare_wiggles(
                amplitudes,
                x_values,
                window.time_seconds,
            )
            figure = go.Figure()
            for index, position in enumerate(wiggles.positions):
                figure.add_trace(go.Scattergl(
                    x=wiggles.traces[:, index],
                    y=wiggles.time_seconds,
                    mode="lines",
                    line={"color": "black", "width": 1},
                    name=f"{position:g}",
                    hovertemplate=(
                        f"position={position:g}<br>time=%{{y:.4f}} s<extra></extra>"
                    ),
                    showlegend=False,
                ))
        figure.update_layout(
            autosize=True,
            height=plot_height,
            margin={"l": 65, "r": 35, "t": 35, "b": 60},
            xaxis_title=x_title,
            yaxis_title="Time [s]",
        )
        figure.update_yaxes(autorange="reversed")
        st.plotly_chart(figure, width="stretch")
        st.caption(
            f"Displaying {window.data.shape[1]:,} traces × "
            f"{window.data.shape[0]:,} samples. Large selections are downsampled."
        )
    except Exception as exc:
        st.error(f"Could not load this gather: {exc}")

with headers_tab:
    fields = st.multiselect(
        "Header fields",
        TH_FIELDS,
        default=list(DEFAULT_HEADER_FIELDS),
    )
    try:
        table = load_header_table(scan, int(record_index), fields)
        header_frame = pd.DataFrame(table.columns)
        header_frame.insert(0, "Trace", table.trace_indices)
        st.dataframe(header_frame, width="stretch", hide_index=True)
        st.download_button(
            "Download displayed headers as CSV",
            header_frame.to_csv(index=False),
            file_name=f"gather-{int(record_index)}-headers.csv",
            mime="text/csv",
        )
        if fields:
            histogram_field = st.selectbox("Histogram field", fields)
            histogram = px.histogram(header_frame, x=histogram_field)
            st.plotly_chart(histogram, width="stretch")
        if len(header_frame) < record.ntraces:
            st.caption(
                f"Showing {len(header_frame):,} evenly sampled rows from "
                f"{record.ntraces:,} traces."
            )
    except Exception as exc:
        st.error(f"Could not load trace headers: {exc}")

with diagnostics_tab:
    diagnostics = dataset_diagnostics(scan)
    bounds = diagnostics.coordinate_bounds
    diagnostic_cols = st.columns(3)
    diagnostic_cols[0].metric("Files", f"{diagnostics.files:,}")
    diagnostic_cols[1].metric("Source X range", f"{bounds[0]:g} to {bounds[1]:g}")
    diagnostic_cols[2].metric("Source Y range", f"{bounds[2]:g} to {bounds[3]:g}")
    diagnostic_frame = pd.DataFrame([
        {
            "Check": check.check,
            "Status": check.status,
            "Details": check.details,
        }
        for check in diagnostics.checks
    ])
    st.dataframe(diagnostic_frame, width="stretch", hide_index=True)
    warning_count = sum(check.status == "Warning" for check in diagnostics.checks)
    if warning_count:
        st.warning(
            f"{warning_count} diagnostic check(s) need review. "
            "Warnings can reflect valid acquisition geometry as well as bad headers."
        )
    else:
        st.success("All metadata diagnostics passed.")
