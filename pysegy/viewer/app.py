"""Streamlit application for local SEG-Y exploration."""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pysegy.viewer.services import (
    dataset_summary,
    load_gather,
    scan_local_dataset,
    source_geometry,
)


st.set_page_config(page_title="pysegy Viewer", layout="wide")
st.title("pysegy Viewer")
st.caption("Explore local SEG-Y geometry and gathers without uploading your data.")

with st.sidebar:
    st.header("Dataset")
    path = st.text_input("SEG-Y file or directory", os.getenv("PYSEGY_VIEWER_PATH", ""))
    pattern = st.text_input("Directory pattern", "*.segy")
    by_receiver = st.toggle("Group by receiver", value=False)
    scan_clicked = st.button("Scan dataset", type="primary", use_container_width=True)

if scan_clicked:
    try:
        with st.spinner("Scanning trace headers…"):
            st.session_state.scan = scan_local_dataset(
                path,
                pattern=pattern if os.path.isdir(os.path.expanduser(path)) else None,
                by_receiver=by_receiver,
            )
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

geometry_tab, gather_tab = st.tabs(["Geometry", "Gather"])

with geometry_tab:
    geometry = source_geometry(scan)
    frame = pd.DataFrame(geometry, columns=["x", "y", "depth"])
    frame["record"] = range(len(frame))
    frame["traces"] = scan.counts
    figure = px.scatter(
        frame,
        x="x",
        y="y",
        color="depth",
        hover_data=["record", "traces"],
        title="Gather locations",
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(figure, use_container_width=True)
    if not geometry.size or not geometry[:, :2].any():
        st.warning("No non-zero source or receiver coordinates were detected.")

with gather_tab:
    record_index = st.number_input(
        "Gather index",
        min_value=0,
        max_value=max(0, len(scan) - 1),
        value=min(st.session_state.get("record_index", 0), len(scan) - 1),
        step=1,
    )
    record = scan[int(record_index)]
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
    percentile = st.slider("Amplitude percentile", 80, 100, 99)
    try:
        window = load_gather(
            scan,
            int(record_index),
            trace_start=trace_range[0],
            trace_stop=trace_range[1],
            sample_start=sample_range[0],
            sample_stop=sample_range[1],
        )
        limit = float(abs(window.data).max())
        if window.data.size:
            limit = float(np.percentile(abs(window.data), percentile))
        image = go.Figure(go.Heatmap(
            z=window.data,
            x=window.trace_indices,
            y=window.time_seconds,
            colorscale="Greys",
            zmin=-limit,
            zmax=limit,
            colorbar={"title": "Amplitude"},
        ))
        image.update_layout(xaxis_title="Trace", yaxis_title="Time [s]")
        image.update_yaxes(autorange="reversed")
        st.plotly_chart(image, use_container_width=True)
        st.caption(
            f"Displaying {window.data.shape[1]:,} traces × "
            f"{window.data.shape[0]:,} samples. Large selections are downsampled."
        )
    except Exception as exc:
        st.error(f"Could not load this gather: {exc}")
