"""Streamlit application for local SEG-Y exploration."""

import os

import colorcet as cc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
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
    prepare_wiggle_fill,
    prepare_wiggles,
    scaled_amplitudes,
    selected_geometry_record,
    validate_comparison_records,
)
from pysegy.viewer.services import (
    DEFAULT_HEADER_FIELDS,
    dataset_diagnostics,
    dataset_summary,
    file_header_info,
    load_gather,
    load_header_table,
    open_native_file_dialog,
    receiver_attribute,
    scan_local_dataset,
    scan_local_dataset_cached,
    source_geometry,
)


SITE_INK = "#494e52"
SITE_ACCENT = "#52adc8"
SITE_ACCENT_DARK = "#367f99"
SITE_SURFACE = "#f5f6f6"
SITE_BORDER = "#e5e7e9"

pio.templates["pysegy_site"] = go.layout.Template(
    layout={
        "font": {
            "family": "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            "color": SITE_INK,
        },
        "title": {"font": {"color": SITE_INK, "size": 20}},
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "colorway": [SITE_ACCENT, "#f28e2b", "#4e79a7", "#59a14f"],
        "xaxis": {
            "gridcolor": "#edf0f2",
            "linecolor": "#c7ccd1",
            "zerolinecolor": "#d8dcdf",
        },
        "yaxis": {
            "gridcolor": "#edf0f2",
            "linecolor": "#c7ccd1",
            "zerolinecolor": "#d8dcdf",
        },
        "legend": {"bgcolor": "rgba(255,255,255,0.88)"},
    }
)
pio.templates.default = "plotly+pysegy_site"

st.set_page_config(
    page_title="pysegy Viewer",
    page_icon="〰",
    layout="wide",
)
st.markdown(
    """
    <style>
    :root {
        --site-ink: #494e52;
        --site-muted: #7a8288;
        --site-accent: #52adc8;
        --site-accent-dark: #367f99;
        --site-surface: #f5f6f6;
        --site-border: #e5e7e9;
    }
    .stApp {
        color: var(--site-ink);
        background: #fff;
    }
    .block-container {
        max-width: 1600px;
        padding-top: 1rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: var(--site-surface);
        border-right: 1px solid var(--site-border);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label {
        color: var(--site-ink);
    }
    [data-testid="stSidebar"] h2 {
        margin-top: .35rem;
        padding-bottom: .35rem;
        border-bottom: 2px solid var(--site-accent);
        font-size: 1.05rem;
        text-transform: uppercase;
        letter-spacing: .08em;
    }
    [data-testid="stMetric"] {
        background: var(--site-surface);
        border: 0;
        border-top: 3px solid var(--site-accent);
        border-radius: 0.15rem;
        padding: 0.75rem 1rem;
    }
    [data-testid="stMetricLabel"] {color: var(--site-muted);}
    [data-testid="stExpander"] {
        background: #fff;
        border-color: var(--site-border);
        border-radius: 0.15rem;
    }
    [data-testid="stButton"] button[kind="primary"] {
        background: var(--site-accent-dark);
        border-color: var(--site-accent-dark);
    }
    [data-testid="stButton"] button:not([kind="primary"]),
    [data-testid="stDownloadButton"] button {
        border-color: #b8c0c5;
        color: var(--site-ink);
    }
    [data-testid="stButton"] button:hover,
    [data-testid="stDownloadButton"] button:hover {
        border-color: var(--site-accent);
        color: var(--site-accent-dark);
    }
    [data-testid="stSegmentedControl"] button[aria-pressed="true"] {
        background: var(--site-accent-dark);
        color: #fff;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--site-border);
        border-radius: 0.15rem;
    }
    .viewer-masthead {
        display: flex;
        align-items: baseline;
        gap: 1rem;
        margin: -.15rem 0 1.25rem;
        padding: .25rem 0 .9rem;
        border-bottom: 1px solid var(--site-border);
    }
    .viewer-wordmark {
        color: var(--site-ink);
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -.035em;
        line-height: 1;
    }
    .viewer-wordmark span {color: var(--site-accent-dark);}
    .viewer-tagline {
        color: var(--site-muted);
        font-size: .95rem;
    }
    h1, h2, h3 {
        color: var(--site-ink);
        letter-spacing: -0.025em;
    }
    h1 {font-size: 2rem !important;}
    h2 {
        padding-bottom: .35rem;
        border-bottom: 1px solid var(--site-border);
    }
    a {color: var(--site-accent-dark);}
    div[data-baseweb="notification"] {border-radius: .15rem;}
    @media (max-width: 700px) {
        .viewer-masthead {display: block;}
        .viewer-tagline {display: block; margin-top: .5rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="viewer-masthead">
      <div class="viewer-wordmark">py<span>segy</span></div>
      <div class="viewer-tagline">
        Local SEG-Y geometry, gather, and header exploration
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

SEISMIC_COLOR_SCALES = {
    "Perceptual gray": plotly_colorscale(cc.cm.CET_L1),
    "Seismic": SEISMIC_COLORSCALE,
    "RTM": RTM_COLORSCALE,
    "Orange–black": ORANGE_BLACK_COLORSCALE,
}
DEPTH_COLOR_SCALE = plotly_colorscale(cc.cm.bgy)
TRACE_COUNT_COLOR_SCALE = plotly_colorscale(cc.cm.fire)
COORDINATE_UNITS = "scaled SEG-Y units"
DEPTH_UNITS = "scaled depth/elevation units"


pending_dataset_path = st.session_state.pop("pending_dataset_path", None)
if pending_dataset_path is not None:
    st.session_state.dataset_path = pending_dataset_path
if "dataset_path" not in st.session_state:
    st.session_state.dataset_path = os.getenv("PYSEGY_VIEWER_PATH", "")


with st.sidebar:
    with st.expander(
        "Dataset",
        expanded=st.session_state.get("scan") is None,
    ):
        path = st.text_input(
            "SEG-Y file or directory",
            key="dataset_path",
            help="The path must be accessible to the machine running the viewer.",
        )
        if st.button("Open file…", width="stretch"):
            try:
                selected_path = open_native_file_dialog()
                if selected_path:
                    st.session_state.pending_dataset_path = selected_path
                    st.rerun()
            except RuntimeError as exc:
                st.warning(str(exc))
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
    st.info(
        "Enter a SEG-Y path on this Streamlit server and select "
        "**Scan dataset**. To inspect files on your computer without uploading "
        "them, run `pysegy-viewer` on that computer."
    )
    st.stop()

pending_record = st.session_state.pop("pending_record_index", None)
if pending_record is not None:
    st.session_state.record_index = int(pending_record)
current_record = int(st.session_state.get("record_index", 0))
st.session_state.record_index = min(max(current_record, 0), len(scan) - 1)


def step_record(delta: int) -> None:
    """Move the shared gather selection from a widget callback."""

    selected = int(st.session_state.get("record_index", 0)) + delta
    st.session_state.record_index = min(max(selected, 0), len(scan) - 1)


def set_record(index: int) -> None:
    """Jump to a gather from a widget callback."""

    st.session_state.record_index = index


navigation_geometry = source_geometry(scan)

with st.sidebar:
    st.divider()
    st.header("Workspace")
    workspace = st.radio(
        "Workspace",
        ["Survey", "Gather"],
        key="workspace",
        label_visibility="collapsed",
    )

record_index = int(st.session_state.record_index)
if workspace == "Gather":
    with st.sidebar:
        st.divider()
        st.header("Gather navigation")
        record_index = st.selectbox(
            "Selected gather",
            options=range(len(scan)),
            format_func=lambda index: f"Gather {index + 1:,}",
            key="record_index",
        )
        first_col, previous_col, next_col, last_col = st.columns(4)
        first_col.button(
            "⇤",
            help="First gather",
            disabled=st.session_state.record_index == 0,
            on_click=set_record,
            args=(0,),
            width="stretch",
        )
        previous_col.button(
            "←",
            help="Previous gather",
            disabled=st.session_state.record_index == 0,
            on_click=step_record,
            args=(-1,),
            width="stretch",
        )
        next_col.button(
            "→",
            help="Next gather",
            disabled=st.session_state.record_index == len(scan) - 1,
            on_click=step_record,
            args=(1,),
            width="stretch",
        )
        last_col.button(
            "⇥",
            help="Last gather",
            disabled=st.session_state.record_index == len(scan) - 1,
            on_click=set_record,
            args=(len(scan) - 1,),
            width="stretch",
        )
        st.caption(f"Gather {int(record_index) + 1:,} of {len(scan):,}")
    record_index = int(st.session_state.record_index)
record = scan[int(record_index)]
selected_source = navigation_geometry[int(record_index)]

if workspace == "Survey":
    st.header("Survey")
    survey_view = st.segmented_control(
        "Survey section",
        ["Overview", "Geometry", "Quality control", "File headers"],
        default="Overview",
        key="survey_view",
        width="stretch",
    )
else:
    st.header(f"Gather {int(record_index) + 1:,}")
    gather_view = st.segmented_control(
        "Gather section",
        ["Seismic", "Compare gathers", "Geometry & depth", "Trace headers"],
        default="Seismic",
        key="gather_view",
        width="stretch",
    )

if workspace == "Survey" and survey_view == "Overview":
    st.subheader("Overview")
    summary = dataset_summary(scan)
    cols = st.columns(5)
    cols[0].metric("Gathers", f"{summary.records:,}")
    cols[1].metric("Traces", f"{summary.traces:,}")
    cols[2].metric("Samples / trace", f"{summary.samples_per_trace:,}")
    cols[3].metric("Sample interval [µs]", f"{summary.sample_interval_us:,}")
    cols[4].metric("Sample format [SEG-Y code]", summary.sample_format)
    if st.session_state.get("cache_hit"):
        st.caption("Loaded unchanged scan metadata from the local cache.")


if workspace == "Survey" and survey_view == "Geometry":
    st.subheader("Geometry")
    geometry = source_geometry(scan)
    frame = pd.DataFrame(geometry, columns=["x", "y", "depth"])
    frame["record"] = range(len(frame))
    frame["traces"] = scan.counts
    gather_kind = "Receiver gather" if record.by_receiver else "Source gather"
    depth_keys = sorted({item.depth_key for item in scan.records})
    depth_header = depth_keys[0] if len(depth_keys) == 1 else "mixed headers"
    depth_label = f"{gather_kind} depth [{DEPTH_UNITS}] ({depth_header})"
    gather_color_label = st.selectbox(
        "Color gather locations by",
        [depth_label, "Trace count"],
    )
    gather_colors = {
        depth_label: frame["depth"],
        "Trace count": frame["traces"],
    }
    frame["color"] = gather_colors[gather_color_label]
    figure = px.scatter(
        frame,
        x="x",
        y="y",
        color="color",
        custom_data=["record", "traces"],
        hover_data=["record", "traces"],
        title=f"{gather_kind} locations",
        labels={
            "x": f"X [{COORDINATE_UNITS}]",
            "y": f"Y [{COORDINATE_UNITS}]",
            "color": gather_color_label,
        },
        color_continuous_scale=(
            TRACE_COUNT_COLOR_SCALE
            if gather_color_label == "Trace count"
            else DEPTH_COLOR_SCALE
        ),
    )
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


if workspace == "Survey" and survey_view == "File headers":
    st.subheader("File headers")
    header_info = file_header_info(scan)
    if header_info.textual:
        st.subheader("Textual file header")
        with st.container(border=True):
            st.text(header_info.textual)
    else:
        st.info("No readable ASCII or EBCDIC textual file header was found.")
    st.subheader("Binary file header")
    binary_units = {
        "dt": "µs",
        "dtOrig": "µs",
        "ns": "samples",
        "nsOrig": "samples",
        "DataSampleFormat": "SEG-Y code",
        "SweepFrequencyStart": "Hz",
        "SweepFrequencyEnd": "Hz",
        "SweepLength": "ms",
    }
    binary_frame = pd.DataFrame([
        {
            "Field": field,
            "Value": value,
            "Unit": binary_units.get(field, "code / count"),
        }
        for field, value in header_info.binary.items()
    ])
    st.dataframe(binary_frame, width="stretch", hide_index=True)
    unique_paths = list(dict.fromkeys(scan.paths))
    st.caption(
        f"This combined scan contains {len(unique_paths):,} file(s). "
        "The displayed file header is the header retained by the scan."
    )


if workspace == "Gather":
    detail_cols = st.columns(4)
    detail_cols[0].metric("Source X [scaled]", f"{selected_source[0]:g}")
    detail_cols[1].metric("Source Y [scaled]", f"{selected_source[1]:g}")
    detail_cols[2].metric(
        f"Source depth [{DEPTH_UNITS}]",
        f"{selected_source[2]:g}",
    )
    detail_cols[3].metric("Traces", f"{record.ntraces:,}")
    if gather_view == "Seismic":
        st.subheader("Seismic data")
        with st.container(border=True):
            st.markdown("**Display controls**")
            trace_control, sample_control, display_control, axis_control = st.columns(
                [2, 2, 1, 1]
            )
            trace_range = trace_control.slider(
                "Trace range [trace number]",
                0,
                record.ntraces,
                (0, record.ntraces),
            )
            sample_range = sample_control.slider(
                "Sample range [sample number]",
                0,
                record.ns,
                (0, record.ns),
            )
            display_mode = display_control.selectbox(
                "Display", ["Image", "Wiggle"]
            )
            axis_mode = axis_control.selectbox(
                "Horizontal axis", ["Trace number", "Receiver X"]
            )
            advanced_controls = st.columns([1.4, 1.4, 1.4, 1, 1.4])
            (
                color_control,
                clip_control,
                gain_control,
                polarity_control,
                height_control,
            ) = advanced_controls
            color_scale = color_control.selectbox(
                "Perceptual color scale",
                list(SEISMIC_COLOR_SCALES),
                disabled=display_mode == "Wiggle",
            )
            percentile = clip_control.slider(
                "Amplitude clipping [%]", 80, 100, 99
            )
            time_gain = gain_control.slider(
                "Time gain exponent [dimensionless]", 0.0, 3.0, 0.0, 0.25
            )
            reverse_polarity = polarity_control.toggle(
                "Reverse polarity", value=False
            )
            plot_height = height_control.slider(
                "Plot height [px]",
                min_value=450,
                max_value=850,
                value=650,
                step=50,
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
                    fill = prepare_wiggle_fill(
                        wiggles.traces[:, index],
                        position,
                        wiggles.time_seconds,
                    )
                    figure.add_trace(go.Scattergl(
                        x=fill.negative_x,
                        y=fill.time_seconds,
                        mode="lines",
                        fill="toself",
                        fillcolor="rgba(33, 102, 172, 0.75)",
                        line={"width": 0},
                        name="Negative amplitude",
                        hoverinfo="skip",
                        showlegend=index == 0,
                    ))
                    figure.add_trace(go.Scattergl(
                        x=fill.positive_x,
                        y=fill.time_seconds,
                        mode="lines",
                        fill="toself",
                        fillcolor="rgba(215, 48, 39, 0.75)",
                        line={"width": 0},
                        name="Positive amplitude",
                        hoverinfo="skip",
                        showlegend=index == 0,
                    ))
                    figure.add_trace(go.Scattergl(
                        x=wiggles.traces[:, index],
                        y=wiggles.time_seconds,
                        mode="lines",
                        line={"color": "black", "width": 1},
                        name=f"{position:g}",
                        hovertemplate=(
                            f"position={position:g}<br>"
                            "time=%{y:.4f} s<extra></extra>"
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

    if gather_view == "Geometry & depth":
        geometry_view = st.segmented_control(
            "Geometry section",
            ["Receiver map", "Depth profile"],
            default="Receiver map",
            key="geometry_view",
        )
        if geometry_view == "Receiver map":
            receivers = record.rec_coordinates
            receiver_step = max(1, int(np.ceil(len(receivers) / 5000)))
            receivers_display = receivers[::receiver_step]
            receiver_color_label = st.selectbox(
                "Color receivers by",
                [
                    f"Receiver depth [{DEPTH_UNITS}] ({record.rec_depth_key})",
                    f"Group water depth [{DEPTH_UNITS}]",
                    "Trace number",
                ],
            )
            if receiver_color_label.startswith("Receiver depth"):
                receiver_colors = receivers_display[:, 2]
            elif receiver_color_label == "Trace number":
                receiver_colors = np.arange(0, record.ntraces, receiver_step)
            else:
                receiver_colors = receiver_attribute(
                    scan, int(record_index), "GroupWaterDepth"
                )[::receiver_step]
            receiver_frame = pd.DataFrame({
                "x": receivers_display[:, 0],
                "y": receivers_display[:, 1],
                "depth": receivers_display[:, 2],
                "color": receiver_colors,
                "trace": np.arange(0, record.ntraces, receiver_step),
            })
            receiver_figure = px.scatter(
                receiver_frame,
                x="x",
                y="y",
                color="color",
                hover_data=["trace", "depth"],
                labels={
                    "x": f"Receiver X [{COORDINATE_UNITS}]",
                    "y": f"Receiver Y [{COORDINATE_UNITS}]",
                    "color": receiver_color_label,
                },
                title="Receiver locations for the selected gather",
                color_continuous_scale=(
                    TRACE_COUNT_COLOR_SCALE
                    if receiver_color_label == "Trace number"
                    else DEPTH_COLOR_SCALE
                ),
            )
            receiver_figure.update_yaxes(scaleanchor="x", scaleratio=1)
            st.plotly_chart(receiver_figure, width="stretch")
        else:
            receivers = record.rec_coordinates
            source_depth = float(selected_source[2])
            depth_figure = go.Figure()
            depth_figure.add_trace(go.Scattergl(
                x=receivers[:, 0],
                y=receivers[:, 2],
                mode="lines+markers",
                name=f"Receiver: {record.rec_depth_key}",
                customdata=np.arange(record.ntraces),
                hovertemplate=(
                    "trace=%{customdata}<br>x=%{x:g}<br>"
                    "depth=%{y:g}<extra></extra>"
                ),
            ))
            depth_figure.add_hline(
                y=source_depth,
                line_dash="dash",
                annotation_text=(
                    f"Source: {record.depth_key} = {source_depth:g}"
                ),
                annotation_position="top left",
            )
            depth_figure.update_layout(
                xaxis_title=f"Receiver X [{COORDINATE_UNITS}]",
                yaxis_title=f"Depth / elevation [{DEPTH_UNITS}]",
                legend_title="Header selected by pysegy",
            )
            if "depth" in record.rec_depth_key.lower():
                depth_figure.update_yaxes(autorange="reversed")
            st.plotly_chart(depth_figure, width="stretch")

    if gather_view == "Compare gathers":
        st.subheader("Side-by-side gather comparison")
        neighboring_record = min(int(record_index) + 1, len(scan) - 1)
        if neighboring_record == int(record_index) and int(record_index) > 0:
            neighboring_record -= 1
        default_comparison = list(dict.fromkeys([
            int(record_index), neighboring_record
        ]))
        selected_records = st.multiselect(
            "Gathers to compare (2–4)",
            options=range(len(scan)),
            default=default_comparison,
            format_func=lambda index: f"Gather {index + 1:,}",
        )
        compare_a, compare_b, compare_c = st.columns(3)
        comparison_scale = compare_a.selectbox(
            "Color scale",
            list(SEISMIC_COLOR_SCALES),
            key="comparison_scale",
        )
        comparison_percentile = compare_b.slider(
            "Amplitude clipping [%]",
            80,
            100,
            99,
            key="comparison_percentile",
        )
        comparison_gain = compare_c.slider(
            "Time gain exponent [dimensionless]",
            0.0,
            3.0,
            0.0,
            0.25,
            key="comparison_gain",
        )
        try:
            comparison_indices = validate_comparison_records(
                selected_records, len(scan)
            )
            sample_intervals = {scan[index].dt for index in comparison_indices}
            if len(sample_intervals) != 1:
                raise ValueError(
                    "Selected gathers must have the same sample interval"
                )
            common_sample_count = min(
                scan[index].ns for index in comparison_indices
            )
            comparison_windows = [
                load_gather(
                    scan,
                    index,
                    sample_stop=common_sample_count,
                    max_traces=250,
                    max_samples=1200,
                )
                for index in comparison_indices
            ]
            comparison_amplitudes = [
                scaled_amplitudes(window, time_gain=comparison_gain)
                for window in comparison_windows
            ]
            shared_limit = max(
                amplitude_limit(values, comparison_percentile)
                for values in comparison_amplitudes
            )
            subplot_titles = []
            for index in comparison_indices:
                source = navigation_geometry[index]
                subplot_titles.append(
                    f"Gather {index + 1:,}<br>"
                    f"<sup>X {source[0]:g}, Y {source[1]:g} "
                    f"[{COORDINATE_UNITS}] · {scan[index].ntraces:,} traces</sup>"
                )
            comparison_figure = make_subplots(
                rows=1,
                cols=len(comparison_indices),
                shared_yaxes=True,
                horizontal_spacing=0.0,
                subplot_titles=subplot_titles,
            )
            for column_number, (index, window, values) in enumerate(zip(
                comparison_indices,
                comparison_windows,
                comparison_amplitudes,
            )):
                comparison_figure.add_trace(
                    go.Heatmap(
                        z=values,
                        x=window.trace_indices,
                        y=window.time_seconds,
                        colorscale=SEISMIC_COLOR_SCALES[comparison_scale],
                        zmin=-shared_limit,
                        zmax=shared_limit,
                        showscale=(
                            column_number == len(comparison_indices) - 1
                        ),
                        colorbar={"title": "Amplitude", "x": 1.01},
                    ),
                    row=1,
                    col=column_number + 1,
                )
                comparison_figure.update_xaxes(
                    title_text="Trace number",
                    showline=True,
                    linewidth=1,
                    linecolor="#808080",
                    row=1,
                    col=column_number + 1,
                )
            comparison_figure.update_yaxes(
                autorange="reversed",
                title_text="Time [s]",
                row=1,
                col=1,
            )
            comparison_figure.update_layout(
                height=700,
                margin={"l": 60, "r": 80, "t": 75, "b": 55},
            )
            st.plotly_chart(comparison_figure, width="stretch")
        except (ValueError, IndexError) as exc:
            st.info(str(exc))

if workspace == "Gather" and gather_view == "Trace headers":
    st.subheader("Trace headers")
    fields = st.multiselect(
        "Header fields",
        TH_FIELDS,
        default=list(DEFAULT_HEADER_FIELDS),
    )
    try:
        table = load_header_table(scan, int(record_index), fields)
        header_frame = pd.DataFrame(table.columns)
        header_frame.insert(0, "Trace", table.trace_indices)
        table_panel, profile_panel = st.columns([1, 2], gap="large")
        with table_panel:
            st.dataframe(
                header_frame,
                width="stretch",
                height=520,
                hide_index=True,
            )
            st.download_button(
                "Download CSV",
                header_frame.to_csv(index=False),
                file_name=f"gather-{int(record_index)}-headers.csv",
                mime="text/csv",
                width="stretch",
            )
            if len(header_frame) < record.ntraces:
                st.caption(
                    f"Showing {len(header_frame):,} sampled rows from "
                    f"{record.ntraces:,} traces."
                )
        with profile_panel:
            if fields:
                profile_a, profile_b = st.columns(2)
                x_field = profile_a.selectbox(
                    "Horizontal profile field",
                    ["Trace", *fields],
                )
                y_field = profile_b.selectbox(
                    "Header field to inspect",
                    fields,
                    index=fields.index("Offset") if "Offset" in fields else 0,
                )
                header_profile = go.Figure(go.Scattergl(
                    x=header_frame[x_field],
                    y=header_frame[y_field],
                    mode="lines" if x_field == "Trace" else "markers",
                    name=y_field,
                    hovertemplate=(
                        f"{x_field}=%{{x:g}}<br>"
                        f"{y_field}=%{{y:g}}<extra></extra>"
                    ),
                ))
                header_profile.update_layout(
                    title=f"{y_field} profile",
                    xaxis_title=x_field,
                    yaxis_title=y_field,
                    height=560,
                )
                st.plotly_chart(header_profile, width="stretch")
    except Exception as exc:
        st.error(f"Could not load trace headers: {exc}")

if workspace == "Survey" and survey_view == "Quality control":
    st.subheader("Quality control")
    diagnostics = dataset_diagnostics(scan)
    bounds = diagnostics.coordinate_bounds
    diagnostic_cols = st.columns(3)
    diagnostic_cols[0].metric("Files", f"{diagnostics.files:,}")
    diagnostic_cols[1].metric(
        "Source X range [scaled]", f"{bounds[0]:g} to {bounds[1]:g}"
    )
    diagnostic_cols[2].metric(
        "Source Y range [scaled]", f"{bounds[2]:g} to {bounds[3]:g}"
    )
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
