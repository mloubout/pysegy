"""Display transformations shared by viewer presentation layers."""

from dataclasses import dataclass

import numpy as np

from .services import GatherWindow


@dataclass(frozen=True)
class WiggleData:
    """Normalized traces and their horizontal plot positions."""

    traces: np.ndarray
    positions: np.ndarray
    time_seconds: np.ndarray


def plotly_colorscale(colormap, samples: int = 17) -> list[list[object]]:
    """Sample a Matplotlib-compatible colormap for use by Plotly.

    Scientific colormaps such as those from :mod:`cmocean` are distributed as
    Matplotlib colormap objects. Plotly expects explicit normalized positions
    and CSS colors instead.
    """

    if samples < 2:
        raise ValueError("A color scale requires at least two samples")
    positions = np.linspace(0.0, 1.0, samples)
    colors = np.asarray(colormap(positions))
    if colors.shape != (samples, 4):
        raise ValueError("Colormap must return RGBA values")
    result = []
    for position, color in zip(positions, colors):
        red, green, blue = np.rint(color[:3] * 255).astype(int)
        result.append([
            float(position),
            f"rgb({red},{green},{blue})",
        ])
    return result


def scaled_amplitudes(
    window: GatherWindow,
    *,
    time_gain: float = 0.0,
    reverse_polarity: bool = False,
) -> np.ndarray:
    """Apply non-destructive time gain and polarity to a gather window."""

    if not 0.0 <= time_gain <= 3.0:
        raise ValueError("Time gain must be between 0 and 3")
    data = np.asarray(window.data, dtype=np.float64)
    if time_gain:
        data = data * np.power(window.time_seconds[:, None], time_gain)
    if reverse_polarity:
        data = -data
    return data


def amplitude_limit(data: np.ndarray, percentile: float = 99.0) -> float:
    """Return a finite, positive symmetric color limit."""

    if not 0.0 < percentile <= 100.0:
        raise ValueError("Amplitude percentile must be in (0, 100]")
    finite = np.abs(np.asarray(data)[np.isfinite(data)])
    if not finite.size:
        return 1.0
    limit = float(np.percentile(finite, percentile))
    return limit if limit > 0.0 else 1.0


def horizontal_axis(window: GatherWindow, mode: str) -> tuple[np.ndarray, str]:
    """Return trace-number or receiver-X coordinates for a gather window."""

    if mode == "Trace number":
        return window.trace_indices, "Trace"
    if mode == "Receiver X":
        return window.receiver_coordinates[:, 0], "Receiver X"
    raise ValueError(f"Unknown horizontal axis: {mode}")


def prepare_wiggles(
    data: np.ndarray,
    positions: np.ndarray,
    time_seconds: np.ndarray,
    *,
    max_traces: int = 100,
    width: float = 0.4,
) -> WiggleData:
    """Downsample and independently normalize traces for a wiggle plot."""

    values = np.asarray(data, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(positions):
        raise ValueError("Wiggle data and positions have incompatible shapes")
    if max_traces < 1 or width <= 0:
        raise ValueError("Wiggle display limits must be positive")
    step = max(1, int(np.ceil(values.shape[1] / max_traces)))
    values = values[:, ::step]
    positions = positions[::step]
    if len(positions) > 1:
        nonzero = np.abs(np.diff(positions))
        nonzero = nonzero[nonzero > 0]
        spacing = float(np.median(nonzero)) if nonzero.size else 1.0
    else:
        spacing = 1.0
    peaks = np.max(np.abs(values), axis=0, initial=0.0)
    peaks[peaks == 0.0] = 1.0
    normalized = values / peaks * spacing * width + positions
    return WiggleData(normalized, positions, np.asarray(time_seconds))
