"""
Utility helpers for quick visualisation of seismic data.

The helpers mirror a subset of the `SlimPlotting.jl` API and can work
directly with :class:`~pysegy.SeisBlock` or :class:`~pysegy.ShotRecord`
instances.  When given one of these objects the functions will infer the
sample spacing from the relevant headers so ``spacing`` can be omitted.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Sequence, Union

from .types import SeisBlock
from .scan import ShotRecord
from .utils import get_header

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]


def _extract_spacing(
    src: ArrayLike | SeisBlock | ShotRecord,
    spacing: Tuple[float, float] | None,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Return data array and spacing for ``src``.

    Parameters
    ----------
    src : ArrayLike or SeisBlock or ShotRecord
        Input data object.
    spacing : (float, float), optional
        Pre-defined sample spacing ``(dz, dx)``. When ``None`` the spacing is
        inferred from ``src`` when possible.

    Returns
    -------
    array : numpy.ndarray
        2-D array of samples.
    spacing : tuple[float, float]
        Sample spacing ``(dz, dx)``.
    """

    if isinstance(src, ShotRecord):
        data = np.asarray(src.data)
        if spacing is None:
            dt = src.dt * 1e-6
            recx = src.rec_coordinates[:, 0]
            dx = float(np.median(np.diff(recx))) if len(recx) > 1 else 1.0
            spacing = (dt, dx)
    elif isinstance(src, SeisBlock):
        data = np.asarray(src.data)
        if spacing is None:
            dt = src.fileheader.bfh.dt * 1e-6
            gx = get_header(src, "GroupX")
            dx = float(np.median(np.diff(gx))) if len(gx) > 1 else 1.0
            spacing = (dt, dx)
    else:
        data = np.asarray(src)
        if spacing is None:
            spacing = (1.0, 1.0)
    return data, spacing


def _clip_limits(img: np.ndarray, perc: int = 95, positive: bool = False,
                 vmax: float | None = None) -> Tuple[float, float]:
    """
    Return intensity limits for an image based on percentiles.
    """

    if positive:
        high = np.percentile(img, perc)
        high = vmax if vmax is not None else high
        low = np.amin(img)
        return low, high
    high = np.percentile(np.abs(img), perc)
    high = vmax if vmax is not None else high
    return -high, high


def _plot_with_units(
    image: ArrayLike,
    spacing: Tuple[float, float] | None,
    *,
    perc: int = 95,
    cmap: str = "gray",
    vmax: float | None = None,
    origin: Tuple[float, float] = (0.0, 0.0),
    interp: str = "hanning",
    aspect: str | None = None,
    d_scale: float = 0.0,
    positive: bool = False,
    labels: Tuple[str, str] = ("X", "Depth"),
    cbar: bool = False,
    alpha: float | None = None,
    units: Tuple[str, str] = ("m", "m"),
    name: str = "",
    new_fig: bool = True,
    save: str | None = None,
):
    """
    Display ``image`` using ``spacing`` for axis units.

    ``image`` can be an ``ndarray`` or a :class:`SeisBlock` or
    :class:`ShotRecord`.  When ``spacing`` is ``None`` it will be derived from
    the object headers if possible.
    """
    arr, (dz, dx) = _extract_spacing(image, spacing)
    nz, nx = arr.shape
    oz, ox = origin
    if d_scale != 0:
        depth = np.arange(nz, dtype=float) ** d_scale
    else:
        depth = np.ones(nz, dtype=float)
    scaled = arr * depth[:, None]
    vmin, vmax = _clip_limits(scaled, perc, positive, vmax)
    extent = [ox, ox + (nx - 1) * dx, oz + (nz - 1) * dz, oz]
    if aspect is None:
        aspect = "auto"

    if new_fig:
        plt.figure()

    im = plt.imshow(
        scaled,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        aspect=aspect,
        interpolation=interp,
        extent=extent,
        alpha=alpha,
    )
    plt.xlabel(f"{labels[0]} [{units[0]}]")
    plt.ylabel(f"{labels[1]} [{units[1]}]")
    if name:
        plt.title(name)
    if cbar:
        plt.colorbar(im)
    if save:
        plt.savefig(save, bbox_inches="tight", dpi=150)
    return im


def plot_simage(
    image: ArrayLike | SeisBlock | ShotRecord,
    spacing: Tuple[float, float] | None = None,
    **kw,
):
    """
    Plot a migrated image with depth on the vertical axis.

    Parameters
    ----------
    image : array-like or SeisBlock or ShotRecord
        Data to visualize.
    spacing : (float, float), optional
        ``(dz, dx)`` sample spacing. If omitted and ``image`` is a
        :class:`SeisBlock` or :class:`ShotRecord` the spacing will be
        extracted from the headers.
    """
    kw.setdefault("cmap", "gray")
    kw.setdefault("name", "RTM")
    kw.setdefault("labels", ("X", "Depth"))
    kw.setdefault("units", ("m", "m"))
    return _plot_with_units(image, spacing, **kw)


def plot_velocity(
    image: ArrayLike | SeisBlock | ShotRecord,
    spacing: Tuple[float, float] | None = None,
    **kw,
):
    """
    Plot a velocity model.

    Parameters
    ----------
    image : array-like or SeisBlock or ShotRecord
        Model to display.
    spacing : (float, float), optional
        ``(dz, dx)`` sample spacing derived from ``image`` when omitted.
    """
    kw.setdefault("cmap", "turbo")
    kw.setdefault("name", "Velocity")
    kw.setdefault("labels", ("X", "Depth"))
    kw.setdefault("units", ("m", "m"))
    kw.setdefault("positive", True)
    return _plot_with_units(image, spacing, **kw)


def plot_fslice(
    image: ArrayLike | SeisBlock | ShotRecord,
    spacing: Tuple[float, float] | None = None,
    **kw,
):
    """Display a 2D frequency slice.

    Parameters
    ----------
    image : array-like or SeisBlock or ShotRecord
        Frequency-domain slice to plot.
    spacing : (float, float), optional
        ``(dz, dx)`` sample spacing derived from ``image`` when omitted.
    """
    kw.setdefault("cmap", "seismic")
    kw.setdefault("name", "Frequency slice")
    kw.setdefault("labels", ("X", "X"))
    kw.setdefault("units", ("m", "m"))
    return _plot_with_units(image, spacing, **kw)


def plot_sdata(
    image: ArrayLike | SeisBlock | ShotRecord,
    spacing: Tuple[float, float] | None = None,
    **kw,
):
    """
    Visualize a single shot record.

    ``image`` may be a raw ``ndarray`` or a :class:`SeisBlock` or
    :class:`ShotRecord`.  ``spacing`` follows the same rules as in
    :func:`plot_simage`.
    """
    kw.setdefault("cmap", "gray")
    kw.setdefault("name", "Shot record")
    kw.setdefault("labels", ("Xrec", "T"))
    kw.setdefault("units", ("m", "s"))
    return _plot_with_units(image, spacing, **kw)


def wiggle_plot(
    data: ArrayLike | SeisBlock | ShotRecord,
    xrec: Sequence[float] | None = None,
    time_axis: Sequence[float] | None = None,
    *,
    t_scale: float = 1.5,
    new_fig: bool = True,
):
    """
    Generate a classic wiggle plot for ``data``.

    ``data`` can be an ``ndarray`` or seismic container. When ``xrec`` or
    ``time_axis`` are omitted they are inferred from the headers when possible.
    """
    arr, (dt, dx) = _extract_spacing(data, None)
    if xrec is None:
        xrec = np.arange(arr.shape[1]) * dx
    if time_axis is None:
        time_axis = np.arange(arr.shape[0]) * dt
    xrec = np.asarray(xrec)
    time_axis = np.asarray(time_axis)
    tg = time_axis ** t_scale
    dx = np.diff(xrec, prepend=xrec[0])
    if new_fig:
        plt.figure()
    plt.ylim(time_axis.max(), time_axis.min())
    plt.xlim(xrec.min(), xrec.max())
    for i, xr in enumerate(xrec):
        trace = tg * arr[:, i]
        if np.max(np.abs(trace)) != 0:
            trace = dx[i] * trace / np.max(np.abs(trace)) + xr
        else:
            trace = trace + xr
        plt.plot(trace, time_axis, "k-", linewidth=0.5)
        plt.fill_betweenx(time_axis, xr, trace, where=trace > xr, color="k")
    plt.xlabel("X")
    plt.ylabel("Time")


def compare_shots(
    shot1: ArrayLike | SeisBlock | ShotRecord,
    shot2: ArrayLike | SeisBlock | ShotRecord,
    spacing: Tuple[float, float] | None = None,
    *,
    cmap: Sequence[str] | str = "gray",
    side_by_side: bool = False,
    chunksize: int = 20,
    **kw,
):
    """
    Overlay or juxtapose two shot gathers for comparison.

    Parameters
    ----------
    shot1, shot2 : array-like or SeisBlock or ShotRecord
        Gather data to compare.
    spacing : (float, float), optional
        Spacing applied when not deduced from the inputs.
    """
    arr1, spacing = _extract_spacing(shot1, spacing)
    arr2, spacing = _extract_spacing(shot2, spacing)
    if isinstance(cmap, str):
        cmap = (cmap, cmap)
    if side_by_side:
        pad = np.zeros((arr1.shape[0], 5))
        combo = np.hstack([arr1, pad, arr2[:, ::-1]])
        plot_sdata(combo, spacing, cmap=cmap[0], **kw)
        return

    nrec = min(arr1.shape[1], arr2.shape[1])
    out1 = np.zeros_like(arr1[:, :nrec])
    out2 = np.zeros_like(arr2[:, :nrec])
    for start in range(0, nrec, 2 * chunksize):
        out1[:, start:start + chunksize] = arr1[:, start:start + chunksize]
    for start in range(chunksize, nrec, 2 * chunksize):
        out2[:, start:start + chunksize] = arr2[:, start:start + chunksize]
    plot_sdata(out1, spacing, cmap=cmap[0], **kw)
    _plot_with_units(
        out2,
        spacing,
        cmap=cmap[1],
        new_fig=False,
        alpha=0.25,
        **kw,
    )
