import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
import pandas as pd
from matplotlib.colors import Normalize, LinearSegmentedColormap, to_rgba
from matplotlib.colorbar import ColorbarBase
from matplotlib.patches import Rectangle
from pandas.api.types import is_numeric_dtype


def normalize_channel(channel, vmin: float = None, vmax: float = None):
    assert channel.ndim == 2, f"Expected 2D channel, got {channel.ndim}D"
    vmin = channel.min() if vmin is None else vmin
    vmax = channel.max() if vmax is None else vmax
    if vmin == vmax:
        return np.zeros_like(channel, dtype=float)
    return (np.clip(channel, vmin, vmax) - vmin) / (vmax - vmin)


def normalize_image_channels(image: np.ndarray, vmins: list[float] = None, vmaxs: list[float] = None):
    assert image.ndim == 3, f"Expected 3D image, got {image.ndim}D"
    vmins = image.min(axis=(1, 2)) if vmins is None else vmins
    vmaxs = image.max(axis=(1, 2)) if vmaxs is None else vmaxs
    return np.stack([
        normalize_channel(ch, vmin, vmax)
        for ch, vmin, vmax in zip(image, vmins, vmaxs)
    ])


def create_colormap_from_color(color: str, background: str = "black", reverse: bool = False):
    from matplotlib.colors import to_rgb
    rgb = to_rgb(color)
    if reverse:
        rgb = tuple(1 - x for x in rgb)
    bg = to_rgb(background)
    return LinearSegmentedColormap.from_list("custom_cmap", [(0, bg), (1, rgb)], N=256)


def channel_to_rgba(channel: np.ndarray, cmap=None, cmap_name: str = None):
    cmap = cmap or plt.get_cmap(cmap_name)
    return cmap(channel)


def image_to_rgba(image: np.ndarray, colors=('b', 'g', 'r', 'c', 'm', 'y'), vmins=None, vmaxs=None, blend='additive'):
    assert image.ndim == 3, f"Expected 3D image, got {image.ndim}D"
    assert len(image) <= len(colors), "Too many channels for the number of colors"
    image = normalize_image_channels(image, vmins, vmaxs)
    cmaps = [create_colormap_from_color(c) for c in colors]
    rgba = np.stack([channel_to_rgba(ch, cmap=cmap) for ch, cmap in zip(image, cmaps)], axis=-1)
    if blend == 'additive':
        return np.clip(rgba.sum(axis=-1), 0, 1)
    elif blend == 'weighted':
        return rgba.mean(axis=-1)
    raise ValueError(f"Unknown blend mode: {blend}")


def blend(img1, img2, alpha=0.5):
    return alpha * img1 + (1 - alpha) * img2


def get_legend_handles_from_dict(label_to_color):
    return [Rectangle((0, 0), 1, 1, facecolor=color, label=label) for label, color in label_to_color.items()]


def get_legend_height_and_width(fig: plt.Figure, legend):
    if not fig.canvas.get_renderer():
        fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = legend.get_window_extent(renderer)
    return bbox.height / fig.dpi / fig.get_figheight(), bbox.width / fig.dpi / fig.get_figwidth()


def add_legends_to_fig(fig: plt.Figure, legends: list[dict], pos_y=0.95, pos_x=0.05, direction="horizontal", pad_x=10, pad_y=10, wrap=None):
    pad_x /= fig.get_figwidth() * fig.dpi
    pad_y /= fig.get_figheight() * fig.dpi
    start_x, start_y = pos_x, pos_y
    for n, legend in enumerate(legends, 1):
        if legend["type"] == "discrete":
            handles = get_legend_handles_from_dict(legend["label_to_color"])
            leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(pos_x, pos_y), frameon=True,
                             labelspacing=0.1, title=legend["title"], handlelength=1, handleheight=1,
                             borderaxespad=0, alignment="left", bbox_transform=fig.transFigure,
                             **legend.get("kwargs", {}))
            height, width = get_legend_height_and_width(fig, leg)
        elif legend["type"] == "continuous":
            height = legend["height"] / (fig.get_figheight() * fig.dpi)
            width = legend["width"] / (fig.get_figwidth() * fig.dpi)
            cax = fig.add_axes([pos_x, pos_y - height, width, height])
            norm = Normalize(vmin=legend["vmin"], vmax=legend["vmax"])
            ColorbarBase(cax, cmap=legend["colormap"], norm=norm, orientation=legend["orientation"],
                         label=legend["title"], **legend.get("kwargs", {}))
        else:
            raise NotImplementedError(f"Unknown legend type: {legend['type']}")

        if wrap and n % wrap == 0:
            if direction == "horizontal":
                pos_x = start_x
                pos_y -= height + pad_y
            else:
                pos_y = start_y
                pos_x += width + pad_x
        else:
            if direction == "horizontal":
                pos_x += width + pad_x
            else:
                pos_y -= height + pad_y



