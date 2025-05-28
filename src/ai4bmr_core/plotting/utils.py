import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
import colorcet as cc

def normalize_channel(channel, vmin: float = None, vmax: float = None):
    assert channel.ndim == 2, f'Expected 2D channel, got {channel.ndim}D'

    vmin = vmin or channel.min()
    vmax = vmax or channel.max()

    if vmin == vmax:
        return np.zeros_like(channel, dtype=float)

    channel = np.clip(channel, vmin, vmax)
    return (channel - vmin) / (vmax - vmin)

def normalize_image_channels(image: np.ndarray,
                             vmins: list[float] = None, vmaxs: list[float] = None
                             ):
    assert image.ndim == 3, f'Expected 3D image, got {image.ndim}D'

    vmins = image.min(axis=(1, 2)) if vmins is None else vmins
    vmaxs = image.max(axis=(1, 2)) if vmaxs is None else vmaxs
    assert len(vmins) == len(vmaxs) == len(image)

    image = np.stack([
        normalize_channel(channel, vmin=vmin, vmax=vmax)
        for channel, vmin, vmax in zip(image, vmins, vmaxs)
    ])

    return image

def create_colormap(
        color: str,
        background: str = 'black',
        reverse: bool = False,
):
    """
    Create a custom colormap from a color name

    Args:
        color: Color name or hex
        reverse: Invert colormap direction
        alpha: Alpha channel

    Returns:
        Matplotlib colormap
    """
    import matplotlib.colors as mcolors

    # Get RGB values
    rgb = mcolors.to_rgb(color)

    # Reverse if needed
    if reverse:
        rgb = tuple(1 - x for x in rgb)

    # Create colormap
    bg = mcolors.to_rgb(background)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'custom_cmap',
        [(0, bg), (1, rgb)],
        N=256
    )
    return cmap

def channel_to_rgba(channel: np.ndarray, cmap: mcolors.LinearSegmentedColormap = None, cmap_name: str = None):
    cmap = cmap or plt.get_cmap(cmap_name)
    rgba = cmap(channel)
    return rgba

def image_to_rgba(image: np.ndarray,
                  colors: tuple[str] = ('b', 'g', 'r', 'c', 'm', 'y'),
                  vmins: list[float] = None, vmaxs: list[float] = None,
                  # cmap: mcolors.LinearSegmentedColormap = None, cmap_name: str = None,
                  blend: str = 'additive'):
    """
    Convert a multi-channel image to an RGBA image using specified colors and blending.

    Args:
        image (np.ndarray): A 3D NumPy array of shape (C, H, W), where C is the number of channels.
        colors (tuple[str], optional): A tuple of matplotlib color codes to map each channel. Defaults to ('b', 'g', 'r', 'c', 'm', 'y').
        vmins (list[float], optional): Minimum values for normalizing each channel. Defaults to None (auto-determined).
        vmaxs (list[float], optional): Maximum values for normalizing each channel. Defaults to None (auto-determined).
        blend (str, optional): Blending mode to combine channels into one RGBA image.
            Options are:
            - 'additive': Add all RGBA layers and clip the result to [0, 1].
            - 'weighted': Average the RGBA layers.

    Returns:
        np.ndarray: A 3D RGBA image of shape (H, W, 4) with values in [0, 1].

    Raises:
        AssertionError: If the input image is not 3D or if the number of channels exceeds the number of provided colors.
        ValueError: If an unknown blend mode is specified.
    """

    assert image.ndim == 3, f'Expected 3D image, got {image.ndim}D'
    assert len(image) <= len(colors), f'Expected at most {len(colors)} channels, got {len(image)}'

    image = normalize_image_channels(image, vmins=vmins, vmaxs=vmaxs)
    cmaps = [create_colormap(color) for color in colors]
    rgba = [channel_to_rgba(channel, cmap=cmap) for channel, cmap in zip(image, cmaps)]
    rgba = np.stack(rgba, axis=-1)

    match blend:
        case 'additive':
            rgba = rgba.sum(axis=-1)
            rgba = rgba.clip(0, 1)
        case 'weighted':
            rgba = rgba.mean(axis=-1)
        case _:
            raise ValueError(f'Unknown blend mode: {blend}')

    return rgba


def blend(img1, img2, alpha=0.5):
    return alpha * img1 + (1 - alpha) * img2


def legend_from_dict(label_to_color: dict):
    """Create legend elements from a dictionary mapping labels to colors.

    Args:
        label_to_color (dict): Dictionary with labels as keys and colors as values.

    Returns:
        list: List of matplotlib.patches.Patch objects for use in legend.

    Example:
        # place legend outside the ax
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=6)
        # place legend in the top right corner of the figure
        ax.legend(handles=legend_elements, bbox_to_anchor=(1, 1), bbox_transform=fig.transFigure, fontsize=6)
    """
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor=color, label=label) for label, color in label_to_color.items()]
    return legend_elements


def get_legend_height_and_width(figure: plt.Figure, legend):
    """Get the height and width of a legend in figure coordinates.

    Args:
        figure (matplotlib.figure.Figure): The figure containing the legend
        legend (matplotlib.legend.Legend): The legend

    Returns:
        tuple: (height, width) in figure coordinates
    """
    fig = figure
    leg = legend

    if not fig.canvas.get_renderer():
        fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    bbox = leg.get_window_extent(renderer)
    width = bbox.width / fig.dpi / fig.get_figwidth()
    height = bbox.height / fig.dpi / fig.get_figheight()

    return height, width


def get_legend_handles_from_dict(label_to_color: dict):
    """Create quadratic legend patches from a dictionary mapping labels to colors.

    Args:
        label_to_color (dict): Dictionary with labels as keys and colors as values

    Returns:
        list: List of matplotlib.patches.Patch objects with square aspect ratio
    """
    legend_elements = [Rectangle((0, 0), 1, 1, facecolor=color, label=label) for label, color in label_to_color.items()]
    return legend_elements


def add_legends_to_fig(
    figure: plt.Figure,
    legends: list[dict],
    pos_y: float = 0.95,
    pos_x: float = 0.05,
    direction: str = "horizontal",
    pad_x: int = 10,
    pad_y: int = 10,
    wrap: int = None,
):
    """Add multiple legends (discrete and continuous) to a figure.

    Args:
        figure (plt.Figure): Matplotlib figure to add legends to
        legends (list[dict]): List of legend specifications
        pos_y (float, optional): Starting y position in figure coordinates. Defaults to 0.95.
        pos_x (float, optional): Starting x position in figure coordinates. Defaults to 0.05.
        direction (str, optional): Direction to stack legends ('horizontal' or 'vertical'). Defaults to 'horizontal'.
        pad_x (int, optional): Horizontal padding in pixels. Defaults to 10.
        pad_y (int, optional): Vertical padding in pixels. Defaults to 10.
        wrap (int, optional): Number of legends per row/column before wrapping. Defaults to None.

    Example:
        legends = [
            # Discrete legend example
            {
                'type': 'discrete',
                'label_to_color': {'Class A': 'red', 'Class B': 'blue'},
                'title': 'Categories'
            },
            # Continuous legend example
            {
                'type': 'continuous',
                'height': 100,  # in pixels
                'width': 20,    # in pixels
                'vmin': 0,
                'vmax': 1,
                'colormap': plt.cm.viridis,
                'orientation': 'vertical',
                'title': 'Values'
            }
        ]

        fig = plt.figure(figsize=(10, 6))
        add_legends_to_fig(figure=fig, legends=legends)
    """

    fig = figure
    max_height = 0
    max_width = 0
    pos_x_start = pos_x
    pos_y_start = pos_y

    # convert pixel padding to figure coordinates
    pad_x = pad_x / (fig.get_figwidth() * fig.dpi)
    pad_y = pad_y / (fig.get_figheight() * fig.dpi)

    for n, legend in enumerate(legends, start=1):
        if legend["type"] == "discrete":
            handles = get_legend_handles_from_dict(legend["label_to_color"])
            leg = fig.legend(
                handles=handles,
                loc="upper left",  # Position of legend
                bbox_to_anchor=(pos_x, pos_y),
                frameon=True,
                labelspacing=0.1,
                title=legend["title"],
                handlelength=1,
                handleheight=1,
                borderaxespad=0,
                alignment="left",
                bbox_transform=fig.transFigure,
                # todo: this will break if a kwargs is provided twice
                **legend.get("kwargs", {}),
            )
            height, width = get_legend_height_and_width(figure=fig, legend=leg)

        elif legend["type"] == "continuous":
            height = legend["height"] / (fig.get_figheight() * fig.dpi)
            width = legend["width"] / (fig.get_figwidth() * fig.dpi)

            cbar_ax = fig.add_axes([pos_x, pos_y - height, width, height])
            norm = Normalize(vmin=legend["vmin"], vmax=legend["vmax"])
            cb = ColorbarBase(
                cbar_ax,
                cmap=legend["colormap"],
                norm=norm,
                orientation=legend["orientation"],
                label=legend["title"],
                # todo: this will break if a kwargs is provided twice
                **legend.get("kwargs", {}),
            )

            # note: compute height and width including tick labels
            bbox = cbar_ax.get_tightbbox()
            width = bbox.width / (fig.get_figwidth() * fig.dpi)
            height = bbox.height / (fig.get_figheight() * fig.dpi)

        else:
            raise NotImplementedError(f'Unknown legend type: {legend["type"]}')

        max_height = max(max_height, height)
        max_width = max(max_width, width)

        if wrap is not None and n % wrap == 0:
            if direction == "horizontal":
                pos_x = pos_x_start
                pos_y -= max_height + pad_y
            else:
                pos_y = pos_y_start
                pos_x += max_width + pad_x
        else:
            if direction == "horizontal":
                pos_x += width + pad_x
            elif direction == "vertical":
                pos_y -= height + pad_y


def get_grid_dims(n_samples) -> (int, int):
    n_row = int(np.ceil(np.sqrt(n_samples)))
    n_col = int(np.ceil(n_samples / n_row))
    return n_row, n_col


from pathlib import Path

import colorcet as cc
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_rgba
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def normalize_data(data, scale="minmax"):
    # NORMALIZE

    censoring = 0.999
    cofactor = 1

    x = np.arcsinh(data / cofactor)
    thres = np.quantile(x, censoring, axis=0)

    for idx, t in enumerate(thres):
        x.values[:, idx] = np.where(x.values[:, idx] > t, t, x.values[:, idx])

    if scale == "minmax":
        data = pd.DataFrame(
            MinMaxScaler().fit_transform(x), columns=x.columns, index=x.index
        )
    elif scale == "standard":
        data = pd.DataFrame(
            StandardScaler().fit_transform(x), columns=x.columns, index=x.index
        )
    elif scale is None:
        data = x
    else:
        raise NotImplementedError()

    return data

def normalize(data: pd.DataFrame, scale: str = 'minmax', exclude_zeros: bool = False):
    import numpy as np

    index = data.index
    columns = data.columns
    x = data.values

    censoring = 0.999
    cofactor = 1
    x = np.arcsinh(x / cofactor)

    if exclude_zeros:
        masked_x = np.where(x == 0, np.nan, x)
        thres = np.nanquantile(masked_x, censoring, axis=0)
    else:
        thres = np.nanquantile(x, censoring, axis=0)

    x = np.minimum(x, thres)
    assert (x.max(axis=0) <= thres).all()

    if scale == "minmax":
        x = MinMaxScaler().fit_transform(x)
    elif scale == "standard":
        x = StandardScaler().fit_transform(x)
    else:
        raise NotImplementedError()

    return pd.DataFrame(x, index=index, columns=columns)


def prepare_data(base_dir: Path, scale="minmax"):
    # %% data
    base_dir = base_dir.expanduser()
    dataset = PCa(
        base_dir=base_dir,
        mask_version="cleaned",
        image_version="filtered",
        measurement_version="for_clustering",
    )
    dataset.create_clinical_metadata()
    dataset.setup()

    data = dataset._data["intensity"]
    assert data.isna().any().any() == False
    metadata = dataset._data["clinical_metadata"]

    metadata_cols = ["slide_code", "donor_block_id", "pat_id"]
    assert metadata[metadata_cols].isna().sum().sum() == 0

    data = data.join(metadata[metadata_cols]).set_index(metadata_cols, append=True)
    data = data.sort_index(level=["sample_name"])

    # NORMALIZE
    censoring = 0.999
    cofactor = 1
    x = np.arcsinh(data / cofactor)
    thres = np.quantile(x, censoring, axis=0)
    for idx, t in enumerate(thres):
        x.values[:, idx] = np.where(x.values[:, idx] > t, t, x.values[:, idx])

    if scale == "minmax":
        data = pd.DataFrame(
            MinMaxScaler().fit_transform(x), columns=x.columns, index=x.index
        )
    elif scale == "standard":
        data = pd.DataFrame(
            StandardScaler().fit_transform(x), columns=x.columns, index=x.index
        )
    elif scale is None:
        data = x
    else:
        raise NotImplementedError()

    return data


def create_color_maps_from_index(data: pd.DataFrame):
    color_maps = {}
    for label_name in set(data.index.names) - {"object_id"}:
        n = len(cc.glasbey_category10)
        labels = np.sort(data.index.get_level_values(label_name).unique())
        color_map = {
            label: cc.glasbey_category10[i % n] for i, label in enumerate(labels)
        }
        color_maps[label_name] = color_map
    return color_maps

def get_colorcet_map(item: list, as_int: bool = True) -> dict:
    uniq = sorted(set(item))
    glasbey_colors = cc.glasbey_bw[:len(uniq)]
    scale = 255 if as_int else 1
    color_map = {
        i: tuple(int(scale * c) for c in rgb)
        for i, rgb in zip(uniq, glasbey_colors)
    }
    return color_map


def create_color_maps_from_frame(data: pd.DataFrame):
    color_maps = {}
    for col_name in set(data.columns):
        n = len(cc.glasbey_category10)
        labels = np.sort(data[col_name].unique())
        color_map = {
            label: cc.glasbey_category10[i % n] for i, label in enumerate(labels)
        }
        color_maps[col_name] = color_map
    return color_maps


def create_color_maps(data: pd.DataFrame):
    color_maps = {}
    for col in data:
        if data[col].dtype.name == "category":
            n = len(cc.glasbey_category10)
            labels = np.sort(data[col].unique())
            color_map = {
                label: to_rgba(cc.glasbey_category10[i % n])
                for i, label in enumerate(labels)
            }
            color_maps[col] = color_map
        elif is_numeric_dtype(data[col]):
            cmap = LinearSegmentedColormap.from_list(col, cc.linear_kry_0_97_c73)
            color_maps[col] = cmap
        else:
            print(f"WARNING: {col} is not a category or numeric dtype")
    return color_maps


def create_legends(row_annotations, color_maps):
    legends = []
    for i in row_annotations:
        type_ = "discrete" if row_annotations[i].dtype == "category" else "continuous"
        if type_ == "discrete":
            legend = {"type": type_, "label_to_color": color_maps[i], "title": i}
        else:
            legend = {
                "type": type_,
                "height": 75,
                "width": 10,
                "vmin": row_annotations[i].min(),
                "vmax": row_annotations[i].max(),
                "colormap": color_maps[i],
                "orientation": "vertical",
                "title": i,
            }
        legends.append(legend)

    return legends


def map_row_annotations_to_colors(row_data, color_maps):
    row_colors = row_data.copy()
    cat_cols = row_colors.select_dtypes("category").columns
    for label_name in cat_cols:
        cmap = color_maps[label_name]
        row_colors[label_name] = [cmap[v] for v in row_colors[label_name]]

    num_cols = row_data.select_dtypes(["float", "int"]).columns
    for label_name in num_cols:
        cmap = color_maps[label_name]
        row_colors[label_name] = [cmap(v) for v in row_colors[label_name]]
    return row_colors


def normalize_row_annotations(row_annotations):
    row_annotations_norm = row_annotations.copy()

    num_cols = row_annotations.select_dtypes(["float", "int"]).columns
    for col in num_cols:
        from matplotlib.colors import Normalize

        norm = Normalize(
            vmin=row_annotations[col].min(), vmax=row_annotations[col].max()
        )
        row_annotations_norm[col] = norm(row_annotations[col])
    return row_annotations_norm

