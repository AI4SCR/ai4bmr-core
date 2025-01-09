import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize


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
    from matplotlib.patches import Patch
    from matplotlib.patches import Rectangle

    legend_elements = [Rectangle((0, 0), 1, 1, facecolor=color, label=label) for label, color in label_to_color.items()]
    # legend_elements = [Patch(facecolor=color, label=label) for label, color in label_to_color.items()]
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
    n_col = n_samples // n_row
    return n_row, n_col


def get_grid_dims(n_samples) -> (int, int):
    n_row = int(np.ceil(np.sqrt(n_samples)))
    n_col = n_samples // n_row
    return n_row, n_col
