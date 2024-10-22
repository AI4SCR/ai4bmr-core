def legend_from_dict(label_to_color: dict):
    """Create a legend from a dict with labels as keys and colors as values.
    See: https://matplotlib.org/stable/users/explain/axes/legend_guide.html#legend-guide

    Example:
        # place legend outside the ax
        cg.ax_heatmap.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=6)
        # place legend in the top right corner of the figure
        cg.ax_heatmap.legend(handles=legend_elements, bbox_to_anchor=(1, 1), bbox_transform=cg.ax_heatmap.figure.transFigure, fontsize=6)
    Args:
        label_to_color:

    Returns:

    """
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor=color, label=label) for label, color in label_to_color.items()]
    return legend_elements


import numpy as np


def get_grid_dims(n_samples) -> (int, int):
    n_row = int(np.ceil(np.sqrt(n_samples)))
    n_col = n_samples // n_row
    return n_row, n_col
