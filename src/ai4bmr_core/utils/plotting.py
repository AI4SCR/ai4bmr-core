def legend_from_dict(label_to_color: dict):
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor=color, label=label) for label, color in label_to_color.items()]
    return legend_elements
