# %%
import pandas as pd
import colorcet as cc
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
import pandas as pd
from matplotlib.colors import Normalize, LinearSegmentedColormap, to_rgba, to_rgb
from matplotlib.colorbar import ColorbarBase
from matplotlib.patches import Rectangle
from pandas.api.types import is_numeric_dtype


def create_color_maps(data: pd.DataFrame, as_rgba: bool = False) -> dict:
    to_tuple = to_rgba if as_rgba else to_rgb
    color_maps = {}
    for col in data:
        if data[col].dtype.name == "category":
            n = len(cc.glasbey_category10)
            labels = np.sort(data[col].unique())
            color_map = {
                label: to_tuple(cc.glasbey_category10[i % n])
                for i, label in enumerate(labels)
            }
            color_maps[col] = color_map
        elif is_numeric_dtype(data[col]):
            cmap = LinearSegmentedColormap.from_list(col, cc.linear_kry_0_97_c73)
            color_maps[col] = cmap
        else:
            print(f"WARNING: {col} is not a category or numeric dtype")
    return color_maps


def map_row_annotations_to_colors(row_data: pd.DataFrame, color_maps: dict, dtypes: dict[str, str] = None):
    row_colors = row_data.copy()
    for col in row_data:
        dtype = dtypes.get(col) if dtypes else None
        cmap = color_maps.get(col)
        if dtype == "discrete" or (dtype is None and row_data[col].dtype.name == "category"):
            row_colors[col] = [cmap[v] for v in row_data[col]]
        elif dtype == "continuous" or (dtype is None and is_numeric_dtype(row_data[col])):
            row_colors[col] = [cmap(v) for v in row_data[col]]
    return row_colors


def create_legends(row_annotations: pd.DataFrame, color_maps: dict, dtypes: dict[str, str] = None):
    legends = []
    for col in row_annotations:
        dtype = dtypes.get(col) if dtypes else None
        if dtype == "discrete" or (dtype is None and row_annotations[col].dtype.name == "category"):
            legends.append({"type": "discrete", "label_to_color": color_maps[col], "title": col})
        elif dtype == "continuous" or (dtype is None and is_numeric_dtype(row_annotations[col])):
            legends.append({
                "type": "continuous", "height": 75, "width": 10,
                "vmin": row_annotations[col].min(), "vmax": row_annotations[col].max(),
                "colormap": color_maps[col], "orientation": "vertical", "title": col
            })
    return legends


def clustermap(data,
               metadata: pd.DataFrame = None,
               num_cells: int = None,
               row_color_legends: list[str] = None,
               color_maps: dict = None,
               clustermap_kwargs: dict = None,
               add_legend_kwargs: dict = None,
               ):
    import seaborn as sns
    from ai4bmr_core.utils.plotting import add_legends_to_fig

    clustermap_kwargs = clustermap_kwargs or {}
    add_legend_kwargs = add_legend_kwargs or {}
    add_legend_kwargs = {'pos_x': 0.01, 'pos_y': 0.99, 'pad_x': 10, 'pad_y': 10, 'wrap': 4, **add_legend_kwargs}

    if num_cells:
        num_cells = min(num_cells, len(data))
        pdat = data.sample(num_cells, random_state=0)
    else:
        pdat = data

    row_color_legends = row_color_legends or []

    if metadata is not None:
        row_data = metadata.loc[pdat.index, row_color_legends]
        color_maps = color_maps or create_color_maps(data=row_data)
        row_colors = map_row_annotations_to_colors(row_data, color_maps)
    else:
        row_colors = None

    cg = sns.clustermap(data=pdat, row_colors=row_colors, **clustermap_kwargs)

    if row_colors is not None and row_color_legends is not None:
        legends = create_legends(row_annotations=row_data, color_maps=color_maps)
        add_legends_to_fig(cg.figure, legends, **add_legend_kwargs)

    return cg
