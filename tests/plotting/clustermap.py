from ai4bmr_core.plotting.clustermap import clustermap
import pandas as pd
import numpy as np

num_rows, num_cols = 5, 3
data = pd.DataFrame(np.random.randn(num_rows, num_cols), columns=[f'col_{i}' for i in range(num_cols)])
row_metadata = pd.DataFrame(np.random.randn(num_rows), columns=['sample_id'], index=data.index)
col_metadata = pd.DataFrame(np.random.randint(0, 2, (num_cols, 1)), index=data.columns, columns=['type_id'])

# %%
color_maps = {
    'sample_id': None,
    'type_id': None
}

# get colormaps
# normalize values
# define legends
#

# %%
import pandas as pd
def clustermap(data,
               row_data: pd.DataFrame = None,
               col_data: pd.DataFrame = None,
               num_obs: int = None,
               include_in_legend: list[str] = None,
               color_maps: dict = None,
               ):

    import seaborn as sns
    from ai4bmr_core.plotting.utils import create_color_maps, create_legends, map_row_annotations_to_colors
    from ai4bmr_core.utils.plotting import add_legends_to_fig

    if num_obs:
        num_cells = min(num_obs, len(data))
        pdat = data.sample(num_cells, random_state=0)
    else:
        pdat = data

    row_color_legends = row_color_legends or []

    if metadata is not None:
        color_maps = color_maps or create_color_maps(data=metadata)
        row_colors = map_row_annotations_to_colors(metadata, color_maps)
    else:
        row_colors = None

    cg = sns.clustermap(data=pdat, row_colors=row_colors)

    if row_colors is not None and row_color_legends is not None:
        legends = create_legends(row_annotations=metadata[row_color_legends], color_maps=color_maps)
        add_legends_to_fig(
            cg.figure, legends, pos_y=0.99, pos_x=0.01, pad_x=10, pad_y=10, wrap=4
        )

    return cg

# %%