# %%
import pandas as pd
def clustermap(data,
               metadata: pd.DataFrame = None,
               num_cells: int = None,
               row_color_legends: list[str] = None,
               color_maps: dict = None,
               ):
    import seaborn as sns
    from utils import create_color_maps, create_legends, map_row_annotations_to_colors
    from ai4bmr_core.utils.plotting import add_legends_to_fig

    if num_cells:
        num_cells = min(50_000, len(data))
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