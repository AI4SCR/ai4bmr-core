from ai4bmr_core.datasets.DatasetConfig import DatasetConfig


def test_DatasetConfig():
    config = DatasetConfig()
    assert (
        config._data_dir == "/Users/adrianomartinelli/data/ai4src/graph-concept-learner"
    )
