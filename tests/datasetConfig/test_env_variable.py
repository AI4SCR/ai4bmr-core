import os

from ai4bmr_core.datasets.DatasetConfig import DatasetConfig
from pathlib import Path
import os

print("Hello", os.getcwd())


def test_env_variable():
    config = DatasetConfig()
    assert config.data_dir == Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner"
    )
