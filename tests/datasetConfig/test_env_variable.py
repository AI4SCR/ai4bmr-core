from ai4bmr_core.data_models.Dataset import Dataset
from pathlib import Path
import os

print("Hello", os.getcwd())


def test_env_variable():
    config = Dataset()
    assert config.base_dir == Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner"
    )
