import pandas as pd

from ai4bmr_core.datasets.Dataset import BaseDataset
from pathlib import Path


def test_string():
    class D(BaseDataset):
        _id: str = "Hello"
        _name: str = "World"
        _data: str = "None"

        def load(self):
            return "Hello World"

    d = D()
    assert d._data == "Hello World"
    assert d.data_dir == Path.home() / ".cache" / "ai4bmr" / "datasets" / "World"
    assert d.raw_dir == d.data_dir / "raw"
    assert d.processed_dir == d.data_dir / "processed"
    assert d.force_download is False


def test_pandas():
    class D(BaseDataset):
        _id: str = "Hello"
        _name: str = "World"
        _data: pd.DataFrame = pd.DataFrame()

        def load(self):
            return pd.DataFrame()

    d = D()
    assert (d._data == pd.DataFrame()).all().all()
    assert d.data_dir == Path.home() / ".cache" / "ai4bmr" / "datasets" / "World"
    assert d.raw_dir == d.data_dir / "raw"
    assert d.processed_dir == d.data_dir / "processed"
    assert d.force_download is False


def test_custom_dtype():
    class CustomType:
        value: int = 1

        def __eq__(self, other):
            if isinstance(other, CustomType):
                return self.value == other.value
            return False

    class D(BaseDataset):
        _id: str = "Hello"
        _name: str = "World"
        _data: CustomType = CustomType()

        def load(self):
            return CustomType()

    d = D()
    assert d._data == CustomType()
    assert d.data_dir == Path.home() / ".cache" / "ai4bmr" / "datasets" / "World"
    assert d.raw_dir == d.data_dir / "raw"
    assert d.processed_dir == d.data_dir / "processed"
    assert d.force_download is False

    # model_config = super().model_config.update(dict(arbitrary_types_allowed=True))
