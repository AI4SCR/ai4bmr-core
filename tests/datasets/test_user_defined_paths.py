from pathlib import Path

from ai4bmr_core.datasets.Dataset import BaseDataset


def test_data_dir_paths():
    class D(BaseDataset):
        _id: str = "Hello"
        _name: str = "World"
        _data: str = ""

        def load(self):
            return "Hello World"

    d = D(data_dir=Path("~/Downloads/ai4bmr-core/").expanduser())
    assert d.data_dir == Path("~/Downloads/ai4bmr-core/").expanduser()
    assert d.dataset_dir == d.data_dir / "datasets" / "World"
    assert d.raw_dir == d.dataset_dir / "01_raw"
    assert d.processed_dir == d.dataset_dir / "02_processed"


def test_dataset_dir_paths():
    class D(BaseDataset):
        _id: str = "Hello"
        _name: str = "World"
        _data: str = ""

        def load(self):
            return "Hello World"

    d = D(dataset_dir=Path("~/Downloads/ai4bmr-core/").expanduser())
    assert d.data_dir == Path.home() / ".cache" / "ai4bmr"
    assert d.dataset_dir == Path("~/Downloads/ai4bmr-core/").expanduser()
    assert d.raw_dir == d.dataset_dir / "01_raw"
    assert d.processed_dir == d.dataset_dir / "02_processed"


def test_raw_processed_paths():
    class D(BaseDataset):
        _id: str = "Hello"
        _name: str = "World"
        _data: str = ""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def load(self):
            return "Hello World"

    d = D(
        raw_dir=Path("~/Downloads/random_path/raw").expanduser(),
        processed_dir=Path("~/processed").expanduser(),
    )
    assert d.data_dir == Path.home() / ".cache" / "ai4bmr"
    assert d.dataset_dir == d.data_dir / "datasets" / "World"
    assert d.raw_dir == Path("~/Downloads/random_path/raw").expanduser()
    assert d.processed_dir == Path("~/processed").expanduser()


def test_dir_raw_processed_paths():
    class D(BaseDataset):
        _id: str = "Hello"
        _name: str = "World"
        _data: str = ""

        def load(self):
            return "Hello World"

    d = D(
        data_dir=Path("~/Downloads/data_dir/").expanduser(),
        raw_dir=Path("~/Downloads/random_path/raw").expanduser(),
        processed_dir=Path("~/processed").expanduser(),
    )
    assert d.data_dir == Path("~/Downloads/data_dir/").expanduser()
    assert d.raw_dir == Path("~/Downloads/random_path/raw").expanduser()
    assert d.processed_dir == Path("~/processed").expanduser()
