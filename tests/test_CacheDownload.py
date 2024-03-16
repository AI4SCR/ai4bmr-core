from ai4bmr_core.datasets.Dataset import BaseDataset
from pathlib import Path
import pandas as pd


class D(BaseDataset):
    # required fields
    # model_config = super().model_config.update(dict(arbitrary_types_allowed=True))
    _id: str = "Hello"
    _name: str = "World"
    _data: None | pd.DataFrame = None
    # optional fields
    raw_dir: Path = Path("~/Downloads/ai4bmr-core/").expanduser()
    processed_dir: Path = Path("~/Downloads/ai4bmr-core/").expanduser()
    _urls: dict[str, str] = {
        "download.zip": "https://www.stats.govt.nz/assets/Uploads/Business-employment-data/Business-employment-data-December-2023-quarter/Download-data/business-employment-data-december-2023-quarter.zip"
    }

    def load(self):
        return pd.read_csv(self.raw_dir / "download.zip")

    @property
    def processed_files(self) -> list[Path]:
        return [self.processed_dir / "data.csv"]

    def load_cache(self) -> pd.DataFrame:
        # NOTE: handle loading of `processed_files` here
        return pd.read_csv(self.processed_files[0])

    def save_cache(self):
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(self.processed_dir / "data.csv")


def test_download_and_cache():
    d = D()
    assert isinstance(d._data, pd.DataFrame)
    assert d.force_download == False
