from pathlib import Path

from dotenv import find_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import computed_field


class DatasetConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(".env", usecwd=True),
        # arbitrary_types_allowed=True,
        # protected_namespaces=("settings_",),
        extra="ignore",
    )

    # dataset fields, required
    _id: str  # protected, this should not be set by the user
    _name: str  # protected, this should not be set by the user
    _data: None  # note: type defined on the subclass as we don't know it here yet

    # dataset fields, optional
    _description: str = ""
    _urls: None | dict[str, str] = None
    _data_dir: None | Path = None
    _dataset_dir: None | Path = None
    _raw_dir: None | Path = None
    _processed_dir: None | Path = None

    # user fields, optional
    force_download: bool = False
    force_caching: bool = False

    @computed_field
    @property
    def data_dir(self) -> Path:
        return (
            Path.home() / ".cache" / "ai4bmr"
            if self._data_dir is None
            else Path(self._data_dir)
        )

    @data_dir.setter
    def data_dir(self, value: Path | str):
        self._data_dir = Path(value) if value else None

    @computed_field
    @property
    def dataset_dir(self) -> Path:
        return (
            self._dataset_dir
            if self._dataset_dir
            else self.data_dir / "datasets" / self._name
        )

    @dataset_dir.setter
    def dataset_dir(self, value: Path | str):
        self._dataset_dir = Path(value) if value else None

    @computed_field
    @property
    def raw_dir(self) -> Path:
        return self._raw_dir if self._raw_dir else self.dataset_dir / "01_raw"

    @raw_dir.setter
    def raw_dir(self, value: Path | str):
        self._raw_dir = Path(value) if value else None

    @computed_field
    @property
    def processed_dir(self) -> Path:
        return (
            self._processed_dir
            if self._processed_dir
            else self.dataset_dir / "02_processed"
        )

    @processed_dir.setter
    def processed_dir(self, value: Path | str):
        self._processed_dir = Path(value) if value else None

    @computed_field
    @property
    def raw_files(self) -> list[Path]:
        return (
            [self.raw_dir / i for i in self._urls]
            if self.raw_dir and self._urls
            else []
        )

    @computed_field
    @property
    def is_downloaded(self) -> bool:
        return all([i.exists() for i in self.raw_files]) if self.raw_files else False

    @computed_field
    @property
    def processed_files(self) -> list[Path]:
        # define a list of files that are produced by self.
        return []

    @computed_field
    @property
    def is_cached(self) -> bool:
        return (
            all([i.exists() for i in self.processed_files])
            if self.processed_files
            else False
        )
