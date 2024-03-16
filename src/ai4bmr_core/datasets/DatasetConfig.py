from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from dotenv import find_dotenv


class DatasetConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(".env", usecwd=True),
        # arbitrary_types_allowed=True,
        # protected_namespaces=("settings_",),
        extra="ignore",
    )

    # required fields
    _id: str  # protected, this should not be set by the user
    _name: str  # protected, this should not be set by the user
    data: None  # note: type defined on the subclass as we don't know it here yet

    # optional fields
    description: str = ""
    # TODO: if data_dir is set, compute ~/.cache/{name}/{raw, processed}_dir
    data_dir: None | Path | str = None
    urls: None | dict[str, str] = None
    raw_dir: None | Path = None
    processed_dir: None | Path = None
    force_download: bool = False
    force_caching: bool = False

    @property
    def raw_files(self) -> list[Path]:
        return (
            [self.raw_dir / i for i in self.urls]
            if self.raw_dir and self.urls
            else None
        )

    @property
    def is_downloaded(self) -> bool:
        return all([i.exists() for i in self.raw_files])

    @property
    def processed_files(self) -> list[Path]:
        # define a list of files that are produced by self.
        return []

    @property
    def is_cached(self) -> bool:
        return (
            all([i.exists() for i in self.processed_files])
            if self.processed_files
            else False
        )
