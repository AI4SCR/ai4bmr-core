from pathlib import Path

from pydantic import computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .MixIns import CreateFolderHierarchy


class Project(BaseSettings, CreateFolderHierarchy):
    model_config = SettingsConfigDict(
        env_prefix="AI4BMR_PROJECT_",
        extra="ignore",
    )

    base_dir: Path
    name: str = None

    @field_validator("base_dir")
    @classmethod
    def expand_base_dir(cls, value) -> Path:
        return Path(value).expanduser()

    @computed_field
    @property
    def configurations_dir(self) -> Path:
        return self.base_dir / "00_configurations"

    @computed_field
    @property
    def datasets_dir(self) -> Path:
        return self.base_dir / "01_datasets"

    @computed_field
    @property
    def experiments_dir(self) -> Path:
        return self.base_dir / "02_experiments"
