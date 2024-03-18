from pathlib import Path

from dotenv import find_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import computed_field

from .MixIns import CreateFolderHierarchy


class Project(BaseSettings, CreateFolderHierarchy):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(".env", usecwd=True),
        env_prefix="AI4BMR_PROJECT_",
        extra="ignore",
    )

    base_dir: Path
    name: str

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
