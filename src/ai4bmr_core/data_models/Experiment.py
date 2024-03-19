from pathlib import Path

from dotenv import find_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .MixIns import CreateFolderHierarchy
from .Project import Project


class Experiment(BaseSettings, CreateFolderHierarchy):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(".env", usecwd=True),
        env_prefix="AI4BMR_EXPERIMENT_",
        extra="ignore",
    )

    project: Project
    name: str
    # TODO: do we really want to support providing a configurable experiment_dir?
    base_dir: None | Path = None

    @computed_field
    @property
    def experiment_dir(
        self,
    ) -> Path:
        if self.base_dir:
            return self.base_dir
        else:
            return self.project.experiments_dir / self.name

    @computed_field
    @property
    def configs_dir(self) -> Path:
        return self.experiment_dir / "00_configurations"

    @computed_field
    @property
    def samples_dir(self) -> Path:
        return self.experiment_dir / "01_samples"

    @computed_field
    @property
    def models_dir(self) -> Path:
        return self.experiment_dir / "02_models"

    @computed_field
    @property
    def predictions_dir(self) -> Path:
        return self.experiment_dir / "03_predictions"

    @computed_field
    @property
    def results_dir(self) -> Path:
        return self.experiment_dir / "04_results"

    def get_model_config_path(self, model_name: str) -> Path:
        return self.configs_dir / f"model_{model_name}.yaml"

    def get_data_config_path(self) -> Path:
        return self.configs_dir / f"data.yaml"

    def get_sample_path(
        self, stage: str, sample_name: str, suffix: str = ".json"
    ) -> Path:
        return self.samples_dir / stage / f"{sample_name}{suffix}"

    def get_model_dir(self, model_name: str) -> Path:
        return self.models_dir / model_name

    def get_prediction_path(
        self, dataset_name: str, model_name: str, sample_name: str
    ) -> Path:
        return self.prediction_dir / dataset_name / model_name / sample_name

    def get_result_path(
        self, dataset_name: str, model_name: str, sample_name: str
    ) -> Path:
        return self.result_dir / model_name / dataset_name / sample_name
