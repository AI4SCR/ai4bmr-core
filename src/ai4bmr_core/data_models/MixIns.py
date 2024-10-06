import pickle
from pathlib import Path


class CreateFolderHierarchy:
    def create_folder_hierarchy(self):
        dirs = self.get_dirs()
        for d in dirs:
            path = getattr(self, d)
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_dirs(cls):
        return [i for i in cls.__dict__["model_computed_fields"] if i.endswith("_dir")]


class JsonIO:
    @classmethod
    def model_validate_from_json(cls, path):
        import json

        with open(path, "r") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def model_dump_to_json(self, path: Path | str):
        with open(path, "w") as f:
            f.write(self.model_dump_json())


class YamlIO:
    # TODO: add typing
    # NOTE: why do we support overriding fields with kwargs?
    @classmethod
    def model_validate_from_yaml(cls, path, **kwargs):
        import yaml

        with open(path) as f:
            items = yaml.safe_load(f)
            if isinstance(items, list):
                return [cls(**{**i, **kwargs}) for i in items]
            else:
                return cls(**{**items, **kwargs})

    def model_dump_to_yaml(self, path: Path | str):
        import yaml

        model_dict = self.dict()
        with open(path, "w") as file:
            yaml.dump(model_dict, file, sort_keys=True)


class PickleIO:
    def to_pickle(
        self, path: Path | str, exists: str = "raise", check_integrity: bool = True
    ):
        if check_integrity and hasattr(self, "check_integrity"):
            # TODO: raise error if `check_integrity = True` but method does not exist
            assert self.check_integrity()

        if Path(path).exists():
            if exists == "skip":
                return
            elif exists == "overwrite":
                pass
            elif exists == "raise":
                raise FileExistsError(f"{path} already exists")

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, path):
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)
