import pandas as pd

from ai4bmr_core.datasets.Dataset import BaseDataset


def test_string():
    class D(BaseDataset):
        _id: str = "Hello"
        _name: str = "World"
        data: str = "None"

        def load(self):
            return "Hello World"

    d = D()
    assert d.data == "Hello World"
    assert d.force_download is False


def test_pandas():
    class D(BaseDataset):
        _id: str = "Hello"
        _name: str = "World"
        data: pd.DataFrame = pd.DataFrame()

        def load(self):
            return pd.DataFrame({"Hello": [1, 2, 3], "World": [4, 5, 6]})

    d = D()
    assert (
        (d.data == pd.DataFrame({"Hello": [1, 2, 3], "World": [4, 5, 6]})).all().all()
    )
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
        data: CustomType = CustomType()

        def load(self):
            return CustomType()

    d = D()
    assert d.data == CustomType()
    assert d.force_download == False

    # model_config = super().model_config.update(dict(arbitrary_types_allowed=True))
