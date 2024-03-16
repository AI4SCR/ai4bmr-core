import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from .DatasetConfig import DatasetConfig
from ..log.log import logger


class BaseDataset(ABC, DatasetConfig):
    def __init__(
        self,
        *,
        data_dir: None | Path | str = None,
        raw_dir: Path = None,
        processed_dir: Path = None,
        force_download: bool = False,
        force_caching: bool = False,
        **kwargs,
    ):
        """
        The `BaseDataset` class is an abstract class that provides a template for creating new datasets.
        It consumes the user fields of the DatasetConfig class.

        If `processed_files` is defined on the subclass, caching is enabled and the `processed_dir` is used to store
        the output of the `load` function. If `processed_files` is not defined, caching is disabled.

        If `urls` is defined, downloading is enabled and the `raw_dir` is used to store the downloaded files.

        Args:
            data_dir:
            raw_dir:
            processed_dir:
            force_download:
            force_caching:
            **kwargs:
        """

        if data_dir is None and raw_dir.parent != processed_dir.parent:
            logger.warning(
                f"""
                Divergent path configuration. You are saving data in different directories.
                raw_dir: {raw_dir}
                processed_dir: {processed_dir}
                """
            )
        if data_dir is not None and (raw_dir is not None or processed_dir is not None):
            logger.warning(
                f"""
                Paths are over-configured. You are setting `data_dir` and `raw_dir` or `processed_dir` at the same time.
                data_dir (\033[93mignored\033[0m): {data_dir}
                raw_dir (\033[92mused\033[0m): {raw_dir}
                processed_dir (\033[92mused\033[0m): {processed_dir}
                """
            )

        # note: we initialize the `BaseDataset` with default values for `raw_dir` and `processed_dir`
        super().__init__(
            # we pass these values directly, because they do not need to be post-processed like the other fields
            force_download=force_download,
            force_caching=force_caching,
            **kwargs,
        )
        # note: after super(), we can access the initialized values and overwrite them if necessary

        # note: if the user defines a data_dir use it. If not, use the default value defined on the subclass.
        #   If the subclass does not define a default value, use the default value defined on the DatasetConfig class.
        self.data_dir = data_dir if data_dir else self._data_dir

        # note: see explanation for `data_dir`
        self.processed_dir = processed_dir if processed_dir else self._processed_dir
        # self.processed_dir = self.processed_dir if self.processed_dir else self.data_dir / "processed"

        # note: see explanation for `data_dir`
        self.raw_dir = raw_dir if raw_dir else self._raw_dir
        # self.raw_dir = self.raw_dir if self.raw_dir else self.data_dir / "raw"

        # self.force_download = force_download
        if self.force_download and self.raw_dir.exists():
            # NOTE: we delete the `raw_dir` if  `force_download` is True to ensure that all files are newly
            #  downloaded and not just some of them in case of an exception occurs during the download.
            #  Furthermore, we check if the folder exists and do not use `is_downloaded` as this is only True if the
            #  previously attempted downloads were successful.
            shutil.rmtree(self.raw_dir)

        if self._urls and (self.force_download or not self.is_downloaded):
            self.download()

        if self.is_cached and not self.force_caching:
            logging.info("loading from cache")
            self._data = self.load_cache()
        else:
            logging.info("load")
            self._data = self.load()
            self.save_cache()

    @abstractmethod
    def load(self):
        raise NotImplementedError()

    def load_cache(self) -> any:
        # NOTE: handle loading of `processed_files` here
        # files = self.processed_files
        return None

    def save_cache(self) -> None:
        if self.processed_dir:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
        # add logic to save `processed_files` here

    def download(self):
        try:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            for file_name, url in self._urls.items():
                if (self.raw_dir / file_name).exists():
                    continue
                logging.info(f"Downloading from {url} to {self.raw_dir / file_name}")
                self._download_progress(url, self.raw_dir / file_name)
        except Exception as e:
            # if an exception occurs, and the raw_dir is empty, delete it
            # NOTE: there is the case where the raw_dir is not empty, but only some files have been downloaded
            if list(self.raw_dir.iterdir()) == 0:
                shutil.rmtree(self.raw_dir)
            raise e

    @staticmethod
    def _download_progress(url: str, fpath: Path):
        from tqdm import tqdm
        from urllib.request import urlopen, Request

        blocksize = 1024 * 8
        blocknum = 0

        try:
            with urlopen(Request(url, headers={"User-agent": "dataset-user"})) as rsp:
                total = rsp.info().get("content-length", None)
                with tqdm(
                    unit="B",
                    unit_scale=True,
                    miniters=1,
                    unit_divisor=1024,
                    total=total if total is None else int(total),
                ) as t, fpath.open("wb") as f:
                    block = rsp.read(blocksize)
                    while block:
                        f.write(block)
                        blocknum += 1
                        t.update(len(block))
                        block = rsp.read(blocksize)
        except (KeyboardInterrupt, Exception):
            # Make sure file doesn’t exist half-downloaded
            if fpath.is_file():
                fpath.unlink()
            raise
