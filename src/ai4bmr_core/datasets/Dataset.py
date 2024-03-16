from pathlib import Path
import shutil
from abc import ABC, abstractmethod
import logging
from .DatasetConfig import DatasetConfig


class BaseDataset(ABC, DatasetConfig):

    def __init__(self, force_download: bool = False):
        super().__init__()
        self.force_download = force_download
        if self.force_download and self.raw_dir.exists():
            # NOTE: we delete the `raw_dir` if  `force_download` is True to ensure that all files are newly
            #  downloaded and not just some of them in case of an exception occurs during the download.
            #  Furthermore, we check if the folder exists and do not use `is_downloaded` as this is only True if the
            #  previously attempted downloads were successful.
            shutil.rmtree(self.raw_dir)

        if self.urls and (self.force_download or not self.is_downloaded):
            self.download()

        if self.is_cached:
            logging.info('loading from cache')
            self.data = self.load_cache()
        else:
            logging.info('load')
            self.data = self.load()
            self.save_cache()

    @abstractmethod
    def load(self):
        raise NotImplementedError()

    def load_cache(self) -> any:
        # NOTE: handle loading of `processed_files` here
        # return self.processed_files
        return None

    def save_cache(self) -> None:
        if self.processed_dir:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
        # add logic to save `processed_files` here

    def download(self):
        try:
            for file_name, url in self.urls.items():
                if (self.raw_dir / file_name).exists():
                    continue
                logging.info(f'Downloading from {url} to {self.raw_dir / file_name}')
                self._download_progress(url, self.raw_dir / file_name)
        except Exception as e:
            # if an exception occurs, and the raw_dir is empty, delete it
            # NOTE: there is the case where the raw_dir is not empty, but only some files have been downloaded
            if list(self.raw_dir.iterdir()) == 0:
                shutil.rmtree(self.raw_dir)

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
                        total=total if total is None else int(total)
                ) as t, fpath.open('wb') as f:
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
