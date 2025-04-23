from pathlib import Path

import numpy as np
import tifffile


def save_mask(mask: np.ndarray, save_path: Path):
    assert mask.dtype in ["uint32", "uint16", 'int64', 'int32']

    if mask.max() < 65536:
        mask = mask.astype("uint16")
    else:
        mask = mask.astype("uint32")

    tifffile.imwrite(save_path, mask, compression="deflate")


def save_image(img: np.ndarray, save_path: Path):
    import tifffile

    tifffile.imwrite(save_path, img, compression="deflate")
