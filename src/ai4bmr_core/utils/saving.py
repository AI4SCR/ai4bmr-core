from pathlib import Path

import numpy as np


def save_mask(mask: np.ndarray, save_path: Path):
    from skimage.io import imsave

    if mask.max() < 65536:
        mask = mask.astype("uint16")
    else:
        mask = mask.astype("uint32")
    imsave(save_path, mask, check_contrast=False, plugin="tifffile", compression="deflate")


def save_image(img: np.ndarray, save_path: Path):
    from skimage.io import imsave

    imsave(save_path, img, plugin="tifffile", compression="deflate")
