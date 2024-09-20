from pathlib import Path
import os


def create_symlink(src_path: Path, dest_path: Path, relative=True):
    if relative:
        # src_path = src_path.relative_to(dest_path)  # this requires src_path to be a subpath of dest_path
        src_path = os.path.relpath(src_path, start=dest_path.parent)
        dest_path.symlink_to(src_path)


def symlink_dir_files(
    src_dir: Path, dest_dir: Path, relative=True, regex=None, glob=None
) -> list[Path]:
    import re

    # perform glob search
    if glob is not None:
        list_of_files = src_dir.glob(glob)
    else:
        list_of_files = src_dir.iterdir()
    # perform regex search
    if regex is not None:
        list_of_files = filter(lambda x: re.search(regex, x.name), list_of_files)

    symlinks = []
    for src_path in list_of_files:
        dest_path = dest_dir / src_path.name
        create_symlink(src_path, dest_path, relative=relative)
        symlinks.append(dest_path)
    return symlinks
