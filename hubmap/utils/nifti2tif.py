import tifffile
from hubmap.utils.utils import load_filepaths, load_nifti
from os.path import join
from tqdm import tqdm
import shutil
from pathlib import Path


def nifti2tif(load_dir, save_dir, extension):
    shutil.rmtree(save_dir, ignore_errors=True)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    names = load_filepaths(load_dir, return_path=False, return_extension=False, extension=extension)

    for name in tqdm(names):
        image = load_nifti(join(load_dir, "{}.nii.gz".format(name)))
        tifffile.imwrite(join(save_dir, "{}.tif".format(name)), image)


if __name__ == "__main__":
    load_dir = "/home/k539i/Documents/datasets/original/hubmap-hacking-the-human-vasculature/preprocessed/train/images"
    save_dir = "/home/k539i/Documents/datasets/original/hubmap-hacking-the-human-vasculature/preprocessed/train/images_tif"
    extension = "_0000.nii.gz"

    nifti2tif(load_dir, save_dir, extension)

    load_dir = "/home/k539i/Documents/datasets/original/hubmap-hacking-the-human-vasculature/preprocessed/train/border_core"
    save_dir = "/home/k539i/Documents/datasets/original/hubmap-hacking-the-human-vasculature/preprocessed/train/border_core_tif"
    extension = ".nii.gz"

    nifti2tif(load_dir, save_dir, extension)

    load_dir = "/home/k539i/Documents/datasets/original/hubmap-hacking-the-human-vasculature/preprocessed/train/semantic_seg"
    save_dir = "/home/k539i/Documents/datasets/original/hubmap-hacking-the-human-vasculature/preprocessed/train/semantic_seg_tif"
    extension = ".nii.gz"

    nifti2tif(load_dir, save_dir, extension)