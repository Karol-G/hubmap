import tifffile
from hubmap.utils.utils import load_filepaths, load_nifti
from os.path import join
from tqdm import tqdm


def nifti2tif(load_dir, save_dir):
    names = load_filepaths(load_dir, return_path=False, return_extension=False, extension=".nii.gz")

    for name in tqdm(names):
        image = load_nifti(join(load_dir, "{}.nii.gz".format(name)))
        tifffile.imwrite(join(save_dir, "{}.tif".format(name)), image)


if __name__ == "__main__":
    load_dir = "/home/k539i/Documents/network_drives/cluster-checkpoints/nnUNetV2/Dataset503_hubmap/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation"
    save_dir = "/home/k539i/Documents/network_drives/cluster-checkpoints/nnUNetV2/Dataset503_hubmap/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation_tif"

    nifti2tif(load_dir, save_dir)