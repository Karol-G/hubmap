import numpy as np
from scipy.ndimage import label as nd_label
import cc3d
from tqdm import tqdm
import copy
from acvl_utils.miscellaneous.ptqdm import ptqdm
import numpy_indexed as npi
from skimage.morphology import disk
from skimage.morphology import dilation
from scipy.ndimage.morphology import distance_transform_edt
from typing import Tuple, Optional, Type, Any
import json
import tifffile
from os.path import join
from pathlib import Path
import shutil
from hubmap.utils.utils import load_filepaths
from skimage.transform import resize
from scipy.ndimage import binary_fill_holes


def postprocess_border_core2instance(load_dir, save_dir, core_labels, border_label, resize_factor, min_abs_instance_size):
    shutil.rmtree(save_dir, ignore_errors=True)
    Path(join(save_dir, "instance_seg")).mkdir(parents=True, exist_ok=True)
    Path(join(save_dir, "semantic_seg")).mkdir(parents=True, exist_ok=True)

    names = load_filepaths(load_dir, return_path=False, return_extension=False)
    instance_classes = {}

    for name in tqdm(names):
        border_core = tifffile.imread(join(load_dir, "{}.tif".format(name)))
        instance_seg, num_instances, name_instance_classes, semantic_seg = multiclass_border_core2multiclass_instance(border_core, core_labels=core_labels, border_label=border_label, min_abs_instance_size=min_abs_instance_size)
        instance_classes[name] = name_instance_classes
        if resize_factor is not None:
            inverse_resize_factor = 1 / resize_factor
            instance_seg = resize(instance_seg, (instance_seg.shape[0]*inverse_resize_factor, instance_seg.shape[1]*inverse_resize_factor), order=1, anti_aliasing=False, preserve_range=True)
            semantic_seg = resize(semantic_seg, (semantic_seg.shape[0]*inverse_resize_factor, semantic_seg.shape[1]*inverse_resize_factor), order=1, anti_aliasing=False, preserve_range=True)
        tifffile.imwrite(join(save_dir, "instance_seg", "{}.tif".format(name)), instance_seg)
        tifffile.imwrite(join(save_dir, "semantic_seg", "{}.tif".format(name)), semantic_seg)
    
    with open(join(save_dir, 'instance_classes.json'), 'w') as json_file:
        json.dump(instance_classes, json_file)


def multiclass_border_core2multiclass_instance(border_core: np.ndarray, processes: Optional[int] = None, progressbar: bool = False, dtype: Type = np.uint8, core_labels: Tuple[int] = (1,), border_label: int = 2, min_abs_instance_size: int = 4) -> Tuple[np.ndarray, int]:
    """
    Convert the border-core segmentation of an entire image into an instance segmentation.

    Args:
        border_core (np.ndarray): The border-core segmentation of the entire image.
        processes (Optional[int], default=None): Number of processes to use. If None, it uses a single process.
        progressbar (bool, default=True): Whether to show progress bar.
        dtype (Type, default=np.uint16): The data type for the output segmentation.
        core_labels: The core labels.
        border_label: The border label.

    Returns:
        Tuple[np.ndarray, int]: The instance segmentation of the entire image, Number of instance_seg.
    """

    component_seg = cc3d.connected_components(border_core > 0)
    component_seg = component_seg.astype(dtype)
    instance_seg = np.zeros_like(border_core, dtype=dtype)
    num_instances = 0
    props = {i: bbox for i, bbox in enumerate(cc3d.statistics(component_seg)["bounding_boxes"])}
    del props[0]

    border_core_component2instance = border_core_component2instance_dilation

    if processes is None or processes == 0:
        for index, (label, bbox) in enumerate(tqdm(props.items(), desc="Border-Core2Instance", disable=not progressbar)):
            filter_mask = component_seg[bbox[:2]] == label
            border_core_patch = copy.deepcopy(border_core[bbox[:2]])
            border_core_patch[filter_mask != 1] = 0
            # Multiclass to single class: Set all core labels to 1 and the border label to 2
            shape = border_core_patch.shape
            border_core_patch = npi.remap(border_core_patch.flatten(), list(core_labels), [1] * len(core_labels))
            border_core_patch = border_core_patch.reshape(shape)
            border_core_patch[border_core_patch == border_label] = 2
            instances_patch = border_core_component2instance(border_core_patch).astype(dtype)
            instances_patch[instances_patch > 0] += num_instances
            num_instances = max(num_instances, np.max(instances_patch))
            patch_labels = np.unique(instances_patch)
            patch_labels = patch_labels[patch_labels > 0]
            for patch_label in patch_labels:
                instance_seg[bbox[:2]][instances_patch == patch_label] = patch_label
    else:
        border_core_patches = []
        for index, (label, bbox) in enumerate(props.items()):
            filter_mask = component_seg[bbox[:2]] == label
            border_core_patch = copy.deepcopy(border_core[bbox[:2]])
            border_core_patch[filter_mask != 1] = 0
            # Multiclass to single class: Set all core labels to 1 and the border label to 2
            shape = border_core_patch.shape
            border_core_patch = npi.remap(border_core_patch.flatten(), list(core_labels), [1] * len(core_labels))
            border_core_patch = border_core_patch.reshape(shape)
            border_core_patch[border_core_patch == border_label] = 2
            border_core_patches.append(border_core_patch)

        instances_patches = ptqdm(border_core_component2instance, border_core_patches, processes, desc="Border-Core2Instance", disable=not progressbar)

        for index, (label, bbox) in enumerate(tqdm(props.items())):
            instances_patch = instances_patches[index].astype(dtype)
            instances_patch[instances_patch > 0] += num_instances
            num_instances = max(num_instances, int(np.max(instances_patch)))
            patch_labels = np.unique(instances_patch)
            patch_labels = patch_labels[patch_labels > 0]
            for patch_label in patch_labels:
                instance_seg[bbox[:2]][instances_patch == patch_label] = patch_label

    instance_seg = filter_small_particles(instance_seg, min_abs_instance_size)

    instance_seg = hole_filling(instance_seg)

    # Relabel instances sequentially
    if instance_seg.any():
        instances = np.unique(instance_seg)
        instances = instances[instances > 0]
        shape = instance_seg.shape
        instance_seg = npi.remap(instance_seg.flatten(), instances, range(1, len(instances) + 1))
        instance_seg = instance_seg.reshape(shape)
        instances = np.unique(instance_seg)
        instances = instances[instances > 0]

    instance_classes = compute_instance_classes(instance_seg, border_core, border_label)

    semantic_seg = np.zeros_like(instance_seg)
    if instance_seg.any():
        for instance in instances:
            semantic_seg[instance_seg == instance] = instance_classes[instance]

    return instance_seg, num_instances, instance_classes, semantic_seg


def border_core_component2instance_dilation(patch: np.ndarray, core_label: int = 1, border_label: int = 2) -> np.ndarray:
    """
    Convert a patch that consists of an entire connected component of the border-core segmentation into an instance segmentation using a morphological dilation operation.

    This method starts with the core instance_seg and progressively dilates them until all border voxels are covered, hence performing the instance segmentation.

    :param patch: An entire connected component patch of the border-core segmentation.
    :param core_label: The core label.
    :param border_label: The border label.
    :return: The instance segmentation of this connected component patch.
    """
    core_instances = np.zeros_like(patch, dtype=np.uint16)
    num_instances = nd_label(patch == core_label, output=core_instances)
    if num_instances == 0:
        return patch
    patch, core_instances, num_instances = remove_small_cores(patch, core_instances, core_label, border_label)
    core_instances = np.zeros_like(patch, dtype=np.uint16)
    num_instances = nd_label(patch == core_label, output=core_instances)  # remove_small_cores invalidates the previous core_instances, so recompute it. The computation time is neglectable.
    if num_instances == 0:
        return patch
    instance_seg = copy.deepcopy(core_instances)
    border = patch == border_label
    while np.sum(border) > 0:
        ball_here = disk(3)

        dilated = dilation(core_instances, ball_here)
        dilated[patch == 0] = 0
        diff = (core_instances == 0) & (dilated != core_instances)
        instance_seg[diff & border] = dilated[diff & border]
        border[diff] = 0
        core_instances = dilated

    return instance_seg


def remove_small_cores(
    patch: np.ndarray, 
    core_instances: np.ndarray, 
    core_label: int, 
    border_label: int, 
    min_distance: float = 1, 
    min_ratio_threshold: float = 0.95,  # 0.95 
    max_distance: float = 3, 
    max_ratio_threshold: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, int]: 
    """
    Removes small cores in a patch based on the distance transform of the core label.

    Args:
        patch (np.ndarray): An entire connected component patch of the border-core segmentation.
        core_instances (np.ndarray): The labeled core instance_seg in the patch.
        core_label (int): The label for cores.
        border_label (int): The label for borders.
        min_distance (float, default=1): The minimum distance for removal.
        min_ratio_threshold (float, default=0.95): The minimum ratio threshold for removal.
        max_distance (float, default=3): The maximum distance for removal.
        max_ratio_threshold (float, default=0.0): The maximum ratio threshold for removal.

    Returns:
        Tuple[np.ndarray, np.ndarray, int]: The updated patch after removing small cores, the updated core instance_seg, and the number of cores.
    """

    distances = distance_transform_edt(patch == core_label)
    core_ids = np.unique(core_instances)

    core_ids_to_remove = []
    for core_id in core_ids:
        core_distances = distances[core_instances == core_id]
        num_min_distances = np.count_nonzero(core_distances <= min_distance)
        num_max_distances = np.count_nonzero(core_distances >= max_distance)
        num_core_voxels = np.count_nonzero(core_instances == core_id)
        min_ratio = num_min_distances / num_core_voxels
        max_ratio = num_max_distances / num_core_voxels
        if (min_ratio_threshold is None or min_ratio >= min_ratio_threshold) and (max_ratio_threshold is None or max_ratio <= max_ratio_threshold):
            core_ids_to_remove.append(core_id)

    num_cores = len(core_ids) - len(core_ids_to_remove)

    if len(core_ids_to_remove) > 0:
        # print("Filtered cores: ", len(core_ids_to_remove))
        target_values = np.zeros_like(core_ids_to_remove, dtype=int)
        shape = patch.shape
        core_instances = npi.remap(core_instances.flatten(), core_ids_to_remove, target_values)
        core_instances = core_instances.reshape(shape)

        patch[(patch == core_label) & (core_instances == 0)] = border_label

    return patch, core_instances, num_cores


def filter_small_particles(
    instance_seg: Any,
    min_abs_instance_size: float
) -> Any:
    """
    Filter out small particles from the instance predictions.

    Args:
        instance_seg: The instance segmentation.
        min_abs_instance_size: The minimum absolute instance size used for filtering.

    Returns:
        instance_seg: The filtered instance segmentation.
    """
    if min_abs_instance_size is None:
        return instance_seg

    instances, counts = np.unique(instance_seg, return_counts=True)
    counts = counts[instances > 0]
    instances = instances[instances > 0]
    instances_to_remove = instances[counts < min_abs_instance_size]

    if len(instances_to_remove) > 0:
        shape = instance_seg.shape
        instance_seg = npi.remap(instance_seg.flatten(), instances_to_remove, [0] * len(instances_to_remove))
        instance_seg = instance_seg.reshape(shape)

    return instance_seg


def hole_filling(instance_seg):
    instances = np.unique(instance_seg)
    instances = instances[instances > 0]

    for instance in instances:
        mask = instance_seg == instance
        mask = binary_fill_holes(mask)
        instance_seg[mask == 1] = instance

    return instance_seg


def compute_instance_classes(instance_seg, border_core, border_label, default_label=1):
    border_core[border_core == border_label] = 0
    instances = np.unique(instance_seg)
    instances = instances[instances > 0]

    instance_classes = {}
    for instance in instances:
        core_values = border_core[instance_seg == instance]
        core_values, counts = np.unique(core_values, return_counts=True)
        counts = counts[core_values > 0]
        core_values = core_values[core_values > 0]
        if len(core_values) == 0:
            core_value = default_label
        else:
            core_value = core_values[np.argmax(counts)]
        instance_classes[int(instance)] = int(core_value)
    return instance_classes


if __name__ == "__main__":
    load_dir = "/home/k539i/Documents/datasets/original/hubmap-hacking-the-human-vasculature/predictions/border_core/"
    save_dir = "/home/k539i/Documents/datasets/original/hubmap-hacking-the-human-vasculature/predictions/postprocessed/"
    core_labels = (1, 2)
    border_label = 3
    resize_factor = None
    min_abs_instance_size = 10

    postprocess_border_core2instance(load_dir, save_dir, core_labels, border_label, resize_factor, min_abs_instance_size)