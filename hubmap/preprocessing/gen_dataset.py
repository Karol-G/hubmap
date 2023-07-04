import json
import numpy as np
import tifffile
from os.path import join
from pathlib import Path
import shutil
from tqdm import tqdm
from hubmap.utils.utils import load_filepaths, save_nifti
import cv2
from skimage.transform import resize
from acvl_utils.cropping_and_padding.bounding_boxes import pad_bbox, bounding_box_to_slice
from skimage.measure import regionprops
from skimage.morphology import disk
from skimage.morphology import binary_erosion
from acvl_utils.cropping_and_padding.bounding_boxes import regionprops_bbox_to_proper_bbox
from hubmap.utils.normalizer import Zscore
from acvl_utils.miscellaneous.ptqdm import ptqdm


def generate_dataset(polygons_filepath, train_load_dir, save_dir, labels, ignore_label, border_label, include_unannotated=False, border_thickness=5, resize_factor=2, sample_percentage=None, zscore=None, processes=12):
    shutil.rmtree(join(save_dir, "images"), ignore_errors=True)
    shutil.rmtree(join(save_dir, "semantic_seg"), ignore_errors=True)
    shutil.rmtree(join(save_dir, "instance_seg"), ignore_errors=True)
    shutil.rmtree(join(save_dir, "border_core"), ignore_errors=True)
    Path(join(save_dir, "images")).mkdir(parents=True, exist_ok=True)
    Path(join(save_dir, "semantic_seg")).mkdir(parents=True, exist_ok=True)
    Path(join(save_dir, "instance_seg")).mkdir(parents=True, exist_ok=True)
    Path(join(save_dir, "border_core")).mkdir(parents=True, exist_ok=True)
    
    if zscore is None:
        normalizer = Zscore(sample_percentage=sample_percentage, channel_dim=-1, num_channels=3)

    # Process images with annotations
    polygons = {}
    with open(polygons_filepath, 'r') as f:
         polygons_list = list(f)

    if processes is None:
        for json_str in tqdm(polygons_list):
            image_polygons = json.loads(json_str)
            name = image_polygons["id"]
            image = tifffile.imread(join(train_load_dir, "{}.tif".format(name)))
            if zscore is None:
                normalizer.sample(image)

            if resize_factor is not None:
                image = resize(image.astype(np.float64), (image.shape[0]*resize_factor, image.shape[1]*resize_factor, 3), order=3, anti_aliasing=False, preserve_range=True)
                image = np.rint(image).astype(np.uint8)
            if zscore is not None:
                image = image.astype(np.float32)
                image = image - zscore["mean"]
                image = image / zscore["std"]
            save_nifti(join(save_dir, "images", "{}_0000.nii.gz".format(name)), image[..., 0], spacing=(1, 1))
            save_nifti(join(save_dir, "images", "{}_0001.nii.gz".format(name)), image[..., 1], spacing=(1, 1))
            save_nifti(join(save_dir, "images", "{}_0002.nii.gz".format(name)), image[..., 2], spacing=(1, 1))
            
            semantic_seg = np.zeros(image.shape[:2], dtype=np.uint8)
            instance_seg = np.zeros(image.shape[:2], dtype=np.uint8)
            instances = {}
            for instance_label, class_annotation in enumerate(image_polygons["annotations"]):
                instance_label += 1
                new_class_annotation = {}
                new_class_annotation["label"] = labels[class_annotation["type"]]
                coordinates = np.asarray(class_annotation["coordinates"]).squeeze()
                contour = coordinates.reshape((-1, 1, 2))
                if resize_factor is not None:
                    contour *= resize_factor
                cv2.fillPoly(semantic_seg, [contour], color=new_class_annotation["label"])
                cv2.fillPoly(instance_seg, [contour], color=instance_label)
                new_class_annotation["coordinates"] = coordinates.tolist()
                instances[instance_label] = new_class_annotation
            
            save_nifti(join(save_dir, "semantic_seg", "{}.nii.gz".format(name)), semantic_seg)
            save_nifti(join(save_dir, "instance_seg", "{}.nii.gz".format(name)), instance_seg)
            semantic_seg[semantic_seg == labels["unsure"]] = ignore_label[labels["unsure"]]
            border_core = instance2border_core(instance_seg, border_thickness=border_thickness)  # , border_label=border_label, semantic_seg=semantic_seg)
            save_nifti(join(save_dir, "border_core", "{}.nii.gz".format(name)), border_core)
            
            polygons[name] = instances
    else:
        results = ptqdm(process_image, polygons_list, processes, train_load_dir=train_load_dir, save_dir=save_dir, labels=labels, 
                ignore_label=ignore_label, border_label=border_label, border_thickness=border_thickness, resize_factor=resize_factor, zscore=zscore)
        
        for name, instances in results:
            polygons[name] = instances

    if include_unannotated:
        # Process images without annotations
        names = load_filepaths(train_load_dir, return_path=False, return_extension=False)

        for name in tqdm(names):
            if name not in polygons:
                image = tifffile.imread(join(train_load_dir, "{}.tif".format(name)))

                if resize_factor is not None:
                    image = resize(image.astype(np.float64), (image.shape[0]*resize_factor, image.shape[1]*resize_factor, 3), order=3, anti_aliasing=False, preserve_range=True)
                    image = np.rint(image).astype(np.uint8)
                if zscore is not None:
                        image = image.astype(np.float32)
                        image -= zscore["mean"]
                        image /= zscore["std"]
                save_nifti(join(save_dir, "images", "{}_0000.nii.gz".format(name)), image[..., 0], spacing=(1, 1))
                save_nifti(join(save_dir, "images", "{}_0001.nii.gz".format(name)), image[..., 1], spacing=(1, 1))
                save_nifti(join(save_dir, "images", "{}_0002.nii.gz".format(name)), image[..., 2], spacing=(1, 1))
                
                seg = np.full(image.shape[:2], fill_value=labels["unsure"], dtype=np.uint8)
                save_nifti(join(save_dir, "semantic_seg", "{}.nii.gz".format(name)), seg)
                save_nifti(join(save_dir, "instance_seg", "{}.nii.gz".format(name)), seg)
                save_nifti(join(save_dir, "border_core", "{}.nii.gz".format(name)), seg)
                polygons[name] = {1: {"label": labels["unsure"], "coordinates": None}}

    with open(join(save_dir, 'metadata.json'), 'w') as json_file:
        json.dump(polygons, json_file)

    if zscore is None:
        zscore = normalizer.get_zscore()

        with open(join(save_dir, 'zscore.json'), 'w') as json_file:
            json.dump(zscore, json_file)


def process_image(json_str, train_load_dir, save_dir, labels, ignore_label, border_label, border_thickness, resize_factor, zscore):
    image_polygons = json.loads(json_str)
    name = image_polygons["id"]
    image = tifffile.imread(join(train_load_dir, "{}.tif".format(name)))

    if resize_factor is not None:
        image = resize(image.astype(np.float64), (image.shape[0]*resize_factor, image.shape[1]*resize_factor, 3), order=3, anti_aliasing=False, preserve_range=True)
        image = np.rint(image).astype(np.uint8)
    if zscore is not None:
        image = image.astype(np.float32)
        image = image - zscore["mean"]
        image = image / zscore["std"]
    save_nifti(join(save_dir, "images", "{}_0000.nii.gz".format(name)), image[..., 0], spacing=(1, 1))
    save_nifti(join(save_dir, "images", "{}_0001.nii.gz".format(name)), image[..., 1], spacing=(1, 1))
    save_nifti(join(save_dir, "images", "{}_0002.nii.gz".format(name)), image[..., 2], spacing=(1, 1))
    
    semantic_seg = np.zeros(image.shape[:2], dtype=np.uint8)
    instance_seg = np.zeros(image.shape[:2], dtype=np.uint8)
    instances = {}
    for instance_label, class_annotation in enumerate(image_polygons["annotations"]):
            instance_label += 1
            new_class_annotation = {}
            new_class_annotation["label"] = labels[class_annotation["type"]]
            coordinates = np.asarray(class_annotation["coordinates"]).squeeze()
            contour = coordinates.reshape((-1, 1, 2))
            if resize_factor is not None:
                contour *= resize_factor
            cv2.fillPoly(semantic_seg, [contour], color=new_class_annotation["label"])
            cv2.fillPoly(instance_seg, [contour], color=instance_label)
            new_class_annotation["coordinates"] = coordinates.tolist()
            instances[instance_label] = new_class_annotation
    
    save_nifti(join(save_dir, "semantic_seg", "{}.nii.gz".format(name)), semantic_seg)
    save_nifti(join(save_dir, "instance_seg", "{}.nii.gz".format(name)), instance_seg)
    semantic_seg[semantic_seg == labels["unsure"]] = ignore_label[labels["unsure"]]
    border_core = instance2border_core(instance_seg, border_thickness=border_thickness)  # , border_label=border_label, semantic_seg=semantic_seg)
    save_nifti(join(save_dir, "border_core", "{}.nii.gz".format(name)), border_core)
    return name, instances


def instance2border_core(instance_segmentation: np.ndarray,
                                            border_thickness: float = 2, center_label: int = 1, border_label: int = 2, semantic_seg: np.ndarray = None) -> np.ndarray:
    assert np.issubdtype(instance_segmentation.dtype,
                         np.unsignedinteger), 'instance_segmentation must be an array of type unsigned ' \
                                              'integer (can be uint8, uint16 etc)'
    border_semantic = np.zeros_like(instance_segmentation, dtype=np.uint8)
    selem = disk(border_thickness)
    pad_amount = 1
    instance_properties = regionprops(instance_segmentation)
    for ip in instance_properties:
        bbox = regionprops_bbox_to_proper_bbox(ip['bbox'])
        if pad_amount != 0:
            bbox = pad_bbox(bbox, pad_amount, instance_segmentation.shape)
        slicer = bounding_box_to_slice(bbox)
        instance_cropped = instance_segmentation[slicer]
        instance_mask = instance_cropped == ip["label"]
        instance_mask_eroded = binary_erosion(instance_mask, selem)
        if semantic_seg is not None:
            single_coords = ip["coords"][0, ...]
            center_label = semantic_seg[single_coords[0], single_coords[1]]
        border_semantic[slicer][(~instance_mask_eroded) & instance_mask] = border_label
        border_semantic[slicer][instance_mask_eroded & instance_mask] = center_label
    return border_semantic


if __name__ == "__main__":
    polygons_filepath = "/home/k539i/Documents/datasets/original/hubmap-hacking-the-human-vasculature/polygons.jsonl"
    train_load_dir = "/home/k539i/Documents/datasets/original/hubmap-hacking-the-human-vasculature/train"
    save_dir = "/home/k539i/Documents/datasets/original/hubmap-hacking-the-human-vasculature/preprocessed/train"
    labels = {"background": 0, "blood_vessel": 1, "glomerulus": 2, "unsure": 3}
    ignore_label = {3: 4}
    border_label = 3
    zscore = None
     
    with open(join(save_dir, "zscore.json"), 'r') as f:
        zscore = json.load(f)

    generate_dataset(polygons_filepath, train_load_dir, save_dir, labels, ignore_label, border_label, zscore=zscore)
