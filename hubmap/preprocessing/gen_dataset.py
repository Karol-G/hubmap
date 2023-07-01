import json
import numpy as np
import tifffile
from os.path import join
from PIL import Image
from pathlib import Path
import shutil
from tqdm import tqdm
from hubmap.utils.utils import load_filepaths
import cv2
from hubmap.conversion.border_core_conversion import convert_instanceseg_to_semantic_patched as instance2border_core
from skimage.transform import resize


def generate_dataset(polygons_filepath, train_load_dir, save_dir, labels, ignore_label, border_label, include_unannotated=False, border_thickness=3, resize_factor=2):
    shutil.rmtree(save_dir, ignore_errors=True)
    Path(join(save_dir, "images")).mkdir(parents=True, exist_ok=True)
    Path(join(save_dir, "semantic_seg")).mkdir(parents=True, exist_ok=True)
    Path(join(save_dir, "instance_seg")).mkdir(parents=True, exist_ok=True)
    Path(join(save_dir, "border_core")).mkdir(parents=True, exist_ok=True)

    # Process images with annotations
    polygons = {}
    with open(polygons_filepath, 'r') as f:
         polygons_list = list(f)

    for json_str in tqdm(polygons_list):
        image_polygons = json.loads(json_str)
        name = image_polygons["id"]
        image = tifffile.imread(join(train_load_dir, "{}.tif".format(name)))

        if resize_factor is None:
          shutil.copy(join(train_load_dir, "{}.tif".format(name)), join(save_dir, "images", "{}.tif".format(name)))
        else:
            image = resize(image.astype(np.float64), (image.shape[0]*resize_factor, image.shape[1]*resize_factor, 3), order=3, anti_aliasing=False, preserve_range=True)
            image = np.rint(image).astype(np.uint8)
            image_pil = Image.fromarray(image)
            image_pil.save(join(save_dir, "images", "{}.tif".format(name)))
        
        semantic_seg = np.zeros(image.shape[:2], dtype=np.uint8)
        instance_seg = np.zeros(image.shape[:2], dtype=np.uint8)
        instances = {}
        for instance_label, class_annotation in enumerate(image_polygons["annotations"]):
             instance_label += 1
             new_class_annotation = {}
             new_class_annotation["label"] = labels[class_annotation["type"]]
             coordinates = np.asarray(class_annotation["coordinates"]).squeeze()
             contour = coordinates.reshape((-1, 1, 2))
             cv2.fillPoly(semantic_seg, [contour], color=new_class_annotation["label"])
             cv2.fillPoly(instance_seg, [contour], color=instance_label)
             new_class_annotation["coordinates"] = coordinates.tolist()
             instances[instance_label] = new_class_annotation
        
        semantic_seg_pil = Image.fromarray(semantic_seg)
        semantic_seg_pil.save(join(save_dir, "semantic_seg", "{}.tif".format(name)))

        instance_seg_pil = Image.fromarray(instance_seg)
        instance_seg_pil.save(join(save_dir, "instance_seg", "{}.tif".format(name)))

        semantic_seg[semantic_seg == labels["unsure"]] = ignore_label[labels["unsure"]]
        border_core = instance2border_core(instance_seg, border_thickness=border_thickness, border_label=border_label, semantic_seg=semantic_seg)
        border_core_pil = Image.fromarray(border_core)
        border_core_pil.save(join(save_dir, "border_core", "{}.tif".format(name)))
        
        polygons[name] = instances

    if include_unannotated:
        # Process images without annotations
        names = load_filepaths(train_load_dir, return_path=False, return_extension=False)

        for name in tqdm(names):
            if name not in polygons:
               image = tifffile.imread(join(train_load_dir, "{}.tif".format(name)))

               if resize_factor is None:
                    shutil.copy(join(train_load_dir, "{}.tif".format(name)), join(save_dir, "images", "{}.tif".format(name)))
               else:
                    image = resize(image.astype(np.float64), (image.shape[0]*resize_factor, image.shape[1]*resize_factor, 3), order=3, anti_aliasing=False, preserve_range=True)
                    image = np.rint(image).astype(np.uint8)
                    image_pil = Image.fromarray(image)
                    image_pil.save(join(save_dir, "images", "{}.tif".format(name)))

               seg = np.full(image.shape[:2], fill_value=labels["unsure"], dtype=np.uint8)
               seg = Image.fromarray(seg)
               seg.save(join(save_dir, "semantic_seg", "{}.tif".format(name)))
               seg.save(join(save_dir, "instance_seg", "{}.tif".format(name)))
               seg.save(join(save_dir, "border_core", "{}.tif".format(name)))
               polygons[name] = {1: {"label": labels["unsure"], "coordinates": None}}

    with open(join(save_dir, 'metadata.json'), 'w') as json_file:
        json.dump(polygons, json_file)


if __name__ == "__main__":
     polygons_filepath = "/home/k539i/Documents/datasets/original/hubmap-hacking-the-human-vasculature/polygons.jsonl"
     train_load_dir = "/home/k539i/Documents/datasets/original/hubmap-hacking-the-human-vasculature/train"
     save_dir = "/home/k539i/Documents/datasets/original/hubmap-hacking-the-human-vasculature/preprocessed/train"
     labels = {"background": 0, "blood_vessel": 1, "glomerulus": 2, "unsure": 3}
     ignore_label = {3: 4}
     border_label = 3

     generate_dataset(polygons_filepath, train_load_dir, save_dir, labels, ignore_label, border_label)
