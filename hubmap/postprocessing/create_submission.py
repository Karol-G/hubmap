import numpy as np
from hubmap.utils.utils import load_filepaths
from os.path import join
import json
import tifffile
from tqdm import tqdm
import base64
from pycocotools import _mask as coco_mask
import typing as t
import zlib
import pandas as pd


def create_submission(load_dir, instance_classes_filepath, save_filepath, confidence=1.0):
    df = {"id": [], "height": [], "width": [], "prediction_string": []}
    names = load_filepaths(load_dir, return_path=False, return_extension=False)

    with open(instance_classes_filepath, 'r') as f:
        instance_classes = json.load(f)

    for name in tqdm(names):
        instance_seg = tifffile.imread(join(load_dir, "{}.tif".format(name)))
        instances = instance_classes[name]
        prediction_string = ""

        for instance, label in instances.items():
            if label == 1:  # Only blood vessels are the target class,glomerulus are not relevant
                instance = int(instance)
                label = int(label)
                mask = instance_seg == instance
                base64_str = encode_binary_mask(mask)
                prediction_string += " {} {} {}".format(0, confidence, base64_str)
        
        prediction_string = prediction_string[1:]
        df["id"].append(name)
        df["height"].append(instance_seg.shape[0])
        df["width"].append(instance_seg.shape[1])
        df["prediction_string"].append(prediction_string)

    df = pd.DataFrame.from_dict(df)
    df.to_csv(save_filepath, index=False)


def encode_binary_mask(mask: np.ndarray) -> t.Text:
    """Converts a binary mask into OID challenge encoding ascii text."""

    # check input mask --
    if mask.dtype != bool:
        raise ValueError(
            "encode_binary_mask expects a binary mask, received dtype == %s" %
            mask.dtype)

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(
            "encode_binary_mask expects a 2d mask, received shape == %s" %
            mask.shape)

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    base64_str = base64_str.decode('utf-8')
    return base64_str


if __name__ == "__main__":
    load_dir = "/home/k539i/Documents/datasets/original/hubmap-hacking-the-human-vasculature/predictions/instance_seg/instance_seg/"
    instance_classes_filepath = "/home/k539i/Documents/datasets/original/hubmap-hacking-the-human-vasculature/predictions/instance_seg/instance_classes.json"
    save_filepath = "/home/k539i/Documents/datasets/original/hubmap-hacking-the-human-vasculature/predictions/instance_seg/submission.csv"

    create_submission(load_dir, instance_classes_filepath, save_filepath)
