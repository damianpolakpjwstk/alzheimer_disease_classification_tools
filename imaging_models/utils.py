"""Helper functions and classes for training, evaluating, and explaining CNN models."""
import numpy as np
import torch

from data_analysis_preparation.utils import get_bounding_box


def crop_to_bounding_box(image_data):
    xmin, xmax, ymin, ymax, zmin, zmax, width, height, depth, min_val, max_val = get_bounding_box(image_data)
    cropped_image_data = image_data[xmin:xmax, ymin:ymax, zmin:zmax]
    return cropped_image_data


def to_shape(a, shape):
    y_, x_, z_ = shape
    y, x, z = a.shape[-3], a.shape[-2], a.shape[-1]
    y_pad = (y_ - y)
    x_pad = (x_ - x)
    z_pad = (z_ - z)
    return np.pad(a, (
    (y_pad // 2, y_pad // 2 + y_pad % 2), (x_pad // 2, x_pad // 2 + x_pad % 2), (z_pad // 2, z_pad // 2 + z_pad % 2)),
                  'constant')
