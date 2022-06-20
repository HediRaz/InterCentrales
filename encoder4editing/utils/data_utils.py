# Code adapted from pix2pixHD:
# https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""Utilities for making dataset."""

import os

IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM',
        '.bmp', '.BMP', '.tiff'
        ]


def is_image_file(filename):
    """Check if a file is an image."""
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(directory):
    """Make a dataset."""
    images = []
    assert os.path.isdir(directory), f'{directory} is not a valid directory'
    for root, _, fnames in sorted(os.walk(directory)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images
