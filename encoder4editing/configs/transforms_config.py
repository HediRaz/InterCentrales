"""Transformation configurations."""

from abc import abstractmethod

from torchvision import transforms


class TransformsConfig():
    """Base class for transformation configurations."""

    def __init__(self, opts):
        self.opts = opts

    @abstractmethod
    def get_transforms(self):
        """Return the transforms."""


class EncodeTransforms(TransformsConfig):
    """Encode and normalize images."""

    def get_transforms(self):
        """Return the transforms."""
        transforms_dict = {
                'transform_gt_train': transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                        ]), 'transform_source': None,
                'transform_test': transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                        ]), 'transform_inference': transforms.Compose([
                                transforms.Resize((256, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5],
                                                     [0.5, 0.5, 0.5])
                                ])
                }
        return transforms_dict


class CarsEncodeTransforms(TransformsConfig):
    """Encode and normalize images from Cars dataset."""

    def get_transforms(self):
        """Return the transforms."""
        transforms_dict = {
                'transform_gt_train': transforms.Compose([
                        transforms.Resize((192, 256)),
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                        ]), 'transform_source': None,
                'transform_test': transforms.Compose([
                        transforms.Resize((192, 256)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                        ]), 'transform_inference': transforms.Compose([
                                transforms.Resize((192, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5],
                                                     [0.5, 0.5, 0.5])
                                ])
                }
        return transforms_dict
