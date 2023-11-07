# Various custom torchvision datasets.

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
from PIL import Image, PngImagePlugin
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from typing import Callable, Optional
import json

PngImagePlugin.MAX_TEXT_CHUNK = 1048576 * 100


class ImageNetKaggle(VisionDataset):
    """
    Dataset class for the ImageNet / ILSVRC2012 dataset.
    Code taken from Paul Gavrikov's tutorial:
    https://towardsdatascience.com/downloading-and-using-the-imagenet-dataset-with-pytorch-f0908437c4be
    """

    def __init__(self, root: str, split: str = 'train', resize=None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        """
        N.B.: split may be either 'train' or 'val'.
        """
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}

        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)

        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
            self.val_to_syn = json.load(f)

        samples_dir = os.path.join(root, "ILSVRC\\Data\\CLS-LOC", split)
        if resize is not None and os.path.isdir(os.path.join(
                root, "ILSVRC\\Data\\CLS-LOC\\resized_{}x".format(resize),
                split)):
            true_samples_dir = os.path.join(
                root, "ILSVRC\\Data\\CLS-LOC\\resized_{}x".format(resize),
                split)
        else:
            true_samples_dir = samples_dir

        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(true_samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)

            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(true_samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]


class FER2013(VisionDataset):
    """
    Class for the FER-2013 dataset.
    Based on PyTorch MNIST dataset:
    https://pytorch.org/vision/stable/_modules/torchvision/datasets/mnist.html
    The below Kaggle notebook by Balmukund was helpful when adapting the
    original PyTorch MNIST dataset code:
    https://www.kaggle.com/balmukund/fer-2013-pytorch-implementation

    Args:
        root (str) - root directory for source data.
        split (str) - split of the data to be used: Training, PrivateTest,
            or PublicTest (default Training)
        transform (callable, optional) - transform for image data.
        target_transform (callable, optional) - transform for target data.
    """
    def __init__(self, root: str, split: str = "Training",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        """
        Initializer for the FER2013 class.
        """
        # Add standard transforms for this dataset.
        transform_list = [transform, transforms.Resize((48, 48))]
        transform = transforms.Compose(transform_list)

        super(FER2013, self).__init__(
            root, transform=transform, target_transform=target_transform)
        data = pd.read_csv(os.path.join(root, "fer2013.csv"))
        data = data[data["Usage"] == split]
        self.data = [[int(y) for y in x.split()] for x in data["pixels"]]
        self.targets = [t for t in data["emotion"]]

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Generates one sample of data.

        Args:
            index (int) - index of the data sample.
        """
        # Get image and target class.
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(np.array(img), mode='L')
        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
