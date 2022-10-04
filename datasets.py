# Various custom torchvision datasets.

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
from PIL import Image
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from typing import Callable, Optional
import json


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
