# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
from logging import getLogger

import cv2
from PIL import ImageFilter
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensor
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset

logger = getLogger()


class MultiCropDataset(Dataset):
    def __init__(
        self,
        data_path,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        return_index=False,
        pil_blur=False,
        split='train'
    ):
#         super(MultiCropDataset, self).__init__(data_path)
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

#         color_transform = [get_color_distortion(), RandomGaussianBlur()]
#         if pil_blur:
#             color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
#                 scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.trans = trans
        
        root_dir = data_path
        if split == 'train':
            self.root_dir = Path(root_dir) / 'CIFAR-10-C-1000/'
            corruptions = ['gaussian_noise', 'shot_noise', 'defocus_blur', 'glass_blur', 'zoom_blur', 'snow', 'frost', 'brightness', 'contrast', 'pixelate']
            other_idx = [0, 1, 2, 5, 6, 7] 
        if split == 'val':
            self.root_dir = Path(root_dir) / 'CIFAR-10-C-1000/'
            corruptions = ['speckle_noise', 'gaussian_blur', 'saturate']
            other_idx = [3, 9]
        if split == 'test':
            self.root_dir = Path(root_dir) / 'CIFAR-10-C/'
            corruptions = ['impulse_noise', 'motion_blur', 'fog', 'elastic_transform']
            other_idx = [4, 8]
        print("loading cifar-10-c", split)
        other = [load_corruption(self.root_dir / (corruption + '.npy')) for corruption in ['spatter', 'jpeg_compression']]
        other = np.concatenate(other, axis=0)[other_idx]
        data = [load_corruption(self.root_dir / (corruption + '.npy')) for corruption in corruptions]
        data = np.concatenate(data, axis=0)
        self._X = np.concatenate([other, data], axis=0)

        self.n_groups = self._X.shape[0]
        self.groups = list(range(self.n_groups))

        self.image_shape = (3, 32, 32)
        self._X = self._X.reshape((-1, 32, 32, 3))
        if split == 'test':
            self._y = np.load(self.root_dir / 'labels.npy')[:10000]
            self._y = np.tile(self._y, self.n_groups)       
            self.group_ids = np.array([[i]*10000 for i in range(self.n_groups)]).flatten()
        else:
            other_labels = [load_corruption(self.root_dir / (corruption + '_labels.npy')) for corruption in ['spatter', 'jpeg_compression']]
            other_labels = np.concatenate(other_labels, axis=0)[other_idx]
            data_labels = [load_corruption(self.root_dir / (corruption + '_labels.npy')) for corruption in corruptions]
            data_labels = np.concatenate(data_labels, axis=0)
            self._y = np.concatenate([other_labels, data_labels], axis=0).flatten()
            self.group_ids = np.array([[i]*1000 for i in range(self.n_groups)]).flatten()
        
        self._len = len(self.group_ids)
        print("loaded")
        
        self.group_counts, _ = np.histogram(self.group_ids,
                                            bins=range(self.n_groups + 1),
                                            density=False)
#         self.transform = get_transform()
        print("split: ", split)

    def __len__(self):
        return self._len
    
    def __getitem__(self, index):
#         x = self.transform(**{'image': self._X[index]})['image']
        x = self._X[index]
        image = Image.fromarray(x)
        y = torch.tensor(self._y[index], dtype=torch.long)
        g = torch.tensor(self.group_ids[index], dtype=torch.long)
#         path, _ = self.samples[index]
#         image = self.loader(path)
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops


class RandomGaussianBlur(object):
    def __call__(self, img):
        do_it = np.random.rand() > 0.5
        if not do_it:
            return img
        sigma = np.random.rand() * 1.9 + 0.1
        return cv2.GaussianBlur(np.asarray(img), (23, 23), sigma)


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

def get_transform():
    transform = albumentations.Compose([
        albumentations.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225], max_pixel_value=255, 
                                p=1.0, always_apply=True),
        ToTensor(),
    ])
    return transform

def load_corruption(file):
    data = np.load(file)
    return np.array(np.array_split(data, 5))