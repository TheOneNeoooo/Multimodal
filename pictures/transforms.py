# Text-IF/transforms.py
# 该文件定义了一系列图像变换操作，用于数据增强和预处理。
# 这些变换操作包括：
# - pad_if_smaller: 如果图像尺寸小于指定尺寸，则进行填充。
# - Compose: 组合多个变换操作。
# - Resize: 调整图像大小。
# - Resize_16: 调整图像大小为16的倍数。
# - RandomHorizontalFlip: 随机水平翻转图像。
# - RandomVerticalFlip: 随机垂直翻转图像。
# - RandomCrop: 随机裁剪图像。
# - CenterCrop: 中心裁剪图像。
# - ToTensor: 将图像转换为 Tensor。

import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    """
    如果图像的最小尺寸小于指定尺寸，则进行填充。

    Args:
        img (PIL.Image): 待填充的图像。
        size (int): 指定的最小尺寸。
        fill (int): 填充值，默认为0。

    Returns:
        PIL.Image: 填充后的图像。
    """
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    """
    组合多个变换操作。

    Args:
        transforms (list): 包含多个变换操作的列表。
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, image_gt, target_gt, image_full):
        """
        对图像和目标应用一系列变换。

        Args:
            image (PIL.Image): 待变换的图像。
            target (PIL.Image): 待变换的目标。
            image_gt (PIL.Image): 待变换的图像 ground truth。
            target_gt (PIL.Image): 待变换的目标 ground truth。
            image_full (PIL.Image): 待变换的完整图像。

        Returns:
            tuple: 变换后的图像、目标、图像 ground truth、目标 ground truth 和完整图像。
        """
        for t in self.transforms:
            image, target, image_gt, target_gt, image_full = t(image, target, image_gt, target_gt, image_full)
        return image, target, image_gt, target_gt, image_full


class Resize(object):
    """
    调整图像大小。

    Args:
        size (int or tuple): 指定的大小。如果为 int，则表示将图像的较小边调整为该大小，并保持纵横比。
                             如果为 tuple，则表示将图像调整为该大小。
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, image_gt, target_gt):
        """
        调整图像和目标的大小。

        Args:
            image (PIL.Image): 待调整大小的图像。
            target (PIL.Image): 待调整大小的目标。
            image_gt (PIL.Image): 待调整大小的图像 ground truth。
            target_gt (PIL.Image): 待调整大小的目标 ground truth。

        Returns:
            tuple: 调整大小后的图像、目标、图像 ground truth 和目标 ground truth。
        """
        image = F.resize(image, self.size)
        target = F.resize(target, self.size, interpolation=T.InterpolationMode.NEAREST)
        image_gt = F.resize(image_gt, self.size, interpolation=T.InterpolationMode.NEAREST)
        target_gt = F.resize(target_gt, self.size, interpolation=T.InterpolationMode.NEAREST)

        return image, target, image_gt, target_gt


class Resize_16(object):
    """
    调整图像大小为16的倍数。
    """
    def __init__(self):
        pass

    def __call__(self, image, target, image_gt, target_gt, image_full):
        """
        调整图像和目标的大小为16的倍数。

        Args:
            image (PIL.Image): 待调整大小的图像。
            target (PIL.Image): 待调整大小的目标。
            image_gt (PIL.Image): 待调整大小的图像 ground truth。
            target_gt (PIL.Image): 待调整大小的目标 ground truth。
            image_full (PIL.Image): 待调整大小的完整图像。

        Returns:
            tuple: 调整大小后的图像、目标、图像 ground truth、目标 ground truth 和完整图像。
        """
        width, height = image.size

        new_width = (width // 16) * 16
        new_height = (height // 16) * 16

        image = F.resize(image, (new_height, new_width))
        target = F.resize(target, (new_height, new_width), interpolation=T.InterpolationMode.NEAREST)
        image_gt = F.resize(image_gt, (new_height, new_width), interpolation=T.InterpolationMode.NEAREST)
        target_gt = F.resize(target_gt, (new_height, new_width), interpolation=T.InterpolationMode.NEAREST)
        image_full = F.resize(image_full, (new_height, new_width), interpolation=T.InterpolationMode.NEAREST)

        return image, target, image_gt, target_gt, image_full


class RandomHorizontalFlip(object):
    """
    随机水平翻转图像。

    Args:
        flip_prob (float): 翻转的概率。
    """
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target, image_gt, target_gt, image_full):
        """
        随机水平翻转图像和目标。

        Args:
            image (PIL.Image): 待翻转的图像。
            target (PIL.Image): 待翻转的目标。
            image_gt (PIL.Image): 待翻转的图像 ground truth。
            target_gt (PIL.Image): 待翻转的目标 ground truth。
            image_full (PIL.Image): 待翻转的完整图像。

        Returns:
            tuple: 翻转后的图像、目标、图像 ground truth、目标 ground truth 和完整图像。
        """
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
            image_gt = F.hflip(image_gt)
            target_gt = F.hflip(target_gt)
            image_full = F.hflip(image_full)
        return image, target, image_gt, target_gt, image_full


class RandomVerticalFlip(object):
    """
    随机垂直翻转图像。

    Args:
        flip_prob (float): 翻转的概率。
    """
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target, image_gt, target_gt, image_full):
        """
        随机垂直翻转图像和目标。

        Args:
            image (PIL.Image): 待翻转的图像。
            target (PIL.Image): 待翻转的目标。
            image_gt (PIL.Image): 待翻转的图像 ground truth。
            target_gt (PIL.Image): 待翻转的目标 ground truth。
            image_full (PIL.Image): 待翻转的完整图像。

        Returns:
            tuple: 翻转后的图像、目标、图像 ground truth、目标 ground truth 和完整图像。
        """
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
            image_gt = F.vflip(image_gt)
            target_gt = F.vflip(target_gt)
            image_full = F.vflip(image_full)
        return image, target, image_gt, target_gt, image_full


class RandomCrop(object):
    """
    随机裁剪图像。

    Args:
        size (int): 裁剪的大小。
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, image_gt, target_gt, image_full):
        """
        随机裁剪图像和目标。

        Args:
            image (PIL.Image): 待裁剪的图像。
            target (PIL.Image): 待裁剪的目标。
            image_gt (PIL.Image): 待裁剪的图像 ground truth。
            target_gt (PIL.Image): 待裁剪的目标 ground truth。
            image_full (PIL.Image): 待裁剪的完整图像。

        Returns:
            tuple: 裁剪后的图像、目标、图像 ground truth、目标 ground truth 和完整图像。
        """
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size)
        image_gt = pad_if_smaller(image_gt, self.size)
        target_gt = pad_if_smaller(target_gt, self.size)
        image_full = pad_if_smaller(image_full, self.size)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        image_gt = F.crop(image_gt, *crop_params)
        target_gt = F.crop(target_gt, *crop_params)
        image_full = F.crop(image_full, *crop_params)
        return image, target, image_gt, target_gt, image_full

class CenterCrop(object):
    """
    中心裁剪图像。

    Args:
        size (int): 裁剪的大小。
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, image_gt, target_gt):
        """
        中心裁剪图像和目标。

        Args:
            image (PIL.Image): 待裁剪的图像。
            target (PIL.Image): 待裁剪的目标。
            image_gt (PIL.Image): 待裁剪的图像 ground truth。
            target_gt (PIL.Image): 待裁剪的目标 ground truth。

        Returns:
            tuple: 裁剪后的图像、目标、图像 ground truth 和目标 ground truth。
        """
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        image_gt = F.center_crop(image_gt, self.size)
        target_gt = F.center_crop(target_gt, self.size)
        return image, target, image_gt, target_gt


class ToTensor(object):
    """
    将图像转换为 Tensor。
    """
    def __call__(self, image, target, image_gt, target_gt, image_full):
        """
        将图像和目标转换为 Tensor。

        Args:
            image (PIL.Image): 待转换的图像。
            target (PIL.Image): 待转换的目标。
            image_gt (PIL.Image): 待转换的图像 ground truth。
            target_gt (PIL.Image): 待转换的目标 ground truth。
            image_full (PIL.Image): 待转换的完整图像。

        Returns:
            tuple: 转换后的图像、目标、图像 ground truth、目标 ground truth 和完整图像。
        """
        image = F.to_tensor(image)
        target = F.to_tensor(target)
        image_gt = F.to_tensor(image_gt)
        target_gt = F.to_tensor(target_gt)
        image_full = F.to_tensor(image_full)
        return image, target, image_gt, target_gt, image_full