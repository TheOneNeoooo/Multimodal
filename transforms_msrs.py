"""
支持语义mask的数据变换
"""
import random
import numpy as np
import torch
from torchvision.transforms import functional as F
from torchvision import transforms as T


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, vi_img, ir_img, mask):
        for t in self.transforms:
            vi_img, ir_img, mask = t(vi_img, ir_img, mask)
        return vi_img, ir_img, mask


class Resize:
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, vi_img, ir_img, mask):
        vi_img = F.resize(vi_img, self.size)
        ir_img = F.resize(ir_img, self.size)
        mask = F.resize(mask, self.size, interpolation=F.InterpolationMode.NEAREST)
        return vi_img, ir_img, mask


class Resize_16:
    """调整图像大小为16的倍数"""
    def __init__(self):
        pass

    def __call__(self, vi_img, ir_img, mask):
        width, height = vi_img.size
        new_width = (width // 16) * 16
        new_height = (height // 16) * 16
        size = (new_height, new_width)

        vi_img = F.resize(vi_img, size)
        ir_img = F.resize(ir_img, size)
        mask = F.resize(mask, size, interpolation=F.InterpolationMode.NEAREST)
        return vi_img, ir_img, mask


class RandomHorizontalFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, vi_img, ir_img, mask):
        if random.random() < self.flip_prob:
            vi_img = F.hflip(vi_img)
            ir_img = F.hflip(ir_img)
            mask = F.hflip(mask)
        return vi_img, ir_img, mask


class RandomVerticalFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, vi_img, ir_img, mask):
        if random.random() < self.flip_prob:
            vi_img = F.vflip(vi_img)
            ir_img = F.vflip(ir_img)
            mask = F.vflip(mask)
        return vi_img, ir_img, mask


class RandomCrop:
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, vi_img, ir_img, mask):
        w, h = vi_img.size
        th, tw = self.size

        if w < tw or h < th:
            # 计算缩放比例
            scale = max(tw / w, th / h) + 0.01
            new_w, new_h = int(w * scale), int(h * scale)
            
            # torchvision 的 F.resize 要求尺寸格式为 (height, width)
            size = (new_h, new_w) 
            
            vi_img = F.resize(vi_img, size)
            ir_img = F.resize(ir_img, size)
            mask = F.resize(mask, size, interpolation=F.InterpolationMode.NEAREST)
            
            # 【关键修复】：更新 w 和 h，以便下面正确计算随机裁剪的范围
            w, h = new_w, new_h 

        # 随机裁剪
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        vi_img = F.crop(vi_img, i, j, th, tw)
        ir_img = F.crop(ir_img, i, j, th, tw)
        mask = F.crop(mask, i, j, th, tw)
        return vi_img, ir_img, mask


class ToTensor:
    def __call__(self, vi_img, ir_img, mask):
        vi_img = F.to_tensor(vi_img)
        ir_img = F.to_tensor(ir_img)
        # mask转为tensor，添加通道维度 [H, W] -> [1, H, W]
        mask_np = np.array(mask)
        if mask_np.ndim == 2:
            mask_np = mask_np[np.newaxis, ...]
        mask = torch.from_numpy(mask_np).float()
        return vi_img, ir_img, mask


class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, vi_img, ir_img, mask):
        normalize = T.Normalize(mean=self.mean, std=self.std)
        vi_img = normalize(vi_img)
        # 红外图像如果是单通道，也需要进行归一化
        if ir_img.shape[0] == 1:
            ir_img = normalize(torch.cat([ir_img]*3, dim=0))
        return vi_img, ir_img, mask
