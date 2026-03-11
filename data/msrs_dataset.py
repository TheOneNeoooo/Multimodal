"""
MSRS语义分割数据集加载器 - 支持详细文本描述
每对图像对应一个详细描述和对应的语义mask

语义类别（内部使用）:
- enhance_person: 类别1 (行人)
- enhance_vehicle: 类别2,3 (建筑/车辆)
- enhance_background: 类别0 (背景)
"""
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
import json


class MSRSDataset(Dataset):
    """MSRS语义可控数据集 - 支持详细文本描述"""

    # 语义任务定义
    TASK_CLASSES = {
        'enhance_person': 1,        # 行人：类别1
        'enhance_vehicle': [2, 3], # 建筑/车辆：类别2和3
        'enhance_background': 0,   # 背景：类别0
    }

    # 类别ID到名称的映射 (根据MSRS数据集palette)
    CLASS_ID_TO_NAME = {
        0: 'unlabelled',    # 背景/未标注
        1: 'car',           # 汽车
        2: 'person',        # 行人
        3: 'bike',          # 自行车
        4: 'curve',         # 曲线/道路边界
        5: 'car_stop',      # 停车标志
        6: 'guardrail',     # 护栏
        7: 'color_cone',    # 锥形路标
        8: 'bump'           # 减速带
    }

    # 用于融合任务的语义分组
    SEMANTIC_GROUPS = {
        'person': [2],           # 行人
        'vehicle': [1, 3, 4, 5, 6, 7, 8],  # 车辆及其他目标
        'background': [0]        # 背景
    }

    def __init__(self, data_root, phase='train', transform=None,
                 descriptions_file=None, use_semantic_prompt=True):
        """
        Args:
            data_root: 数据根目录，如 '/root/workspace/MSRS'
            phase: 'train' 或 'test'
            transform: 数据增强
            descriptions_file: 文本描述JSON文件路径（可选）
            use_semantic_prompt: 是否使用语义prompt（从mask自动生成）
        """
        self.data_root = data_root
        self.phase = phase
        self.transform = transform
        self.descriptions_file = descriptions_file
        self.use_semantic_prompt = use_semantic_prompt

        # 设置路径
        self.ir_root = os.path.join(data_root, phase, 'ir')
        self.vi_root = os.path.join(data_root, phase, 'vi')
        self.mask_root = os.path.join(data_root, phase, 'Segmentation_labels')

        # 获取图像列表
        self.image_names = sorted([f for f in os.listdir(self.ir_root)
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])

        # 加载详细描述（如果提供）
        self.descriptions = self._load_descriptions()

        print(f"MSRS Dataset loaded: {len(self.image_names)} {phase} images")
        print(f"Use semantic prompt: {use_semantic_prompt}")

    def _load_descriptions(self):
        """加载详细的文本描述"""
        if self.descriptions_file and os.path.exists(self.descriptions_file):
            with open(self.descriptions_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _create_semantic_prompt(self, mask):
        """
        根据mask自动生成语义prompt
        例如: "This image contains a person and a vehicle. Enhance the person in the infrared image while preserving the vehicle details."
        """
        unique_classes = np.unique(mask)
        class_names = []

        for cls_id in unique_classes:
            if cls_id in self.CLASS_ID_TO_NAME and cls_id != 0:  # 忽略背景
                class_names.append(self.CLASS_ID_TO_NAME[cls_id])

        if not class_names:
            return "This is an infrared-visible image fusion task."

        # 构建语义prompt
        n = len(class_names)
        if n == 1:
            return f"Enhance the {class_names[0]} in this scene for better detection."
        elif n == 2:
            return f"Enhance both the {class_names[0]} and {class_names[1]} in this infrared image."
        else:
            # 3个或更多类别: "Enhance the person, vehicle, and building in this scene."
            # 使用更清晰的语法
            last = class_names[-1]
            others = class_names[:-1]
            return f"Enhance the {', '.join(others)}, and {last} in this scene."

    
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        base_name = name.replace('.png', '').replace('.jpg', '')

        ir_path = os.path.join(self.ir_root, name)
        vi_path = os.path.join(self.vi_root, name)
        mask_path = os.path.join(self.mask_root, name)

        image_ir = Image.open(ir_path).convert('RGB')
        image_vi = Image.open(vi_path).convert('RGB')

        mask_pil = Image.fromarray(np.array(Image.open(mask_path)).astype(np.uint8))

        if self.transform is not None:
            image_vi, image_ir, mask_tensor = self.transform(image_vi, image_ir, mask_pil)
        else:
            from torchvision.transforms import functional as F
            image_vi = F.to_tensor(image_vi)
            image_ir = F.to_tensor(image_ir)
            mask_tensor = torch.from_numpy(np.array(mask_pil)[np.newaxis, ...]).float()

        mask_array = mask_tensor.squeeze().numpy() 
        unique_classes = np.unique(mask_array)

        person_mask = np.isin(mask_array, [2]).astype(np.float32)
        vehicle_mask = np.isin(mask_array, [1, 3, 4, 5, 6, 7, 8]).astype(np.float32)
        group_masks = {
            'person': torch.from_numpy(person_mask).float().unsqueeze(0),
            'vehicle': torch.from_numpy(vehicle_mask).float().unsqueeze(0)
        }
        
        available_targets = []
        if 2 in unique_classes:
            available_targets.append('person')
        if any(c in unique_classes for c in [1, 3, 4, 5, 6, 7, 8]):
            available_targets.append('vehicle')

        # ==========================================
        # 🔴 终极纯净版抽卡：只有车，或者只有人，绝无 'all'！
        # ==========================================
        if len(available_targets) > 0:
            weights = []
            
            for opt in available_targets:
                if opt == 'vehicle':
                    weights.append(5.0) # 🚗 车和装置：5倍权重！
                elif opt == 'person':
                    weights.append(1.0) # 🚶 行人：1倍权重

            choice = random.choices(available_targets, weights=weights, k=1)[0]
            
            if choice == 'person':
                target_mask = np.isin(mask_array, [2]).astype(np.float32)
                description = "Enhance the person in this scene"
            elif choice == 'vehicle':
                target_mask = np.isin(mask_array, [1, 3, 4, 5, 6, 7, 8]).astype(np.float32)
                description = "Enhance the vehicle and building in this scene"
                
        else:
            # 纯背景图：啥也没有，老老实实当背景
            target_mask = np.zeros_like(mask_array, dtype=np.float32)
            description = "This is an infrared-visible image fusion task."

        target_mask_tensor = torch.from_numpy(target_mask).float().unsqueeze(0)

        return (image_vi, image_ir, target_mask_tensor, group_masks, description, base_name)

        # 完结撒花！6个变量整整齐齐
        return (image_vi, image_ir, target_mask_tensor, group_masks, description, base_name)
    @staticmethod
    def collate_fn(batch):
        """
        自定义batch整理函数
        返回: vi, ir, binary_masks, group_masks_list, descriptions, names
        """
        images_vi, images_ir, binary_masks, group_masks_list, descriptions, names = zip(*batch)

        images_vi = torch.stack(images_vi, dim=0)
        images_ir = torch.stack(images_ir, dim=0)
        binary_masks = torch.stack(binary_masks, dim=0)

        return (images_vi, images_ir, binary_masks, group_masks_list, descriptions, names)


class MSRSDatasetSimple(Dataset):
    """简化版MSRS数据集 - 只支持三种固定语义任务"""

    TASK_LIST = ['enhance_person', 'enhance_vehicle', 'enhance_background']

    TASK_CLASSES = {
        'enhance_person': [1],
        'enhance_vehicle': [2, 3],
        'enhance_background': [0],
    }

    TASK_PROMPTS = {
        'enhance_person': "Highlight pedestrians in the scene for better detection.",
        'enhance_vehicle': "Enhance vehicles and buildings in this infrared image.",
        'enhance_background': "Preserve background texture details in the fusion result.",
    }

    def __init__(self, data_root, phase='train', transform=None, task='enhance_person'):
        self.data_root = data_root
        self.phase = phase
        self.transform = transform
        self.task = task

        self.ir_root = os.path.join(data_root, phase, 'ir')
        self.vi_root = os.path.join(data_root, phase, 'vi')
        self.mask_root = os.path.join(data_root, phase, 'Segmentation_labels')

        self.image_names = sorted([f for f in os.listdir(self.ir_root)
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])

        print(f"MSRS Simple Dataset: {len(self.image_names)} {phase} images, task: {task}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]

        ir_path = os.path.join(self.ir_root, name)
        vi_path = os.path.join(self.vi_root, name)
        mask_path = os.path.join(self.mask_root, name)

        image_ir = Image.open(ir_path).convert('RGB')
        image_vi = Image.open(vi_path).convert('RGB')

        mask = np.array(Image.open(mask_path))
        target_classes = self.TASK_CLASSES[self.task]
        binary_mask = np.isin(mask, target_classes).astype(np.float32)
        mask = torch.from_numpy(binary_mask)

        if self.transform is not None:
            image_vi, image_ir, mask = self.transform(image_vi, image_ir, mask)

        return image_vi, image_ir, image_vi, image_ir, image_vi, mask, self.task, name.replace('.png', '')

    @staticmethod
    def collate_fn(batch):
        images_vi, images_ir, images_vi_gt, images_ir_gt, images_full, masks, tasks, names = zip(*batch)

        images_vi = torch.stack(images_vi, dim=0)
        images_ir = torch.stack(images_ir, dim=0)
        images_vi_gt = torch.stack(images_vi_gt, dim=0)
        images_ir_gt = torch.stack(images_ir_gt, dim=0)
        images_full = torch.stack(images_full, dim=0)
        masks = torch.stack(masks, dim=0)

        return images_vi, images_ir, images_vi_gt, images_ir_gt, images_full, masks, tasks, names


class MSRSTripletDataset(Dataset):
    """MSRS多任务数据集，一次训练三个任务"""

    TASK_LIST = ['enhance_person', 'enhance_vehicle', 'enhance_background']

    def __init__(self, data_root, phase='train', transform=None):
        self.data_root = data_root
        self.phase = phase
        self.transform = transform

        self.datasets = {task: MSRSDatasetSimple(data_root, phase, transform, task)
                        for task in self.TASK_LIST}

        self.total_length = sum(len(d) for d in self.datasets.values())

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        task = self.TASK_LIST[idx % len(self.TASK_LIST)]
        dataset = self.datasets[task]
        sample_idx = idx // len(self.TASK_LIST) % len(dataset)
        return dataset[sample_idx]

    @staticmethod
    def collate_fn(batch):
        return MSRSDatasetSimple.collate_fn(batch)
