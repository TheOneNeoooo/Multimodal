import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


##########################################################################
## 论文创新：掩码引导的语义区域损失
class MaskedSemanticLoss(nn.Module):
    def __init__(self):
        super(MaskedSemanticLoss, self).__init__()

    # 🔴 核心统一：严格使用 YCbCr 的 Y 通道公式提取亮度
    def extract_Y(self, image):
        r = image[:, 0:1, :, :]
        g = image[:, 1:2, :, :]
        b = image[:, 2:3, :, :]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def forward(self, fused_img, ir_img, vis_img, target_mask, task_type=None):
        if target_mask.dim() == 3:
            target_mask = target_mask.unsqueeze(1)

        background_mask = 1 - target_mask

        # 全部使用统一的 Y 通道
        fused_Y = self.extract_Y(fused_img)
        ir_Y = self.extract_Y(ir_img)
        vis_Y = self.extract_Y(vis_img)

        # 理想亮度目标
        ideal_target = torch.max(ir_Y, vis_Y)

        if task_type == 'enhance_vehicle':
            # 强行提亮 0.5 (只作用于 Y 亮度通道，不伤颜色)
            ideal_target = torch.clamp(ideal_target + target_mask * 0.5, 0.0, 1.0)
        elif task_type == 'enhance_person':
            ideal_target = ir_Y
        else:
            ideal_target = torch.max(ir_Y, vis_Y)

        loss_target = torch.mean(target_mask * torch.abs(fused_Y - ideal_target))
        loss_background = torch.mean(background_mask * torch.abs(fused_Y - vis_Y))

        return loss_target + loss_background


class fusion_loss(nn.Module):
    def __init__(self):
        super(fusion_loss, self).__init__()
        self.loss_func_ssim = L_SSIM(window_size=48)
        self.loss_func_Grad = L_Grad_position()
        self.loss_func_Max = L_Intensity()
        self.loss_func_color = L_color()
        self.loss_func_semantic = MaskedSemanticLoss()

    # 🔴 接收 task_type 传给底层
    def forward(self, image_visible, image_infrared, image_fused,
                max_ratio=4, ssim_vis_ratio=1, ssim_ir_ratio=1,
                ssim_ratio=1, color_ratio=20, text_ratio=10, 
                mask=None, semantic_ratio=10.0, task_type=None):
                
        image_visible_gray = self.rgb2gray(image_visible)
        image_infrared_gray = self.rgb2gray(image_infrared)
        image_fused_gray = self.rgb2gray(image_fused)

        loss_max = max_ratio * self.loss_func_Max(image_visible, image_infrared, image_fused)
        loss_ssim = ssim_ratio * (ssim_vis_ratio * self.loss_func_ssim(image_visible, image_fused) + ssim_ir_ratio * self.loss_func_ssim(image_infrared_gray, image_fused_gray))
        loss_color = color_ratio * self.loss_func_color(image_visible, image_fused)
        loss_text = text_ratio * self.loss_func_Grad(image_visible_gray, image_infrared_gray, image_fused_gray)

        loss_semantic = 0
        if mask is not None:
            # 🔴 把 task_type 喂给语义 Loss
            loss_semantic = semantic_ratio * self.loss_func_semantic(image_fused, image_infrared, image_visible, mask, task_type)

        total_loss = loss_max + loss_ssim + loss_color + loss_text + loss_semantic
        return total_loss, loss_ssim, loss_max, loss_color, loss_text, loss_semantic

    def rgb2gray(self, image):
        b, c, h, w = image.size()
        if c == 1:
            return image
        image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image_gray = image_gray.unsqueeze(dim=1)
        return image_gray

class fusion_prompt_loss(nn.Module):
    def __init__(self):
        super(fusion_prompt_loss, self).__init__()
        self.fusion_loss = fusion_loss()

    def forward(self, image_A, image_B, image_fused, task, mask=None):
        """新增参数: mask - 语义分割掩码（可选）"""
        total_loss = 0
        total_ssim_loss = 0
        total_max_loss = 0
        total_color_loss = 0
        total_grad_loss = 0
        total_semantic_loss = 0

        num_tasks = len(task)

        for idx, task_type in enumerate(task):
            img_A = self._get_image(image_A, idx)
            img_B = self._get_image(image_B, idx)
            img_fused = self._get_image(image_fused, idx)

            # 获取当前样本的mask（如果有）
            img_mask = None
            if mask is not None:
                img_mask = self._get_image(mask, idx)

            # 使用语义mask时，优先使用mask引导的语义损失
            # 同时保留原有的参数配置
            if task_type == "low_light":
                loss, ssim_loss, max_loss, color_loss, grad_loss, semantic_loss = self.fusion_loss(img_A, img_B, img_fused,
                                                                                    max_ratio=8, ssim_ratio=1, text_ratio=10, mask=img_mask)
            elif task_type == "over_exposure":
                loss, ssim_loss, max_loss, color_loss, grad_loss, semantic_loss = self.fusion_loss(img_A, img_B, img_fused,
                                                                                    max_ratio=4, ssim_ratio=0, text_ratio=2, mask=img_mask)
            elif task_type == "ir_low_contrast":
                loss, ssim_loss, max_loss, color_loss, grad_loss, semantic_loss = self.fusion_loss(img_A, img_B, img_fused,
                                                                                    max_ratio=8, ssim_ratio=1, text_ratio=10, mask=img_mask)
            elif task_type == "ir_noise":
                loss, ssim_loss, max_loss, color_loss, grad_loss, semantic_loss = self.fusion_loss(img_A, img_B, img_fused,
                                                                                    max_ratio=6, ssim_ratio=1, text_ratio=10, mask=img_mask)
            if task_type in ["enhance_person", "enhance_vehicle", "enhance_background"]:
                # 🔴 必须将 task_type 作为参数传进去！
                loss, ssim_loss, max_loss, color_loss, grad_loss, semantic_loss = self.fusion_loss(
                    img_A, img_B, img_fused,
                    max_ratio=6, ssim_ratio=1, text_ratio=10, mask=img_mask, task_type=task_type
                )
            else:
                raise ValueError(f"Unknown task type: {task_type}")

            total_loss += loss
            total_ssim_loss += ssim_loss
            total_max_loss += max_loss
            total_color_loss += color_loss
            total_grad_loss += grad_loss
            total_semantic_loss += semantic_loss

        # Calculate the average for each loss component
        return total_loss / num_tasks, total_ssim_loss / num_tasks, total_max_loss / num_tasks, total_color_loss / num_tasks, total_grad_loss / num_tasks, total_semantic_loss / num_tasks

    @staticmethod
    def _get_image(images, index):
        return images[index].unsqueeze(0)

class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, image_visible, image_fused):
        ycbcr_visible = self.rgb_to_ycbcr(image_visible)
        ycbcr_fused = self.rgb_to_ycbcr(image_fused)

        cb_visible = ycbcr_visible[:, 1, :, :]
        cr_visible = ycbcr_visible[:, 2, :, :]
        cb_fused = ycbcr_fused[:, 1, :, :]
        cr_fused = ycbcr_fused[:, 2, :, :]

        loss_cb = F.l1_loss(cb_visible, cb_fused)
        loss_cr = F.l1_loss(cr_visible, cr_fused)

        loss_color = loss_cb + loss_cr
        return loss_color

    def rgb_to_ycbcr(self, image):
        r = image[:, 0, :, :]
        g = image[:, 1, :, :]
        b = image[:, 2, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b

        ycbcr_image = torch.stack((y, cb, cr), dim=1)
        return ycbcr_image

class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    # 🔴 核心统一：必须和上面的 extract_Y 一模一样！绝对不能再用 mean！
    def extract_Y(self, image):
        r = image[:, 0:1, :, :]
        g = image[:, 1:2, :, :]
        b = image[:, 2:3, :, :]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def forward(self, image_visible, image_infrared, image_fused):
        gray_visible = self.extract_Y(image_visible)
        gray_infrared = self.extract_Y(image_infrared)
        gray_fused = self.extract_Y(image_fused)

        mask = (gray_infrared > gray_visible).float()

        # 伪标签也变成了严谨的 Y 通道图
        fused_target_gray = mask * gray_infrared + (1 - mask) * gray_visible
        
        # 算亮度 L1 loss，彻底告别颜色冲突
        Loss_intensity = F.l1_loss(fused_target_gray, gray_fused)
        return Loss_intensity

# Use it only if you have a consistent modal preference
class L_Intensity_Consist(nn.Module):
    def __init__(self):
        super(L_Intensity_Consist, self).__init__()

    def forward(self, image_visible, image_infrared, image_fused, vis_compose, ir_compose, consist_mode="l1"):
        if consist_mode == "l2":
            Loss_intensity = (vis_compose * F.mse_loss(image_visible, image_fused) + ir_compose * F.mse_loss(image_infrared, image_fused))/2
        else:
            Loss_intensity = (vis_compose * F.l1_loss(image_visible, image_fused) + ir_compose * F.l1_loss(image_infrared, image_fused))/2
        return Loss_intensity

# use the L_Grad_position or L_Grad
class L_Grad_position(nn.Module):
    def __init__(self):
        super(L_Grad_position, self).__init__()
        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1],
                                                       [-2, 0, 2],
                                                       [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1],
                                                       [0, 0, 0],
                                                       [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.padding = (1, 1, 1, 1)

    def forward(self, image_A, image_B, image_fuse):
        gradient_A_x, gradient_A_y = self.gradient(image_A)
        gradient_B_x, gradient_B_y = self.gradient(image_B)
        gradient_fuse_x, gradient_fuse_y = self.gradient(image_fuse)
        loss = F.l1_loss(gradient_fuse_x, torch.max(gradient_A_x, gradient_B_x)) + F.l1_loss(gradient_fuse_y, torch.max(gradient_A_y, gradient_B_y))
        return loss

    def gradient(self, image):
        image = F.pad(image, self.padding, mode='replicate')
        gradient_x = F.conv2d(image, self.sobel_x, padding=0)
        gradient_y = F.conv2d(image, self.sobel_y, padding=0)
        return torch.abs(gradient_x), torch.abs(gradient_y)

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.padding = (1, 1, 1, 1)

    def forward(self, image_visible, image_infrared, image_fused):
        gray_visible = self.tensor_RGB2GRAY(image_visible)
        gray_infrared = self.tensor_RGB2GRAY(image_infrared)
        gray_fused = self.tensor_RGB2GRAY(image_fused)

        d1 = self.gradient(gray_visible)
        d2 = self.gradient(gray_infrared)
        df = self.gradient(gray_fused)
        edge_loss = F.l1_loss(torch.max(d1, d2), df)
        return edge_loss

    def gradient(self, image):
        image = F.pad(image, self.padding, mode='replicate')
        gradient_x = F.conv2d(image, self.sobel_x, padding=0)
        gradient_y = F.conv2d(image, self.sobel_y, padding=0)
        return torch.abs(gradient_x) + torch.abs(gradient_y)
    
    def tensor_RGB2GRAY(self, image):
        b,c,h,w = image.size()
        if c == 1:
            return image
        image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image_gray = image_gray.unsqueeze(dim=1)
        return image_gray


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=24, window=None, size_average=True, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    return 1 - ret


class L_SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(L_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        (_, channel_2, _, _) = img2.size()

        if channel != channel_2 and channel == 1:
            img1 = torch.concat([img1, img1, img1], dim=1)
            channel = 3

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window.cuda()
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window.cuda()
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

