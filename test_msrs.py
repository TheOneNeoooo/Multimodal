"""
MSRS语义可控融合测试脚本
支持用户输入自定义文本描述来控制融合结果

使用方法:
    python test_msrs.py --data_root /root/workspace/MSRS/test \
                        --weights_path ./experiments/MSRS_train_20260311-134739/weights/best.pth \
                        --input_text "Enhance_the_car_and_vehicle_in_this_scene"
"""
import os
import argparse
import numpy as np
from PIL import Image
import cv2
import clip
import torch
from torchvision.transforms import functional as F
from tqdm import tqdm  # 引入进度条
from model.Text_IF_model import Text_IF as create_model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ==========================================
    # 1. 智能保存路径：按 Prompt 名称自动隔离结果
    # ==========================================
    # 把提示词里的空格替换成下划线，取前30个字符作为文件夹名
    prompt_slug = args.input_text.replace(" ", "_")[:30]
    save_path = os.path.join(args.save_path, prompt_slug)
    os.makedirs(save_path, exist_ok=True)
    print(f"Results will be saved to: {save_path}")

    # ==========================================
    # 2. 安全读取数据列表
    # ==========================================
    ir_root = os.path.join(args.data_root, "ir")
    vi_root = os.path.join(args.data_root, "vi")

    # 以可见光图像为基准进行遍历
    vi_path_list = sorted([f for f in os.listdir(vi_root) if f.endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Found {len(vi_path_list)} images for testing.")

    # ==========================================
    # 3. 加载模型
    # ==========================================
    model_clip, _ = clip.load("ViT-B/32", device=device)
    model = create_model(model_clip).to(device)

    if os.path.exists(args.weights_path):
        # 兼容性处理：防止 GPU 显存峰值，先加载到 CPU
        checkpoint = torch.load(args.weights_path, map_location='cpu')
        # 获取模型权重（兼容是否嵌套了 'model' 字典）
        state_dict = checkpoint.get('model', checkpoint)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded weights from {args.weights_path}")
    else:
        raise FileNotFoundError(f"Weights file not found: {args.weights_path}")
    
    model.eval()

    # ==========================================
    # 4. 文本提示处理 (移出循环，性能飙升！)
    # ==========================================
    text_prompt = args.input_text
    print(f"User Text Prompt: '{text_prompt}'")
    # 只需执行一次 Tokenize
    text_tokens = clip.tokenize([text_prompt]).to(device)

    # ==========================================
    # 5. 推理循环
    # ==========================================
    with torch.no_grad():
        # 使用 tqdm 包装循环，显示华丽的进度条
        for vi_name in tqdm(vi_path_list, desc="Fusing Images"):
            ir_name = vi_name # 假设红外和可见光同名
            ir_img_path = os.path.join(ir_root, ir_name)
            vi_img_path = os.path.join(vi_root, vi_name)

            # 安全校验：确保对应的红外图存在
            if not os.path.exists(ir_img_path):
                print(f"Warning: IR image {ir_name} not found, skipping...")
                continue

            # 读取图像
            ir_img = Image.open(ir_img_path).convert('RGB')
            vi_img = Image.open(vi_img_path).convert('RGB')

            # 调整大小为 16 的倍数 (使用 F.resize 更贴合 PyTorch 生态)
            w, h = vi_img.size
            new_w = (w // 16) * 16
            new_h = (h // 16) * 16
            
            # 使用 BICUBIC 插值保证图像质量
            vi_img = F.resize(vi_img, (new_h, new_w), interpolation=F.InterpolationMode.BICUBIC)
            ir_img = F.resize(ir_img, (new_h, new_w), interpolation=F.InterpolationMode.BICUBIC)

            # 转张量并升维
            ir_tensor = F.to_tensor(ir_img).unsqueeze(0).to(device)
            vi_tensor = F.to_tensor(vi_img).unsqueeze(0).to(device)

            # 前向融合 (使用已经算好的 text_tokens)
            fused = model(vi_tensor, ir_tensor, text_tokens)

            # 后处理与保存
            fused_np = fused.squeeze(0).cpu().numpy()
            fused_np = np.transpose(fused_np, (1, 2, 0)) # CHW -> HWC
            fused_np = np.clip(fused_np, 0, 1)

            # 转换为 OpenCV 格式 (BGR)
            fused_np = (fused_np * 255).astype(np.uint8)
            save_name = os.path.join(save_path, vi_name)
            cv2.imwrite(save_name, fused_np[:, :, ::-1])

    print(f"\n🎉 Done! All fused results are saved to: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='MSRS test data root path')
    parser.add_argument('--weights_path', type=str, required=True, help='Path to model weights')
    parser.add_argument('--save_path', type=str, default='./results', help='Base output save path')
    parser.add_argument('--input_text', type=str, required=True,
                        help='Text prompt for semantic control, e.g., "Enhance the person in this scene"')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    args = parser.parse_args()
    
    main(args)