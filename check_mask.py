import os
import torch
import torchvision
from data.msrs_dataset import MSRSDataset
import transforms_msrs as T
from torch.utils.data import DataLoader

def main():
    print("开始拦截 Dataset，检查掩码输出...")
    
    # 建一个文件夹专门存检查的图
    save_dir = "./check_masks_output"
    os.makedirs(save_dir, exist_ok=True)

    # 1. 挂载和你训练一模一样的数据增强
    train_transform = T.Compose([
        T.RandomCrop(256),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.ToTensor(),
    ])

    # 2. 实例化你的数据集 (注意替换成你的真实 data_root)
    # 这里用 train 集测试，因为动态抽卡主要在 train 里
    dataset = MSRSDataset(data_root='/root/workspace/MSRS', phase='train', transform=train_transform)
    
    # 找几张包含车和人的图存下来
    found_vehicle = 0
    found_person = 0

    for i in range(len(dataset)):
        # 手动向数据集要一张图
        I_vi, I_ir, target_mask, group_masks, description, base_name = dataset[i]
        
        # 判断当前这张图被抽中了什么任务
        if "vehicle" in description and found_vehicle < 3:
            # target_mask 形状是 [1, 256, 256]，里面的值是 0 或 1
            # 把三个张量拼在一起保存，方便对比: [可见光, 红外, 掩码]
            # 掩码是单通道，我们把它复制成三通道以便和彩色图拼图
            mask_vis = target_mask.repeat(3, 1, 1) 
            
            # 拼接成一张长图
            grid = torchvision.utils.make_grid([I_vi, I_ir, mask_vis], nrow=3)
            save_path = os.path.join(save_dir, f"vehicle_task_{base_name}.png")
            torchvision.utils.save_image(grid, save_path)
            
            print(f"✅ 抓到一张【车辆任务】: {base_name}, 掩码亮起像素数: {target_mask.sum().item()}")
            found_vehicle += 1
            
        elif "person" in description and found_person < 3:
            mask_vis = target_mask.repeat(3, 1, 1) 
            grid = torchvision.utils.make_grid([I_vi, I_ir, mask_vis], nrow=3)
            save_path = os.path.join(save_dir, f"person_task_{base_name}.png")
            torchvision.utils.save_image(grid, save_path)
            
            print(f"✅ 抓到一张【行人任务】: {base_name}, 掩码亮起像素数: {target_mask.sum().item()}")
            found_person += 1

        # 只要各找齐 3 张，就停止循环
        if found_vehicle == 3 and found_person == 3:
            break
            
    print(f"\n🎉 抽查完毕！请去 {save_dir} 文件夹下查看生成的拼图！")

if __name__ == '__main__':
    main()