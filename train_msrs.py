"""
MSRS语义可控融合训练脚本
使用方法:

第一次训练
python train_msrs.py --data_root /root/workspace/MSRS

断点续训
python train_msrs.py \
--data_root /root/workspace/MSRS \
--resume ./experiments/MSRS_train_xxx/weights/latest.pth \
--epochs 50
"""

import os
import argparse
import clip
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import datetime
import warnings

from model.Text_IF_model import Text_IF as create_model
from data.msrs_dataset import MSRSDataset
import transforms_msrs as T
from scripts.losses import fusion_prompt_loss

warnings.filterwarnings("ignore", category=UserWarning)


# ================================
# Train
# ================================
def train_one_epoch(model, model_clip, optimizer, lr_scheduler,
                    data_loader, device, epoch, loss_fn):

    model.train()
    model_clip.eval()

    total_loss = 0

    for step, data in enumerate(data_loader):

        I_vi, I_ir, mask, class_masks_list, descriptions, names = data

        text = clip.tokenize(list(descriptions)).to(device)

        I_vi = I_vi.to(device)
        I_ir = I_ir.to(device)
        mask = mask.to(device)

        # forward
        I_fused = model(I_vi, I_ir, text)

        # 自动任务识别
        dynamic_tasks = []
        for desc in descriptions:
            if "vehicle" in desc:
                dynamic_tasks.append("enhance_vehicle")
            elif "person" in desc:
                dynamic_tasks.append("enhance_person")
            else:
                dynamic_tasks.append("enhance_background")

        loss, *_ = loss_fn(I_vi, I_ir, I_fused, task=dynamic_tasks, mask=mask)

        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if step % 10 == 0:
            print(f"[Epoch {epoch} Step {step}] Loss: {loss.item():.4f}")

    return total_loss / (step + 1)


# ================================
# Validation
# ================================
@torch.no_grad()
def evaluate(model, model_clip, data_loader, device, epoch, loss_fn, save_path):

    model.eval()
    total_loss = 0

    for step, data in enumerate(data_loader):

        I_vi, I_ir, mask, class_masks_list, descriptions, names = data

        text = clip.tokenize(list(descriptions)).to(device)

        I_vi = I_vi.to(device)
        I_ir = I_ir.to(device)
        mask = mask.to(device)

        I_fused = model(I_vi, I_ir, text)

        dynamic_tasks = []
        for desc in descriptions:
            if "vehicle" in desc:
                dynamic_tasks.append("enhance_vehicle")
            elif "person" in desc:
                dynamic_tasks.append("enhance_person")
            else:
                dynamic_tasks.append("enhance_background")

        loss, *_ = loss_fn(I_vi, I_ir, I_fused, task=dynamic_tasks, mask=mask)

        total_loss += loss.item()

        # 保存验证图
        if step < 3:
            import torchvision
            img_name = os.path.join(save_path, f"epoch_{epoch}_{names[0]}.png")
            torchvision.utils.save_image(I_fused, img_name)

    return total_loss / (step + 1)


# ================================
# Main
# ================================
def main(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ==========================
    # 实验目录
    # ==========================
    if args.resume != "":

        checkpoint = torch.load(args.resume, map_location=device)

        # 找到原实验目录
        exp_path = os.path.dirname(os.path.dirname(args.resume))

        filefold_path = exp_path

        print("Resume experiment:", filefold_path)

    else:

        os.makedirs("./experiments", exist_ok=True)

        time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filefold_path = f"./experiments/MSRS_train_{time_str}"

        os.makedirs(filefold_path)
        os.makedirs(os.path.join(filefold_path, "weights"))
        os.makedirs(os.path.join(filefold_path, "images"))

    tb_writer = SummaryWriter(log_dir=os.path.join(filefold_path, "log"))

    # ==========================
    # Dataset
    # ==========================
    train_transform = T.Compose([
        T.RandomCrop(256),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.ToTensor(),
    ])

    val_transform = T.Compose([
        T.Resize_16(),
        T.ToTensor(),
    ])

    train_dataset = MSRSDataset(
        args.data_root,
        phase='train',
        transform=train_transform,
        descriptions_file=args.descriptions_file,
        use_semantic_prompt=True
    )

    val_dataset = MSRSDataset(
        args.data_root,
        phase='test',
        transform=val_transform,
        descriptions_file=args.descriptions_file,
        use_semantic_prompt=True
    )

    print("Train:", len(train_dataset))
    print("Val:", len(val_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=MSRSDataset.collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=MSRSDataset.collate_fn
    )

    # ==========================
    # Model
    # ==========================
    model_clip, _ = clip.load("ViT-B/32", device=device)

    model = create_model(model_clip).to(device)

    for param in model.model_clip.parameters():
        param.requires_grad = False

    # ==========================
    # Optimizer
    # ==========================
    pg = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5e-2)

    # ==========================
    # Scheduler
    # ==========================
    def lr_lambda(step):

        warmup = len(train_loader) * args.warmup_epochs

        if step < warmup:
            return step / warmup

        progress = (step - warmup) / (args.epochs * len(train_loader) - warmup)

        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ==========================
    # Loss
    # ==========================
    loss_fn = fusion_prompt_loss().to(device)

    # ==========================
    # Resume
    # ==========================
    start_epoch = 0

    if args.resume != "":

        model.load_state_dict(checkpoint['model'], strict=False)

        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

        if 'scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['scheduler'])

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1

        print("Resume from epoch:", start_epoch)

    # ==========================
    # Training Loop
    # ==========================
    best_loss = 1e10

    for epoch in range(start_epoch, args.epochs):

        train_loss = train_one_epoch(
            model, model_clip, optimizer, lr_scheduler,
            train_loader, device, epoch, loss_fn
        )

        if epoch % args.val_every == 0:

            val_loss = evaluate(
                model, model_clip,
                val_loader, device,
                epoch, loss_fn,
                os.path.join(filefold_path, "images")
            )

            print(f"Epoch {epoch} Train:{train_loss:.4f} Val:{val_loss:.4f}")

            tb_writer.add_scalar("train_loss", train_loss, epoch)
            tb_writer.add_scalar("val_loss", val_loss, epoch)

            if val_loss < best_loss:

                best_loss = val_loss

                torch.save({

                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict()

                }, os.path.join(filefold_path, "weights", "best.pth"))

                print("Saved best model")

        torch.save({

            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict()

        }, os.path.join(filefold_path, "weights", "latest.pth"))

    print("Training completed!")


# ================================
# Args
# ================================
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default='/root/workspace/MSRS')

    parser.add_argument('--descriptions_file', type=str, default=None)

    parser.add_argument('--epochs', type=int, default=50)

    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--warmup_epochs', type=int, default=1)

    parser.add_argument('--val_every', type=int, default=5)

    parser.add_argument('--resume', type=str, default="")

    parser.add_argument('--gpu_id', type=str, default='0')

    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    main(args)