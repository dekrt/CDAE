# basic
import os
import math
import argparse
from time import time

# torch
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder

# aug
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import CosineLRScheduler
from timm.utils import ModelEma

# misc
import numpy as np
from PIL import Image
import logging
from glob import glob

# model
from models import DiT_XL_2, ViTBlock
from diffusers.models import AutoencoderKL

class DiTEncoderClassifier(nn.Module):
    def __init__(self, pretrained_dit, num_classes=1000, num_blocks=14, pool='mean'):
        super().__init__()
        self.patch_embed = pretrained_dit.x_embedder
        self.pos_embed = pretrained_dit.pos_embed
        hidden_dim = pretrained_dit.blocks[0].attn.qkv.in_features
        self.blocks = nn.Sequential(*[
            ViTBlock(dim=hidden_dim, num_heads=pretrained_dit.num_heads, drop_path=0.2) for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.pool = pool
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 4096),
            nn.GELU(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        return self.head(x)

def create_logger(logging_dir):
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        return logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
        return logger

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def train(args):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    if rank == 0:
        os.makedirs(args.result_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.result_dir}/*"))
        experiment_dir = f"{args.result_dir}/{experiment_index:03d}-DiT-Classifier"
        os.makedirs(experiment_dir, exist_ok=True)
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    logger.info(args)

    transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    transform_val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    train_dataset = ImageFolder(args.train_data_path, transform=transform_train)
    val_dataset = ImageFolder(args.val_data_path, transform=transform_val)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=8, pin_memory=True)

    base_dit = DiT_XL_2()
    model = DiTEncoderClassifier(base_dit).to(device)
    model = DDP(model, device_ids=[rank])
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.3)
    lr_scheduler = CosineLRScheduler(optimizer, t_initial=args.epochs * len(train_loader), warmup_t=20 * len(train_loader), warmup_lr_init=1e-6, lr_min=1e-6)

    mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, label_smoothing=0.1, num_classes=1000)
    criterion = SoftTargetCrossEntropy()
    model_ema = ModelEma(model, decay=0.9999, device=device)

    best_acc1 = 0
    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss, correct, total = 0, 0, 0
        start_time = time()

        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(torch.long).to(device)

            with torch.no_grad():
                latent = vae.encode(images).latent_dist.sample().mul_(0.18215)

            if mixup_fn is not None:
                latent, labels = mixup_fn(latent, labels)

            logits = model(latent)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step(epoch * len(train_loader) + step)
            model_ema.update(model)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            target = labels.argmax(dim=1) if labels.ndim == 2 else labels
            correct += (preds == target).sum().item()

            total += labels.size(0)

            if step % 100 == 0 and rank == 0:
                logger.info(f"Epoch [{epoch}/{args.epochs}] Step [{step}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        if epoch % 50 == 0:
            acc1 = evaluate(model, val_loader, device, vae)
            if rank == 0:
                print(acc1)
                logger.info(f"Epoch {epoch} finished. Train Acc: {100 * correct / total:.2f}% Val Acc1: {acc1:.2f}%s")
                if acc1 > best_acc1:
                    best_acc1 = acc1
                    torch.save(model_ema.ema.module.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
                    logger.info(f"New best model saved with Acc1: {acc1:.2f}%")

    if rank == 0:
        torch.save(model_ema.ema.module.state_dict(), os.path.join(checkpoint_dir, 'final_model.pth'))
        logger.info("Final model saved.")

    dist.destroy_process_group()

@torch.no_grad()
def evaluate(model, val_loader, device, vae):
    model.eval()
    correct, total = 0, 0
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device=device, dtype=torch.long)
        latent = vae.encode(images).latent_dist.sample().mul_(0.18215)
        logits = model(latent)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return 100 * correct / total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-path", type=str)
    parser.add_argument("--val-data-path", type=str)
    parser.add_argument('--result-dir', type=str, default='./classifier_ckpts')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image-size', type=int, default=256)
    args = parser.parse_args()
    train(args)
