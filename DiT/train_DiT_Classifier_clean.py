import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler import CosineLRScheduler
from timm.utils import ModelEma
from vit_pytorch import ViT
from glob import glob
import logging



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

def get_vit_model():
    return ViT(
        image_size=256,
        patch_size=16,
        num_classes=1000,
        dim=1152,
        depth=12,
        heads=16,
        mlp_dim=4608,
        dropout=0.1,
        emb_dropout=0.1
    )


def create_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])


def train(args):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    if rank == 0:
        os.makedirs(args.result_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.result_dir}/*"))
        experiment_dir = f"{args.result_dir}/{experiment_index:03d}-ViT-Classifier"
        os.makedirs(experiment_dir, exist_ok=True)
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
    

    logger.info(args)

    transform = create_transform(args.image_size)

    train_dataset = datasets.ImageFolder(args.train_data_path, transform=transform)
    val_dataset = datasets.ImageFolder(args.val_data_path, transform=transform)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=8, pin_memory=True)

    model = get_vit_model().to(device)
    model = DDP(model, device_ids=[rank])
    model_ema = ModelEma(model, decay=0.9999, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.3)
    lr_scheduler = CosineLRScheduler(optimizer, t_initial=args.epochs * len(train_loader), warmup_t=20 * len(train_loader), warmup_lr_init=1e-6, lr_min=1e-6)
    criterion = SoftTargetCrossEntropy()
    mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, label_smoothing=0.1, num_classes=1000)

    best_acc1 = 0
    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss, correct, total = 0, 0, 0

        # DEBUG: test whether eval success
        # acc1 = evaluate(model_ema.ema, val_loader, device)

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            if mixup_fn is not None:
                images, labels = mixup_fn(images, labels)

            logits = model(images)
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

        if (epoch + 1) % 25 == 0 and rank == 0:
            acc1 = evaluate(model_ema.ema, val_loader, device)
            logger.info(f"Epoch {epoch + 1} finished. Train Acc: {100 * correct / total:.2f}% Val Acc1: {acc1:.2f}%s")
            if acc1 > best_acc1:
                best_acc1 = acc1
                torch.save(model_ema.ema.module.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
                logger.info(f"New best model saved with Acc1: {acc1:.2f}%")

    if rank == 0:
        torch.save(model_ema.ema.module.state_dict(), os.path.join(checkpoint_dir, "final_model.pth"))
        logger.info("Final model saved.")

    dist.destroy_process_group()


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100 * correct / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-path", type=str)
    parser.add_argument("--val-data-path", type=str)
    parser.add_argument("--result-dir", type=str, default="./vit_ckpts")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()
    train(args)
