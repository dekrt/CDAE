# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder, CIFAR10
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR



from pytorch_metric_learning import losses, miners, testers
from pytorch_metric_learning.losses import SelfSupervisedLoss
from info_nce import InfoNCE, info_nce



#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
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


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    # Note that parameter initialization is done within the DiT constructor

    if args.ckpt:
        ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict)


    if args.using_ema:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    # Setup data(imagenet-1k):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)

    # # Using CIFAR10
    # normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    # train_transform = transforms.Compose(
    #     [
    #         transforms.Resize(256), 
    #         transforms.RandomHorizontalFlip(p=0.5),
    #         transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    #         transforms.RandomGrayscale(p=0.2),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]
    # )
    # # dataset = CIFAR10(
    # #     "/lpai/datasets/cifar10-lhp/versions/0.1.0/CIFAR10", train=True, download=False, transform=train_transform
    # # )

    # dataset = CIFAR10(
    #     "/lpai/zxk/ddae/data", train=True, download=False, transform=train_transform
    # )
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    if args.using_ema:
        # Prepare models for training:
        update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
        model.train()  # important! This enables embedding dropout for classifier-free guidance
        ema.eval()  # EMA model should always be in eval mode
    else:
        model.train()  # important! This enables embedding dropout for classifier-free guidance

    if args.freezing_decoder:
        for i in range(14, 28):
            for param in model.module.blocks[i].parameters():
                param.requires_grad = False


    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_mse_loss = 0
    running_contrastive_loss = 0
    start_time = time()

    if args.lambda_lr:
        base_lr = 5e-5  # 最大学习率
        warmup_steps = 1000
        total_steps = args.epochs * len(dataset) // args.global_batch_size  # 训练总步数

        opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0)

        # 定义 warmup + cosine 的调度器
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                # cosine decay: start from 1, decay to 0
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = LambdaLR(opt, lr_lambda)
    else:
        # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)


    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            if args.modified_timesteps:
                t = torch.randint(0, 100, (x.shape[0],), device=device)
            else:
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y, ret_activation=True)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss_dict_positive = diffusion.training_losses(model, x, t // 10, model_kwargs)

            # add contrastive loss
            loss_fn_contrastive = InfoNCE()
            loss_contrastive = loss_fn_contrastive(loss_dict['feat'], loss_dict_positive['feat'])
            loss_mse = (loss_dict["loss"].mean() + loss_dict_positive["loss"].mean()) / 2
            loss = loss_mse + loss_contrastive
            # print(f"epoch {epoch}: loss = {loss}, loss_noise = {loss_noise}, loss_contrastive = {loss_contrastive}")

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            if args.using_ema:
                update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            running_mse_loss += loss_mse.item()
            running_contrastive_loss += loss_contrastive.item()

            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_mse_loss = torch.tensor(running_mse_loss / log_steps, device=device)
                avg_contrastive_loss = torch.tensor(running_contrastive_loss / log_steps, device=device)

                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_mse_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_contrastive_loss, op=dist.ReduceOp.SUM)

                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_mse_loss = avg_mse_loss.item() / dist.get_world_size()
                avg_contrastive_loss = avg_contrastive_loss.item() / dist.get_world_size()
                current_lr = scheduler.get_last_lr()[0]

                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, "
                    f"noise_loss: {avg_mse_loss:.4f}, "
                    f"contrastive_loss: {avg_contrastive_loss:.4f}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}, "
                    f"LR: {current_lr:.6f}"
                )

                # Reset monitoring variables:
                running_loss = 0
                running_mse_loss = 0
                running_contrastive_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    if args.using_ema:
                        checkpoint = {
                            "model": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "args": args
                        }
                    else:
                        checkpoint = {
                            "model": model.module.state_dict(),
                            "opt": opt.state_dict(),
                            "args": args
                        }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
    if args.using_ema:
        checkpoint = {
            "model": model.module.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "args": args
        }
    else:
        checkpoint = {
            "model": model.module.state_dict(),
            "opt": opt.state_dict(),
            "args": args
        }
    # save final model
    final_checkpoint_path = f"{args.results_dir}/final.pt"
    torch.save(checkpoint, final_checkpoint_path)
    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)

    # added params
    parser.add_argument("--blockname", type=str, default='layer-13')
    parser.add_argument('--noise-time', type=int, default=121)
    parser.add_argument('--using-ema', action='store_true', default=False)
    parser.add_argument('--freezing-decoder', action='store_true', default=False)
    parser.add_argument('--modified_timesteps', action='store_true', default=False)
    parser.add_argument('--lambda-lr', action='store_true', default=False)

    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model)."
    )

    args = parser.parse_args()
    main(args)
