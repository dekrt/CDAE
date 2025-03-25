import argparse
from glob import glob
import os
import random
import numpy as np
from functools import partial

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.cuda.amp import autocast as autocast
from torchvision.datasets import ImageFolder, CIFAR10
from torchvision import transforms
from PIL import Image
import logging

from diffusion import create_diffusion
from download import find_model
from models import DiT_XL_2
from diffusers.models import AutoencoderKL
from models import DiT_models



import sys
sys.path.append("..") 
from utils import init_seeds, gather_tensor, DataLoaderDDP, print0
from datasets import TinyImageNet


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



def get_model(device, ckpt_path):
    model = DiT_XL_2().to(device)
    state_dict = find_model(ckpt_path if ckpt_path is not None else "DiT-XL-2-256x256.pt")
    # state_dict = find_model(f"/lpai/models/ditssl/25-03-04-1/DiT_epoch_1499.pth")
    model.load_state_dict(state_dict)
    model.eval()
    diffusion = create_diffusion(None) # 1000-len betas
    return model, diffusion


def save_model(model, epoch, path="checkpoint.pth"):
    if local_rank == 0:  # Only save from one process (rank 0)
        checkpoint_path = f"{path}_epoch_{epoch}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print0(f"Model saved to {checkpoint_path}")


def denoise_feature(code, model, timestep, blockname, use_amp):
    '''
        Args:
            `image`: Latent codes. (-1, 4, 32, 32) tensor.
            `timestep`: Time step to extract features. int.
            `blockname`: Block to extract features. str.
        Returns:
            Collected feature map.
    '''
    x = code.to(device) # (-1, 4, 32, 32)
    t = torch.tensor([timestep]).to(device).repeat(x.shape[0])
    noise = torch.randn_like(x)
    x_t = diffusion.q_sample(x, t, noise=noise)
    y_null = torch.tensor([1000] * x.shape[0], device=device)

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=use_amp):
            _, acts = model(x_t, t, y_null, ret_activation=True)
        feat = acts[blockname].float().detach()
        # (-1, 256, 1152)
        # we average pool across the sequence dimension to extract
        # a 1152-dimensional vector of features per example
        return feat.mean(dim=1) # (-1, 1152)


class Classifier(nn.Module):
    def __init__(self, feat_func, base_lr, epoch, num_classes):
        super(Classifier, self).__init__()
        self.feat_func = feat_func
        self.loss_fn = nn.CrossEntropyLoss()

        # hidden_size = feat_func(next(iter(valid_loader))[0]).shape[-1]
        sample_img = next(iter(valid_loader))[0].to(device)
        with torch.no_grad():
            sample_latent = vae.encode(sample_img).latent_dist.sample().mul_(0.18215)
        hidden_size = feat_func(sample_latent).shape[-1]
        layers = nn.Sequential(
            # nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, num_classes),
        )
        layers = torch.nn.parallel.DistributedDataParallel(
            layers.to(device), device_ids=[local_rank], output_device=local_rank)
        self.classifier = layers
        self.optim = torch.optim.Adam(self.classifier.parameters(), lr=base_lr)
        self.scheduler = CosineAnnealingLR(self.optim, epoch)

    def train(self, x, y):
        self.classifier.train()
        feat = self.feat_func(x)
        logit = self.classifier(feat)
        loss = self.loss_fn(logit, y)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def test(self, x):
        with torch.no_grad():
            self.classifier.eval()
            feat = self.feat_func(x)
            logit = self.classifier(feat)
            pred = logit.argmax(dim=-1)
            return pred

    def get_lr(self):
        return self.optim.param_groups[0]['lr']

    def schedule_step(self):
        self.scheduler.step()


def train(model, timestep, blockname, epoch, base_lr, use_amp):
    def test():
        preds = []
        labels = []
        for image, label in tqdm(valid_loader, disable=(local_rank!=0)):
            image = image.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(image).latent_dist.sample().mul_(0.18215)
            pred = classifier.test(x)
            preds.append(pred)
            labels.append(label.to(device))

        pred = torch.cat(preds)
        label = torch.cat(labels)
        dist.barrier()
        pred = gather_tensor(pred)
        label = gather_tensor(label)
        acc = (pred == label).sum().item() / len(label)
        return acc

    logger.info(f"Feature extraction: time = {timestep}, name = {blockname}")
    feat_func = partial(denoise_feature, model=model, timestep=timestep, blockname=blockname, use_amp=use_amp)
    DDP_multiplier = dist.get_world_size()
    logger.info("Using DDP, lr = %f * %d" % (base_lr, DDP_multiplier))
    base_lr *= DDP_multiplier
    num_classes = 10 if args.dataset == 'cifar' else 1000

    classifier = Classifier(feat_func, base_lr, epoch, num_classes).to(device)

    for e in range(epoch):
        sampler.set_epoch(e)
        pbar = tqdm(train_loader, disable=(local_rank!=0))
        for i, (image, label) in enumerate(pbar):
            pbar.set_description("[epoch %d / iter %d]: lr: %.1e" % (e, i, classifier.get_lr()))
            image = image.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(image).latent_dist.sample().mul_(0.18215)
            classifier.train(x, label.to(device))
        classifier.schedule_step()
        acc = test()
        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
            logger.info(f"Test acc in epoch {e}: {acc * 100}")
    
    # save layers weights
    if local_rank == 0:
        classifier_path = os.path.join(checkpoint_dir, f"classifier_head.pth")
        torch.save(classifier.classifier.module.state_dict(), classifier_path)
        logger.info(f"Classifier head saved to {classifier_path}")



def get_default_time(dataset, t):
    if t > 0:
        return t
    else:
        return {'cifar': 121, 'imagenet': 81, 'tiny': 81}[dataset]


def get_default_name(dataset, b):
    if b != 'layer-0':
        return b
    else:
        return {'cifar': 'layer-13', 'imagenet': 'layer-13', 'tiny': 'layer-13'}[dataset]
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-path", type=str)
    parser.add_argument("--val-data-path", type=str)
    parser.add_argument("--dataset", default='cifar', type=str, choices=['cifar', 'imagenet', 'tiny'])
    parser.add_argument('--local-rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--use-amp", action='store_true', default=False)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--time', type=int, default=0)
    parser.add_argument('--name', type=str, default='layer-0')
    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model)."
    )
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")



    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    init_seeds(no=local_rank)
    dist.init_process_group(backend='nccl')
    if local_rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        ckpt_string_name = args.ckpt.split('/')[4].replace(".", "-") + '-' + args.ckpt.split('/')[-1].replace(".", "-") 
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{ckpt_string_name}-{args.dataset}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)


    
    torch.cuda.set_device(local_rank)
    device = "cuda:%d" % local_rank
    model, diffusion = get_model(device, args.ckpt)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Setup data:
    if args.dataset == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        train_set = ImageFolder(args.train_data_path, transform=transform)
        valid_set = ImageFolder(args.val_data_path, transform=transform)
    elif args.dataset == 'cifar':
        normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        train_transform = transforms.Compose(
            [
                transforms.Resize(256), 
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.ToTensor(),
                normalize
            ]
        )
        train_set = CIFAR10(root=args.train_data_path, train=True, transform=train_transform)
        valid_set = CIFAR10(root=args.val_data_path, train=False, transform=val_transform)
    elif args.dataset == 'tiny':
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(256, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        train_set = TinyImageNet(root=args.train_data_path, train=True, transform=train_transform)
        valid_set = TinyImageNet(root=args.val_data_path, train=False, transform=val_transform)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")


    train_loader, sampler = DataLoaderDDP(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader, _ = DataLoaderDDP(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # default timestep & blockname values
    args.time = get_default_time(args.dataset, args.time)
    args.name = get_default_name(args.dataset, args.name)

    logger.info(args)
    train(model, timestep=args.time, blockname=args.name, epoch=args.epoch, base_lr=args.lr, use_amp=args.use_amp)
