import argparse
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
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image


from diffusion import create_diffusion
from download import find_model
from models import DiT_XL_2
from diffusers.models import AutoencoderKL

import sys

def get_model(device, ckpt_path):
    model = DiT_XL_2().to(device)
    state_dict = find_model(ckpt_path if ckpt_path is not None else "DiT-XL-2-256x256.pt")
    # state_dict = find_model(f"/lpai/models/ditssl/25-03-04-1/DiT_epoch_1499.pth")
    model.load_state_dict(state_dict)
    model.eval()
    diffusion = create_diffusion(None) # 1000-len betas
    return model, diffusion

device = "cuda:1"

model, diffusion = get_model(device, "/lpai/models/ditssl/cadedit100ep/000-DiT-XL-2/checkpoints/0300000.pt")
print(model)