import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from models import Generator, Discriminator, weights_init
import sys


netG = Generator(1).to('cpu')
netG.load_state_dict(torch.load(sys.argv[1]))

noise = torch.randn(8, 100, 1, 1, device='cpu')

with torch.no_grad():
    fake = netG(noise).detach().cpu()
vutils.save_image(vutils.make_grid(fake, padding=2, normalize=True), sys.argv[2])
