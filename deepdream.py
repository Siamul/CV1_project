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
from models import Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from PIL import Image

def im_convert(tensor):
  image = tensor.to("cpu").clone().detach()
  image = image.numpy().squeeze()
  image = image.transpose(1, 2, 0)
  image = image * np.array((0.5, 0.5, 0.5)) + np.array(
    (0.5, 0.5, 0.5))
  image = image.clip(0, 1)
  np_image = (image * 255).astype(np.uint8)
  return np_image

def load_image(img_path, size=256):
  image = Image.open(img_path).convert('RGB')  
  #print(np.array(image).shape)
  in_transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  
  image = in_transform(image).unsqueeze(0)
  #print(image.shape)
  
  return image

import os

image_loc = '~/driveg/transferredDataset/unrealistic/'
i = 0
targets = []
for filename in os.listdir(image_loc):
  if i > 10:
    break
  image = load_image(image_loc+filename, 256).to(device)
  i+=1
  targets.append(image.clone().requires_grad_(True).to(device))
net = Discriminator(1).to(device)
net.load_state_dict(torch.load('discriminator_weights_finetuned.pth'))
for param in net.parameters():
  param.requires_grad_(False)


im_num = 0
for target in targets:
  im_num += 1
  initial_img = target.clone().detach().cpu()
  optimizer = optim.Adam([target], lr=0.0002)
  criterion = nn.BCELoss()
  i = 1
  while True:
    optimizer.zero_grad()
    target_class = net(target).view(-1)
    if i == 1:
      print(target_class)
      i += 1
    real_label = torch.full((1,), 1, dtype=torch.float, device=device).type(torch.FloatTensor).to(device)
    loss = criterion(target_class, real_label)
    loss.backward(retain_graph=True)
    optimizer.step()  
    #if i == 1:
    #  print(loss.item())
    #if i == 400:
    #  print(loss.item())
    if loss.item() < 0.2:
      break
    #print('Iteration {}, loss: {}'.format(i, loss.item())))
  final_img = target.clone().detach().cpu()
  images = torch.cat(initial_img, final_img)
  vutils.save_image(vutils.make_grid(images, padding=2, normalize=True), 'deepdream_result'+str(im_num)+'.jpg')

