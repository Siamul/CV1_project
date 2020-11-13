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

# Root directory for dataset
dataroot = './images/'

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 16

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 256

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector
nz = 100

# Number of training epochs
num_epochs = 2000

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.RandomHorizontalFlip(),
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(8, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

#Define Generator and Discriminator
netG = Generator(ngpu).to(device)
netG.load_state_dict(torch.load(sys.argv[1]))
genName = sys.argv[1].split(".")[0]
netD = Discriminator(ngpu).to(device)
netD.load_state_dict(torch.load(sys.argv[2]))
disName = sys.argv[2].split(".")[0]


# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

num_epochs = 2000

#mode collapse prevention parameters

#label noise (randomly flip labels for real data upto label_noise_epochs)
label_noise = False
label_noise_prob = 0.3
label_noise_epochs = 1200

#label smoothing (1 set between 1-x and 1+x, 0 set between 0 and 0+y)
label_smoothing = False
x = 0.3
label_smoothing_epochs = 1200

#instance noise (additive Gaussian Noise to the inputs of the discriminator with annealing upto instance_noise_epochs)
instance_noise = True
instance_noise_epochs = 1200
start_variance = 0.05

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device).type(torch.FloatTensor).to(device)
        if(label_smoothing and epoch < label_smoothing_epochs):
          label = label + ((torch.rand(label.shape) - 0.5) * 2 * x).type(torch.FloatTensor).to(device)
          label = torch.clamp(label, min = 0).type(torch.FloatTensor).to(device)
        if(label_noise and epoch < label_noise_epochs and not label_smoothing):
          label = (torch.FloatTensor(b_size,).uniform_() > (label_noise_prob - ((label_noise_prob/label_noise_epochs)*epoch))).type(torch.FloatTensor).to(device)
        elif(label_noise and epoch < label_noise_epochs and label_smoothing):
          label = (torch.FloatTensor(b_size,).uniform_() > label_noise_prob).type(torch.FloatTensor).to(device)
          label = label + ((torch.rand(label.shape) - 0.5) * 2 * x).type(torch.FloatTensor).to(device)
          label = torch.clamp(label, min = 0).type(torch.FloatTensor).to(device)
        if(instance_noise and epoch < instance_noise_epochs):
          #print((real_cpu.shape[1], real_cpu.shape[2], real_cpu.shape[3]))
          gaussian_noises = [(torch.randn(1, real_cpu.shape[1], real_cpu.shape[2], real_cpu.shape[3]) * (start_variance - (start_variance*epoch)/instance_noise_epochs)).type(torch.FloatTensor).to(device)  for i in range(b_size)]
          gaussian_noise_tensor = torch.cat(gaussian_noises)
 #         print("-----------------------------")
#          print(real_cpu[0][0])
  #        print("-----------------------------")
   #       print(gaussian_noise_tensor[0][0])
          real_cpu = real_cpu + gaussian_noise_tensor
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        if(label_smoothing and epoch < label_smoothing_epochs):
          label = label + ((torch.rand(label.shape) - 0.5) * 2 * x).type(torch.FloatTensor).to(device)
          label = torch.clamp(label, min = 0).type(torch.FloatTensor).to(device)
        # Classify all fake batch with D
        fakeD = fake.detach()
        if(instance_noise and epoch < instance_noise_epochs):
          #print((real_cpu.shape[1], real_cpu.shape[2], real_cpu.shape[3]))
          gaussian_noises = [(torch.randn(1, real_cpu.shape[1], real_cpu.shape[2], real_cpu.shape[3]) * (start_variance - (start_variance*epoch)/instance_noise_epochs)).type(torch.FloatTensor).to(device)  for i in range(b_size)]
          gaussian_noise_tensor = torch.cat(gaussian_noises)
          fakeD = fakeD + gaussian_noise_tensor
        output = netD(fakeD).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        if(label_smoothing):
          label = label + (torch.rand(label.shape) * x).type(torch.FloatTensor).to(device)
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

    # Check how the generator is doing by saving G's output on fixed_noise
    if epoch % 100 == 0:
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        vutils.save_image(vutils.make_grid(fake, padding=2, normalize=True), 'image'+str(epoch)+'.jpg')
    if epoch % 500 == 0:
        torch.save(netG.state_dict(), genName+str(epoch)+'.pth')
        torch.save(netD.state_dict(), disName+str(epoch)+'.pth')

    iters += 1

torch.save(netG.state_dict(), genName+'_final.pth')
torch.save(netD.state_dict(), disName+'_final.pth')
