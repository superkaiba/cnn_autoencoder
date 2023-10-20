import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from architectures import architectures
import wandb
import PIL

import os

def print_model(encoder, decoder):
    print("============== Encoder ==============")
    print(encoder)
    print("============== Decoder ==============")
    print(decoder)
    print("")

def create_model(architecture):
    autoencoder = architectures[architecture]()
    print_model(autoencoder.encoder, autoencoder.decoder)
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
        print("Model moved to GPU in order to speed up training.")
    return autoencoder


def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def imshow(img):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def get_grid(images):
    grid = np.transpose(np.uint8(np.array(torchvision.utils.make_grid(images)) * 255), axes=(1,2,0))
    
    return grid

def log_reconstruction(model, testloader):
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    original_grid = get_grid(images)

    images = Variable(images.cuda())
    decoded_imgs = model(images)[1]
    reconstructed_grid = get_grid(decoded_imgs.cpu())

    total_grid = np.concatenate((original_grid, reconstructed_grid), axis=0)
    wandb_grid = wandb.Image(
                PIL.Image.fromarray(total_grid),
            )
    wandb.log({"images/original_vs_reconstructed": wandb_grid})

def tile(images):
    n_images, c, h, w = images.shape
    images = images.cpu().detach()
    h2 = h//2
    new_images = [None for _ in range(n_images)]
    for i in range(0, n_images, 4):
        new_images[i] = np.concatenate((np.concatenate((images[i,:,:h2,:h2], images[i+1,:,:h2,h2:]), axis=2), np.concatenate((images[i+2,:,h2:,:h2], images[i+3,:,h2:,h2:]), axis=2)), axis=1)
        new_images[i+1] = np.concatenate((np.concatenate((images[i+1,:,:h2,:h2], images[i+2,:,:h2,h2:]), axis=2), np.concatenate((images[i+3,:,h2:,:h2], images[i,:,h2:,h2:]), axis=2)), axis=1)
        new_images[i+2] = np.concatenate((np.concatenate((images[i+2,:,:h2,:h2], images[i+3,:,:h2,h2:]), axis=2), np.concatenate((images[i,:,h2:,:h2], images[i+1,:,h2:,h2:]), axis=2)), axis=1)
        new_images[i+3] = np.concatenate((np.concatenate((images[i+3,:,:h2,:h2], images[i,:,:h2,h2:]), axis=2), np.concatenate((images[i+1,:,h2:,:h2], images[i+2,:,h2:,h2:]), axis=2)), axis=1)
    return torch.tensor(new_images)

def log_reconstruction_tiled(model, testloader):
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    tiled_images = tile(images)
    original_grid = get_grid(tiled_images)

    images = Variable(images.cuda())
    latents = model.encoder(images)
    tiled_latents = tile(latents)
    tiled_latents = torch.tensor(tiled_latents, device='cuda')
    decoded_imgs = model.decoder(tiled_latents)
    reconstructed_grid = get_grid(decoded_imgs.cpu())

    total_grid = np.concatenate((original_grid, reconstructed_grid), axis=0)
    wandb_grid = wandb.Image(
                PIL.Image.fromarray(total_grid),
            )
    wandb.log({"images/original_vs_reconstructed_tiled": wandb_grid})

