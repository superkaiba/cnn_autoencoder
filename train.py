# Code taken from https://github.com/chenjie/PyTorch-CIFAR-10-autoencoder
# Numpy
import numpy as np
import wandb

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Torchvision
import torchvision
import torchvision.transforms as transforms

# Matplotlib
import matplotlib.pyplot as plt

# OS
import os
import argparse

from architectures import architectures
from get_args import get_args
from utils import create_model, get_torch_vars, imshow, make_dir, log_reconstruction, log_reconstruction_tiled
from preprocessing import get_loaders
from verify_architecture import get_encoded_size
from datetime import datetime

# Set random seed for reproducibility
SEED = 87
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    
def train(model, criterion, optimizer, trainloader, testloader, args, logdir, n_epochs=100):
    for epoch in range(n_epochs):
        train_loss = 0.0
        num_train_batches = 0
        for i, (inputs, _) in enumerate(trainloader, 0):
            inputs = get_torch_vars(inputs)

            # ============ Forward ============
            encoded, outputs = model(inputs)
            loss = criterion(outputs, inputs)
            # ============ Backward ============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ============ Logging ============
            train_loss += loss.data
            num_train_batches += 1

        test_loss = 0.0
        num_test_batches = 0
        for i, (inputs, _) in enumerate(testloader, 0):
            inputs = get_torch_vars(inputs)

            # ============ Forward ============
            encoded, outputs = model(inputs)
            loss = criterion(outputs, inputs)

            # ============ Logging ============
            test_loss += loss.data
            num_test_batches += 1

        log_reconstruction(model, testloader)
        wandb.log({"train_loss": train_loss / num_train_batches})
        wandb.log({"val_loss": test_loss / num_test_batches})
        if epoch % 10 == 0:
            print("Saving model")
            torch.save(model.state_dict(), f"{logdir}/autoencoder.pkl")

def main():
    args = get_args()
    if args.debug:
        print("DEBUGGING")

    # Create model
    autoencoder = create_model(args.architecture)
    if args.load_dir != "NONE":
        autoencoder.load_state_dict(torch.load(f"{args.load_dir}/autoencoder.pkl"))
    vars(args)["size_encoded"] = get_encoded_size(autoencoder)

    wandb.init(entity="thomasjiralerspong", project="cnn_autoencoder", config=args, name=args.architecture)
    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    logdir = f"{args.log_dir}/weights/{args.architecture}/{dt_string}"
    if not args.debug:
        os.makedirs(logdir, exist_ok=True)
        # make_dir(args.log_dir)
        # make_dir(f"{args.log_dir}/weights")
        # make_dir(f"{args.log_dir}/weights/{args.architecture}/{dt_string}")
    # Define an optimizer and criterion
    criterion = nn.BCELoss()
    optimizer = optim.Adam(autoencoder.parameters())
    trainloader, testloader = get_loaders()
    train(autoencoder, criterion, optimizer, trainloader, testloader, args, logdir, n_epochs=100)


if __name__ == '__main__':
    main()