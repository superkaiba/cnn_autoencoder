from architectures import architectures
from get_args import get_args
from utils import create_model, get_torch_vars, imshow, make_dir, log_reconstruction, log_reconstruction_tiled
from preprocessing import get_loaders
from verify_architecture import get_encoded_size
from datetime import datetime

import torch
import wandb
import torch.nn as nn
import torch.optim as optim
def main():
    args = get_args()
    

    # Create model
    autoencoder = create_model(args.architecture)
    if args.load_dir != "NONE":
        autoencoder.load_state_dict(torch.load(f"{args.load_dir}/autoencoder.pkl"))
        args.log_dir = args.load_dir
    
    vars(args)["size_encoded"] = get_encoded_size(autoencoder)
    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    wandb.init(entity="thomasjiralerspong", project="cnn_autoencoder", config=args, name=f"{args.architecture}-{dt_string}")

    # make_dir(args.log_dir)
    # make_dir(f"{args.log_dir}/weights")
    # make_dir(f"{args.log_dir}/weights/{args.architecture}")
    # Define an optimizer and criterion
    criterion = nn.BCELoss()
    optimizer = optim.Adam(autoencoder.parameters())
    trainloader, testloader = get_loaders()
    log_reconstruction_tiled(autoencoder, testloader)


if __name__ == '__main__':
    main()