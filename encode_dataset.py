from preprocessing import get_loaders
import torchvision
import torchvision.transforms as transforms
from utils import create_model
import torch
import pandas as pd
import os 
df = pd.DataFrame(columns=["idx", "label", "image_path", "latent_path"])

transform = transforms.Compose(
        [transforms.ToTensor(), ])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
root="/home/mila/t/thomas.jiralerspong/scratch/shared/delta_ai/cnn_autoencoder/cifar_normalized"

autoencoder = create_model("original_wider_less_channels")
autoencoder.load_state_dict(torch.load(f"/home/mila/t/thomas.jiralerspong/scratch/shared/delta_ai/cnn_autoencoder/normalized/weights/original_wider_less_channels/26-10-2023-02-50-44/autoencoder.pkl"))

for i, (image, label) in enumerate(trainset):
    image = image.to("cuda")
    latent = autoencoder.encoder(image)

    train_dir_path_latent = f"{root}/encoded/train"
    os.makedirs(train_dir_path_latent,exist_ok=True)
    latent_fpath = f"{train_dir_path_latent}/{i}.pt"
    torch.save(latent, latent_fpath)

    train_dir_path_original = f"{root}/original/train"
    os.makedirs(train_dir_path_original,exist_ok=True)
    image_fpath = f"{train_dir_path_original}/{i}.pt"
    torch.save(image, image_fpath)

    row = {"idx": i, "label": label, "image_path": image_fpath, "latent_path": latent_fpath}
    df.loc[len(df)] = row

df.to_csv(f"{root}/train.csv")

for i, (image, label) in enumerate(testset):
    image = image.to("cuda")
    latent = autoencoder.encode(image)

    test_dir_path_latent = f"{root}/encoded/test"
    os.makedirs(test_dir_path_latent,exist_ok=True)
    latent_fpath = f"{test_dir_path_latent}/{i}.pt"
    torch.save(latent, latent_fpath)

    test_dir_path_original = f"{root}/original/test"
    os.makedirs(test_dir_path_original,exist_ok=True)
    image_fpath = f"{test_dir_path_original}/{i}.pt"
    torch.save(image, image_fpath)

    row = {"idx": i, "label": label, "image_path": image_fpath, "latent_path": latent_fpath}
    df.loc[len(df)] = row

df.to_csv(f"{root}/test.csv")
# for i, (inputs, _) in enumerate(trainloader, 0):
    