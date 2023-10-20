from torch.utils.data import Dataset
import torch
import pandas as pd

class EncodedCIFARDataset(Dataset):
    """Encoded CIFAR dataset."""
    def __init__(self, 
                 train_or_test, 
                 root_dir="/home/mila/t/thomas.jiralerspong/delta_ai/CNN_autoencoder/scratch/cnn_autoencoder/cifar",
                 latent_transform=None,
                 original_transform=None):
        """
        Args:
            root_dir (string): Directory with all the encoded images.
            train_or_test(string): Either "train" or "test"
            latent_transform (callable, optional): Optional transform to be applied
                on latent
            original_transform (callable, optional): Optional transform to be applied
                on original image
        """
        self.train_or_test = train_or_test
        self.root_dir = root_dir
        self.df = pd.read_csv(f"{root_dir}/{self.train_or_test}.csv")
        self.latent_transform = latent_transform
        self.original_transform = original_transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image_path = self.df.iloc[idx, 3]
        latent_path = self.df.iloc[idx, 4]
        label = self.df.iloc[idx, 2]
        
        original = torch.load(image_path, map_location='cpu')
        original.requires_grad = False
        latent = torch.load(latent_path, map_location='cpu')
        latent.requires_grad = False
        
        if self.latent_transform:
            latent = self.latent_transform(latent)
        if self.original_transform:
            original = self.original_transform(original)

        return label, latent, original
    
if __name__ == "__main__":

    trainset = EncodedCIFARDataset(train_or_test="train")
    testset = EncodedCIFARDataset(train_or_test="test")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                                shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                                shuffle=False, num_workers=1)

    for i, (label, latent, original) in enumerate(trainloader):
        print(label)
        print(latent.shape)
        print(original.shape)
        if i == 10:
            break
    for i, (label, latent, original) in enumerate(testloader):
        print(label)
        print(latent.shape)
        print(original.shape)
        if i == 10:
            break
