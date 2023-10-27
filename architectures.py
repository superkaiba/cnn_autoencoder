# Code taken from https://github.com/chenjie/PyTorch-CIFAR-10-autoencoder
# Numpy
import numpy as np
import wandb

# Torch
import torch
import torch.nn as nn

class AutoencoderOriginal(nn.Module):
    def __init__(self):
        super(AutoencoderOriginal, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, kernel_size=4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
			nn.ConvTranspose2d(48, 24, kernel_size=4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, kernel_size=4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, kernel_size=4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class AutoencoderOriginalWider(nn.Module):
    def __init__(self):
        super(AutoencoderOriginalWider, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(6, 12, kernel_size=4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
			nn.ConvTranspose2d(12, 6, kernel_size=4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, kernel_size=4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class AutoencoderOriginalWider(nn.Module):
    def __init__(self):
        super(AutoencoderOriginalWider, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(6, 12, kernel_size=4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
			nn.ConvTranspose2d(12, 6, kernel_size=4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, kernel_size=4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
class AutoencoderWiderLessChannels(nn.Module):
    def __init__(self):
        super(AutoencoderWiderLessChannels, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=4, stride=2, padding=1),          
            nn.ReLU(),
            nn.Conv2d(5, 8, kernel_size=4, stride=2, padding=1),           
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
			nn.ConvTranspose2d(8, 5, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(5, 3, kernel_size=4, stride=2, padding=1),  
            nn.Sigmoid(),
        )
    def encode(self, x):
        encoded = self.encoder(x)
        normalized = self.normalize(encoded)
        return normalized
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded) 
        return encoded, decoded
    
    def normalize(self, x):
        # b, c, h, w = x.shape
        # x = x.view(b, c, -1)
        # min = torch.min(x, dim=2, keepdim=True)[0].unsqueeze(-1) + 1e-7
        # max = torch.max(x, dim=2, keepdim=True)[0].unsqueeze(-1) - 1e-7 
        # x = x.view(b, c, h, w)
        min = torch.min(x) + 1e-7
        max = torch.max(x) - 1e-7
        return (x - min) / (max - min)

class AutoencoderWiderEvenLessChannels(nn.Module):
    def __init__(self):
        super(AutoencoderWiderEvenLessChannels, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(4, 5, kernel_size=4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
			nn.ConvTranspose2d(5, 4, kernel_size=4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(4, 3, kernel_size=4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
class AutoencoderOriginalLessChannels(nn.Module):
    def __init__(self):
        super(AutoencoderOriginalLessChannels, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(6, 9, kernel_size=4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(9, 12, kernel_size=4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
			nn.ConvTranspose2d(12, 9, kernel_size=4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(9, 6, kernel_size=4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, kernel_size=4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class AutoencoderShallow(nn.Module):
    def __init__(self):
        super(AutoencoderShallow, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
                        nn.Conv2d(3, 3, kernel_size=8, stride=4),           
                        nn.ReLU(),
                    )
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(3, 3, kernel_size=8, stride=4),  
                nn.Sigmoid(),
            )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
# https://github.com/rtflynn/Cifar-Autoencoder/blob/master/denoising_autoencoder.py
class AutoencoderDeep(nn.Module):
    def __init__(self):
        super(AutoencoderDeep, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding="same"),           
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding="same"),
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
        )
        self.decoder = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding="same"),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 3, kernel_size=3, stride=1, padding="same"),
                nn.Sigmoid()
            )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
architectures = {
    "original": AutoencoderOriginal,
    "shallow": AutoencoderShallow,
    "deep": AutoencoderDeep,
    "original_less_channels": AutoencoderOriginalLessChannels,
    "original_wider": AutoencoderOriginalWider,
    "original_wider_even_less_channels": AutoencoderWiderEvenLessChannels,
    "original_wider_less_channels": AutoencoderWiderLessChannels,
}