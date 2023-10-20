from architectures import architectures
import torch
import numpy as np
def get_encoded_size(model):

    image = torch.tensor(np.random.rand(3, 32, 32), dtype=torch.float32,device='cuda')
    encoded, decoded = model(image.unsqueeze(0))
    return encoded.numel()

if __name__ == "__main__":
    image = torch.tensor(np.random.rand(3, 32, 32), dtype=torch.float32)
    for name, architecture in architectures.items():
        print(f"Verifying architecture {name}----------")
        print("Input n variables", image.numel())
        model = architecture()
        model.eval()
        encoded, decoded = model(image.unsqueeze(0))
        decoded = decoded.squeeze(0)
        print("Encoded shape:", encoded.squeeze(0).shape)
        print("Encoded n variables", encoded.numel())
        print("Decoded shape:", decoded.shape)
        print("image shape:", image.shape)
        assert(decoded.shape == image.shape)
