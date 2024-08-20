import torch
from torchvision import transforms


clip_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(
        mean=[0.502, 0.510, 0.423],  # Inat2021mini mean and std
        std=[0.241, 0.234, 0.261],
    )
])

to_tensor_transform = torch.tensor
img_to_tensor_transform = transforms.Compose([transforms.PILToTensor()]) 
