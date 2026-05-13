import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

from diffusion import *

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # scale to [-1, 1]
])

dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)


EPOCHS = 1
T=100
BATCH_SIZE = 32
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
constants = DiffusionConstants(t=T)

for epoch in range(EPOCHS):
    for batch_images, _ in dataloader:
        batch_images = batch_images.to(device)
        batch_size = batch_images.shape[0]

        t = torch.randint(0, T, (batch_size,)).to(device)
        noise_image, noise = constants.add_noise(t, batch_images)
        
        pred_noise = model(noise_image, t.float().unsqueeze(1))
        loss = loss_fn(pred_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f}")

 
