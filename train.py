# uv run torchrun --nproc_per_node=2 train.py

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb

from diffusion import *

def setup_distributed():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())
    return dist.get_rank()


def cleanup_distributed():
    dist.destroy_process_group()

def main():
    rank = setup_distributed()
    

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # scale to [-1, 1]
    ])

    dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=False,
        transform=transform
    )

    EPOCHS = 1000
    T=1000
    BATCH_SIZE = 128
    LR = 1e-4


    device = f"cuda:{rank}"
    model = UNet().to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, sampler=sampler, num_workers=2)
    constants = DiffusionConstants(t=T, device=device)

    if rank == 0:
        wandb.init(project="diffusion-from-scratch", config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "t": T,
        })

    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        for i, (batch_images, _) in enumerate(dataloader):
            batch_images = batch_images.to(device)
            batch_size = batch_images.shape[0]

            t = torch.randint(0, T, (batch_size,)).to(device)
            noise_image, noise = constants.add_noise(t, batch_images)

            pred_noise = model(noise_image, t.float().unsqueeze(1))
            loss = loss_fn(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0 and rank == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {i}/{len(dataloader)} | Loss: {loss.item():.4f}")
                wandb.log({
                    "loss": loss.item(),
                    "epoch": epoch,
                    "batch": i,
                })

        if rank==0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f}")
    cleanup_distributed()
    if rank == 0:
        torch.save(model.module.state_dict(), "checkpoints/model.pt")
        wandb.finish()

    

if __name__ == "__main__":
    main()
    