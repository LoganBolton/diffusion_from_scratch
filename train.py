# uv run torchrun --nproc_per_node=2 train.py

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
import os

from diffusion import *
from text_embedding import ClipTextEncoder

def setup_distributed():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())
    return dist.get_rank()

def cleanup_distributed():
    dist.destroy_process_group()

def main():
    
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

    rank = setup_distributed()
    device = f"cuda:{rank}"
    model = UNet().to(device)
    model = DDP(model, device_ids=[rank])
    clip_text_encoder = ClipTextEncoder(device=device)
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

    best_loss = float('inf')
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        batch_losses = []
        for i, (batch_images, class_idxs) in enumerate(dataloader):
            batch_images = batch_images.to(device)
            batch_size = batch_images.shape[0]

            t = torch.randint(0, T, (batch_size,)).to(device)
            noise_image, noise = constants.add_noise(t, batch_images)
            text_embed = clip_text_encoder.batch_embeds(class_idxs)

            pred_noise = model(noise_image, t.float().unsqueeze(1), text_embed)
            loss = loss_fn(pred_noise, noise)
            batch_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_epoch_loss = sum(batch_losses) / len(batch_losses)
        if rank==0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {average_epoch_loss:.4f}")
            wandb.log({
                "loss": average_epoch_loss,
                "epoch": epoch,
                "batch": i,
            })

        if rank == 0:
            if average_epoch_loss < best_loss:
                best_loss = average_epoch_loss
                print(f"Saving New Checkpoint - New best loss: {best_loss:.4f}")
                torch.save(model.module.state_dict(), f"checkpoints/model_best_epoch_{epoch}.pt")
            elif epoch%20 == 0:
                torch.save(model.module.state_dict(), f"checkpoints/model_epoch_{epoch}.pt")
    cleanup_distributed()
    if rank == 0:
        wandb.finish()

    

if __name__ == "__main__":
    main()
    