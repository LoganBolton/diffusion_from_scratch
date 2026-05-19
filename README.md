# diffusion_from_scratch
trying to understand and build a diffusion model from scratch



## todo

                                                                                           
 The user has 2 GPUs and wants to use both for training. DDP is the standard approach — each GPU
  runs its own process with a full model replica, and gradients are synced via all-reduce.

 Files to modify

 - train.py — restructure for DDP
 - diffusion.py — make DiffusionConstants accept a device argument instead of hardcoding
 torch.cuda.is_available()

 Changes

 1. diffusion.py — DiffusionConstants

 - Add a device parameter to __init__ so each process can place tensors on its own GPU (cuda:0,
 cuda:1)
 - Replace the hardcoded 'cuda' if torch.cuda.is_available() else 'cpu' on lines 13-14 with
 self.device

 2. train.py — DDP setup

 - Add imports: torch.distributed, torch.nn.parallel.DistributedDataParallel,
 torch.utils.data.distributed.DistributedSampler
 - Add setup()/cleanup() functions to init/destroy the process group (nccl backend)
 - Wrap training logic in a main() function that:
   - Reads local_rank from environment (set automatically by torchrun)
   - Sets torch.cuda.set_device(local_rank)
   - Creates model on cuda:{local_rank}, wraps in DistributedDataParallel
   - Creates DistributedSampler for the dataset (replaces shuffle=True)
   - Calls sampler.set_epoch(epoch) each epoch for proper shuffling
   - Passes device to DiffusionConstants
   - Only prints from rank 0
 - Guard with if __name__ == "__main__"

 3. How to launch

 Instead of uv run python train.py, run:
 uv run torchrun --nproc_per_node=2 train.py

 Verification

 - Run uv run torchrun --nproc_per_node=2 train.py
 - Confirm both GPUs are utilized (loss prints from rank 0 only)
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌


also after train, make thing that saves outputs