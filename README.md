# diffusion_from_scratch
trying to understand and build a diffusion model from scratch


# Progression

Below is a log of things I add in the learning process.

## v1 - basic DDPM

- Trained a small denoising model on CIFAR-10 images
- Upscaled the 32x32 CIFAR images to be 64x64 
- UNet with self attention.
- Added sinusoidal timestep embeddings

## v2 - distributed training

- Figured out how to make training work across multiple GPUs
- Used Data Distributed Parallelism
- Put model weights on each GPU
    - Split batch samples across both GPUs equally

## v3 - Adding Sampling Code

- After training a model, worte basic sampling code to actually generate images from random noise

## v4 - text conditioning

- Used the text encoder from clip-vit-base-patch32 to embed prompts
- Just did simple predetermined prompts based off CIFAR-10 class, ex: "a photo of a car"
- Added cross attention between text and image

## v5 - add guidance scale

- In the sampling code, generate an image with and without the text prompt
- Use a new hyperparam `w` to encourage the model to generate images more like the text prompt

## v6 - added DDIM sampling

- Optimized the inference process to be 50x faster with minimal degredation to quality 

## v7 - Added DiT architecture 

- Switched from classic U-Net to diffusion transformer. 
- The U-Net performed better in terms of loss and my own qualitative judgement. However, my data is pretty simple with 32x32 images and consistent class level text prompts. I think if I expanded my dataset to be much more complex then DiT would show it's potential more. If anything, this has me thinking more about how scaling isn't as obvious as you migth think with 2026 retrospect. 

## v8 - Added Flow Matching

- Switched from using diffusion to rectified flow for the training objective.

# Things I've learned

### Be very careful about the shape of your tensors!
- Didn't realize that I needed to reshape the `x_t` tensor going through the model depending on what operation is going on 
- Convolutional layers wanted it in form (B, C, H, W)
- Attention layers wanted it in form (B, C, HW)

### Save checkpoints often
Originally, I had my training loop to just save a new checkpoint when it got a better loss. What could go wrong? Actually this was a bad idea. I had a bug in my code that only computed the `best_loss` by checking after each _individual_ batch instead of each epoch (oops). This meant that if I had an outliar batch that just nailed it early on by chance, no other checkpoints would be saved later on in the training run. Instead, I actually wanted to compute the best loss by the average loss of every epoch. I let a training run go overnight and only realized after I only had a checkpoint saved from epoch 10 and not from epoch 120+. Relevant Karpathy quote: "your misconfigured neural net will throw exceptions only if you’re lucky; Most of the time it will train but silently work a bit worse."

### Take some time to **really** understand the details
I spent a lot of time going back and forth with Opus trying to understand the equations before I coded anything. I'm really glad I did this. It saves you a lot of pain trying to understand what you want to make before you actually just jump in and make a dumb decision.

# to do

- add Cosine Noise Schedule
- Precompute VAE latents and text embeddings
- look into EMA
