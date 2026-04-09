import numpy as np
import torch

class DiffusionConstants:
    def __init__(self, timesteps):
        self.timesteps = timesteps
        self.betas = self.get_betas(timesteps)
        self.alphas = 1 - self.betas

        alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1-alpha_bar)

    def get_betas(self, timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, timesteps)
        return betas
    
    def add_noise(self, t, x0):
        epsilon = torch.randn_like(x0)
        reduced_x = self.sqrt_alpha_bar[t] * x0
        noise = self.sqrt_one_minus_alpha_bar[t] * epsilon
        x_t = reduced_x + noise
        return x_t, epsilon