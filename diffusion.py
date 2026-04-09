import numpy as np
import torch


class DiffusionConstants:
    def __init__(self, timesteps, x0):
        self.timesteps = timesteps
        self.x0 = x0
        
        betas = self.get_betas(timesteps)
        alphas = self.get_alpha(betas, timesteps)
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1-alpha_bar)
        self.epsilon = torch.randn_like(x0)

    def get_betas(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        betas = np.linspace(beta_start, beta_end, timesteps)
        return betas
        
    def get_alphas(betas, timesteps):
        alphas = np.empty(timesteps)
        for t in range(timesteps):
            alphas[t] = 1 - betas[t]
        return alphas

    
    def add_noise(t):
        reduced_x = self.sqrt_alpha_bar[t] * self.x0
        noise = self.sqrt_one_minus_alpha_bar[t] * self.epsilon
        x_t = reduced_x + noise
        return x_t