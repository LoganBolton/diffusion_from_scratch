import torch
import torch.nn as nn

class DiffusionConstants:
    def __init__(self, t):
        self.t = t
        self.betas = self.get_betas(t)
        self.alphas = 1 - self.betas
        alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1-alpha_bar)

    def get_betas(self, t):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, t)
        return betas
    
    def add_noise(self, t, x0):
        epsilon = torch.randn_like(x0)
        reduced_x = self.sqrt_alpha_bar[t] * x0
        noise = self.sqrt_one_minus_alpha_bar[t] * epsilon
        x_t = reduced_x + noise
        return x_t, epsilon
    
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.DIM = 256
        
        self.fc1 = nn.Linear(self.DIM, self.DIM)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(self.DIM, self.DIM)
        
    def forward(self, t):
        t = self.embed_timestep(t)
        t = self.act(self.fc1(t))
        t = self.fc2(t)
        return t
    
    def embed_timestep(self, t):
        half_dim = self.DIM // 2
        i = torch.arange(half_dim)
        denoms = 10_000 ** (2 * i / self.DIM)
        sin = torch.sin(t / denoms)
        cos = torch.cos(t / denoms)
        embedding = torch.cat([sin, cos], dim=-1)
        return embedding