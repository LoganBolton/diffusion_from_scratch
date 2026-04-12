import math
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
        
    def forward(self, x, t):
        t = self.embed_timestep(t)
        t = self.act(self.fc1(t))
        t = self.fc2(t)
        pass
    
    def embed_timestep(self, t):
        half_dim = self.DIM // 2
        i = torch.arange(half_dim)
        denoms = 10_000 ** (2 * i / self.DIM)
        sin = torch.sin(t / denoms)
        cos = torch.cos(t / denoms)
        embedding = torch.cat([sin, cos], dim=-1)
        return embedding
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.time_dim = time_dim
        self.time_proj = nn.Linear(time_dim, out_channels)
        
        
        self.gn1 = nn.GroupNorm(num_groups = 32, num_channels = self.in_channels)
        self.gn2 = nn.GroupNorm(num_groups = 32, num_channels = self.out_channels)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels = self.in_channels, out_channels=self.out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels = self.out_channels, out_channels=self.out_channels, kernel_size=3, padding=1)
        
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()
    def forward(self, x, t_embed):
        t_embed = self.time_proj(t_embed) # (batch, out_channels)
        t_embed = t_embed.unsqueeze(-1).unsqueeze(-1) # (batch, out_channels, 1, 1)
        h = x
        h = self.gn1(h)
        h = self.act(h)
        h = self.conv1(h)
        
        h = h + t_embed
        h = self.gn2(h)
        h = self.act(h)
        h = self.conv2(h)
        return h + self.skip_conv(x)
    
    
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups = 32, num_channels = channels)
        self.linear_q = nn.Linear(channels, channels)
        self.linear_k = nn.Linear(channels, channels)
        self.linear_v = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels) 
        
    def forward(self, x):
        h = self.gn(x)
        batch, channels, height, width = h.shape
        h = h.reshape(batch, channels, height * width) # (B, C, H, W) -> (B, C, HW)
        h = h.transpose(1, 2) # (B, HW, C)
        q = self.linear_q(h)
        k = self.linear_k(h)
        v = self.linear_v(h)

        qk = q @ k.transpose(-2, -1) # (B, HW, C) @ (B, C, HW)
        sqrt_d = math.sqrt(channels)
        soft_qkv = torch.softmax((qk / sqrt_d), dim=-1) @ v # (B, HW, C)
        
        out = self.output_proj(soft_qkv)
        out = out.reshape(batch, channels, height, width)
        return out + x
        