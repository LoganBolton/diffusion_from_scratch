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
    
class TimestepEmbed(nn.Module):
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
        
        
class UNet(nn.Module):
    def __init__(self, t_dim=256):
        super().__init__()
        # Image in: 3 channels -> 128 channels
        self.input_conv = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        
        # Encoder level 0: 64x64, 128ch
        self.encode_res1 = ResBlock(128, 128, t_dim)
        self.encode_res2 = ResBlock(128, 128, t_dim)
        self.down1 = nn.Conv2d(128, 128, kernel_size=2, stride=2)
        
        # Encoder level 1: 32x32, 256ch
        self.encode_res3 = ResBlock(128, 256, t_dim)
        self.encode_res4 = ResBlock(256, 256, t_dim)
        self.down2 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        
        # Encoder level 2: 16x16, 512ch 
        self.encode_res5 = ResBlock(256, 512, t_dim)
        self.attn1 = SelfAttention(512)
        self.encode_res6 = ResBlock(512, 512, t_dim)
        self.attn2 = SelfAttention(512)
        self.down4 = nn.Conv2d(512, 512, kernel_size=2, stride=2)

        # Bottleneck: 8x8, 512ch 
        self.bottleneck_res1 = ResBlock(512, 512, t_dim)
        self.bottleneck_attn1 = SelfAttention(512)
        self.bottleneck_res2 = ResBlock(512, 512, t_dim)
        
        # Decoder level 2: 8x8 -> 16x16, 512ch
        self.up3 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.decode_res6 = ResBlock(1024, 512, t_dim)
        self.decode_attn2 = SelfAttention(512)
        self.decode_res5 = ResBlock(512, 512, t_dim)
        self.decode_attn1 = SelfAttention(512)

        # Decoder level 1: 16x16 -> 32x32, 256ch
        self.up2 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)

        self.decode_res4 = ResBlock(768, 256, t_dim)
        self.decode_res3 = ResBlock(256, 256, t_dim)

        # Decoder level 0: 32x32 -> 64x64, 128ch
        self.up1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.decode_res2 = ResBlock(384, 128, t_dim)
        self.decode_res1 = ResBlock(128, 128, t_dim)

        # output
        self.output_conv = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        t_embed = TimestepEmbed(t)
        x = ResBlock.forward(x, self.time_embed)
        x = SelfAttention.forward(x)
        x = ResBlock.forward(x, self.time_embed)
        pass