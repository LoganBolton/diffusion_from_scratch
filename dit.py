import torch 
import torch.nn as nn
import math
from diffusion import TimestepEmbed

class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.GELU(),
            nn.Linear(hidden_dim*4, hidden_dim)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim*6)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        
    def forward(self, x, conditioning):
        mods = self.adaLN_modulation(conditioning)
        y1, b1, a1, y2, b2, a2 = mods.chunk(6, dim=-1)
        
        # Attention block
        h = self.norm1(x) * (1+ y1.unsqueeze(1)) + b1.unsqueeze(1)
        h, _ = self.attn(h, h, h)
        x = x + h* a1.unsqueeze(1) # gated 
        
        # MLP block
        h = self.norm2(x) * (1 + y2.unsqueeze(1)) + b2.unsqueeze(1)
        h = self.mlp(h)
        x = x + h * a2.unsqueeze(1)

        return x
    
class DiT(nn.Module):
    def __init__(self, img_size=64, patch_size=4, hidden_dim=512, num_heads=8, num_layers=6):
        super().__init__()
        self.patch_proj = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        self.time_embed = TimestepEmbed()
        self.time_proj = nn.Linear(256, hidden_dim)
        self.text_proj = nn.Linear(512, hidden_dim)
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_linear = nn.Linear(hidden_dim, patch_size * patch_size * 3)
    
    def patchify(self, x):
        x = self.patch_proj(x) # b, hidden_dim, 16, 16
        x = x.flatten(2) # b, hidden_dim, 256
        x = x.transpose(1, 2) # b, 256, hidden_dim
        return x
    
    def unpatchify(self, x):
        # x is (batch, 256, patch_size * patch_size * 3)
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)          # 16 — grid size

        x = x.reshape(x.shape[0], h, w, p, p, 3)  # lay out the grid and patch pixels
        x = x.permute(0, 5, 1, 3, 2, 4)            # (batch, 3, h, p, w, p)
        x = x.reshape(x.shape[0], 3, h * p, w * p) # (batch, 3, 64, 64)
        return x

    def forward(self, x, t, text_embed):
        x = self.patchify(x)
        x = x + self.pos_embed
        t_embed = self.time_proj(self.time_embed(t))
        text_embed = self.text_proj(text_embed)
        cond = t_embed + text_embed.mean(dim=1)
        
        for block in self.blocks:
            x = block(x, cond)
        x = self.final_norm(x)
        x = self.final_linear(x)
        return self.unpatchify(x)
    
    