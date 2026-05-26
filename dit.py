import torch 
import torch.nn as nn
import math

class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp(nn.Sequential)(
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.GELU(),
            nn.Linear(hidden_dim*4, hidden_dim)
        )
        self.adaLN_modulation(nn.Sequential)(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim*6)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        
    def forward(self, x, conditioning):
        h = self.norm1(x)
        mods = self.adaLN_modulation(conditioning)
        y1, b1, a1, y2, b2, a2 = mods.chunk(6, dim=-1)
        
        h = self.norm1(h) * (1+ y1.unsqueeze(1)) + b1.unsqueeze(1)
        h, _ = self.attn(h, h, h)
        x = x + h* a1.unsqueeze(1) # gated 
        
        h = self.norm2(h) * (1 + y2.unsqueeze(1)) + b2.unsqueeze(1)
        x = self.mlp(x)
        out = x + h * a2.unsqueeze(1)

        return out
        