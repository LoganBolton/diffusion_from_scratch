import torch
import torch.nn as nn
        
class DiffusionConstants:
    def __init__(self, t, device):
        self.t = t
        self.betas = self.get_betas(t)
        self.alphas = 1 - self.betas
        alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar = alpha_bar.to(device)
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1-alpha_bar)
        
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)

    def get_betas(self, t):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, t)
        return betas
    
    def add_noise(self, t, x0):
        epsilon = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bar[t].reshape(-1, 1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].reshape(-1, 1, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_omab * epsilon
        return x_t, epsilon
    
    def sample_step(self, model, x_t, t, t_prev, text_embed, w):
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t]
        t_tensor = torch.full((x_t.shape[0], 1), t, device=x_t.device, dtype=torch.float32)

        noise_cod = model(x_t, t_tensor, text_embed)
        noise_uncod = model(x_t, t_tensor, torch.zeros_like(text_embed))
        pred_noise = noise_uncod + w * (noise_cod - noise_uncod)

        # DDPM (slow) $$x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\hat{\epsilon}_{\theta}\right)$$
        # x_t_minus_1 = (1 / torch.sqrt(alpha)) * (x_t - (beta / sqrt_one_minus_alpha_bar) * pred_noise)
        # if t > 0:
        #     z = torch.randn_like(x_t)
        #     sigma_t = torch.sqrt(beta)
        #     # $$x_{t-1}=x_{t-1}+\sigma_tz$$
        #     x_t_minus_1 = x_t_minus_1 + sigma_t * z

        # DDIM (fast)
        
        # $$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\, \epsilon_\theta}{\sqrt{\bar{\alpha}_t}}$$
        clean_prediction = (x_t - sqrt_one_minus_alpha_bar * pred_noise) \
            / self.sqrt_alpha_bar[t]

        # $$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\, \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}}\, \epsilon_\theta$$
        x_t_minus_1 = self.sqrt_alpha_bar[t_prev] * clean_prediction \
            + self.sqrt_one_minus_alpha_bar[t_prev] * pred_noise
        return x_t_minus_1
        
    
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
        i = torch.arange(half_dim, device=t.device)
        denoms = 10_000 ** (2 * i / self.DIM)
        sin = torch.sin(t / denoms)
        cos = torch.cos(t / denoms)
        embedding = torch.cat([sin, cos], dim=-1)
        return embedding
    