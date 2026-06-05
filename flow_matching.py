import torch.nn as nn
import torch

class FlowMatching:
    def interpolate(self, t, x0):
        eps = torch.randn_like(x0)
        t = t.reshape(-1, 1, 1, 1)
        
        # $$x_t = (1-t)\,x_0 + t\,\epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$
        x_t = (1-t) * x0 + t * eps
        
        # $$v = \frac{dx_t}{dt} = \frac{d}{dt}\left[(1-t)\,x_0 + t\,\epsilon\right] = -x_0 + \epsilon = \epsilon - x_0$$
        target = eps - x0 

        return x_t, target

       
    def sample_step(self, model, x_t, t, TIMESTEPS, t_prev, text_embed, w):
        t_tensor = torch.full((x_t.shape[0], 1), t*TIMESTEPS, device=x_t.device, dtype=torch.float32)
        v_cond = model(x_t, t_tensor, text_embed)
        v_uncod = model(x_t, t_tensor, torch.zeros_like(text_embed))
        pred_v = v_uncod + w * (v_cond - v_uncod)
        dt = t - t_prev
        x_t_prev = x_t - pred_v * dt
        return x_t_prev