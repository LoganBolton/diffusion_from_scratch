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

       
    def sample_step():
        pass