from diffusion import *
import torchvision
import torch

def main():
    # load checkpoint
    model = UNet()
    model.load_state_dict(torch.load("checkpoints/model.pt"))
    model.eval()
    model.to("cuda")

    TIMESTEPS = 1000
    constants = DiffusionConstants(t=TIMESTEPS, device="cuda")
    x_t = torch.randn(1, 3, 64, 64).to("cuda")
    
    with torch.no_grad():
        for t in range(TIMESTEPS -1, -1, -1):
            x_t_minus_1 = constants.sample_step(model, x_t, t)
            x_t = x_t_minus_1
            
            scaled_x_t = (x_t + 1) / 2
            if t % 10 == 0:
                print(f"Step {t}/{TIMESTEPS}")
                torchvision.utils.save_image(scaled_x_t, f"outputs/sample_{t}.png")
    torchvision.utils.save_image(scaled_x_t, f"outputs/sample_{TIMESTEPS}.png")
    

if __name__ == "__main__":
    main()