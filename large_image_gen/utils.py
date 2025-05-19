import torch
from einops import rearrange

# Model prediction wrapper
def model_pred(model, xt, t, cond, w=0):
    t_cond = torch.tensor([t]).float().to(model.device).tile(xt.shape[0]).view(-1)
    with torch.cuda.amp.autocast():
        with model.ema_scope():
            with torch.no_grad():
                if w != 0:
                    bs = xt.shape[0]
                    pred_eps = model.model.diffusion_model(
                        torch.cat((xt, xt), dim=0),
                        torch.cat((t_cond, t_cond), dim=0),
                        torch.cat((cond, 0*cond), dim=0)
                    )
                    pred_eps = (w+1)*pred_eps[:bs,...] - w*pred_eps[bs:,...]
                else:
                    pred_eps = model.model.diffusion_model(xt, t_cond, cond)
    return pred_eps

def gaussian_kernel(size=64, mu=0, sigma=1):
    x = torch.linspace(-1, 1, size)
    x = torch.stack((x.tile(size, 1), x.tile(size, 1).T), dim=0)

    d = torch.linalg.norm(x - mu, dim=0)
    x = torch.exp(-(d**2) / sigma**2)
    x = x / x.max()
    return x
    
def decode_large_image(latent, model, sliding_window_size=16, sigma=0.8):
    f = 4
    lt_sz = 64
    out_img = torch.zeros((latent.shape[0], 3, 4 * latent.shape[2], 4 * latent.shape[3])).to(latent.device)
    avg_map = torch.zeros_like(out_img).to(latent.device)

    # Blending kernel that focuses at the center of each patch
    kernel = gaussian_kernel(size=f * lt_sz, sigma=sigma).to(model.device)

    for i in range(0, latent.shape[2] - lt_sz + 1, sliding_window_size):
        for j in range(0, latent.shape[3] - lt_sz + 1, sliding_window_size):
            with torch.no_grad():
                decoded = model.decode_first_stage(latent[:, :, i : i + lt_sz, j : j + lt_sz])
                out_img[:, :, i * f : (i + lt_sz) * f, j * f : (j + lt_sz) * f] += decoded * kernel.view(1, 1, 256, 256)
                avg_map[:, :, i * f : (i + lt_sz) * f, j * f : (j + lt_sz) * f] += torch.ones_like(
                    decoded
                ) * kernel.view(1, 1, 256, 256)

    out_img /= avg_map
    out_img = torch.clamp((out_img + 1) / 2.0, min=0.0, max=1.0)
    out_img = (out_img * 255).to(torch.uint8)
    return out_img.cpu().numpy().transpose([0, 2, 3, 1])
