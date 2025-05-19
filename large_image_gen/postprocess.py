import numpy as np
import torch
from einops import rearrange

# Postprocess a generated large image using the diffusion model at the highest available magnification (20x)
# Based on https://github.com/cvlab-stonybrook/Large-Image-Diffusion/blob/main/notebooks/large_image_generation.ipynb

MAG_DICT = {
    "20x": 0,
    "10x": 1,
    "5x": 2,
    "2_5x": 3,
    "1_25x": 4,
    "0_625x": 5,
    "0_3125": 6,
    "0.15625": 7,
}

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """helper function to spherically interpolate two arrays v1 v2"""

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2

def get_conditioning(
    model,
    i,
    j,
    embeddings,
    uncond=False,
    patch_size=64,
    embedding_spatial=(16, 16),
):
    """Find 4 nearest neighbors and extract their embeddings
    e1 - e2       e_top
     |    |  -->    |    -->  e_interp
    e3 - e4       e_bot
    """
    if not uncond:
        # Coordinates of the 4 nearest neighbors
        i1, i2 = (i // patch_size,) * 2
        j1, j3 = (j // patch_size,) * 2
        i3, i4 = (i // patch_size + 1,) * 2
        j2, j4 = (j // patch_size + 1,) * 2

        # Add padding
        embeddings_padded = torch.nn.functional.pad(
            embeddings.reshape(1, -1, embedding_spatial[0], embedding_spatial[1]),
            (0, 1, 0, 1),
            mode="replicate",
        ).view(-1, embedding_spatial[0]+1, embedding_spatial[1]+1)

        # Extract embeddings
        e1 = embeddings_padded[:, i1, j1]
        e2 = embeddings_padded[:, i2, j2]
        e3 = embeddings_padded[:, i3, j3]
        e4 = embeddings_padded[:, i4, j4]
        # Compute distances
        t1 = (j / patch_size - j1) / (j2 - j1)
        t2 = (i / patch_size - i1) / (i3 - i1)

        e_top = slerp(t1, e1, e2)
        e_bot = slerp(t1, e3, e4)
        e_interp = slerp(t2, e_top, e_bot).view(-1,1,1)

        # Normalize
        e_interp = (e_interp - e_interp.mean(0, keepdim=True)) / e_interp.std(0, keepdim=True)
        cond_dict_20x = dict(ssl_feat=[e_interp],
                             mag=torch.tensor([MAG_DICT["20x"]]).long().to(model.device))
        return  model.get_learned_conditioning(cond_dict_20x)

    else:
        return torch.zeros((1,65,512), device=model.device)


def postprocess_image(model, xt_20x_all, ssl_feat, t0, stride=50, guidance=3.0, sliding_window_size=16, emb_h=4, emb_w=4, batch_size=16):
    device = model.device

    # Add noise
    atbar = model.alphas_cumprod[t0-1].view(1,1,1,1).to(device)
    xt_20x_all_postprocessed = torch.sqrt(atbar)*xt_20x_all.clone() + torch.sqrt(1-atbar)*torch.randn_like(xt_20x_all)
    x = rearrange(xt_20x_all_postprocessed, '(b p1 p2) c h w -> b c (p1 h) (p2 w)', p1=emb_h, p2=emb_w)

    # Conditioning
    lt_sz = 64
    img_cond_lis = []
    no_cond_lis = []
    for j in range(0, x.shape[2] - lt_sz + 1, sliding_window_size):
        for k in range(0, x.shape[3] - lt_sz + 1, sliding_window_size):
            # Prompt
            with torch.no_grad():
                img_cond = get_conditioning(model, j, k, ssl_feat, uncond=False, embedding_spatial=(emb_h, emb_w))
                no_cond = get_conditioning(model, j, k, ssl_feat, uncond=True, embedding_spatial=(emb_h, emb_w))

            img_cond_lis.append(img_cond)
            no_cond_lis.append(no_cond)
    img_cond_lis = torch.vstack(img_cond_lis)
    no_cond_lis = torch.vstack(no_cond_lis)
    batch_size = len(img_cond_lis)

    for idx, t in enumerate(range(t0, 0, -stride)):
        atbar = model.alphas_cumprod[t-1].view(1,1,1,1).to(device)
        atbar_prev = model.alphas_cumprod[max(t-1-stride,0)].view(1,1,1,1).to(device)
        beta_tilde = (model.betas[t-1] * (1 - atbar_prev) / (1 - atbar)).view(1,1,1,1).to(device)

        x = rearrange(xt_20x_all_postprocessed, '(b p1 p2) c h w -> b c (p1 h) (p2 w)', p1=emb_h, p2=emb_w)
        t_cond = torch.tensor(batch_size * 2 * [t]).view(-1)
        # Denoise sliding window views
        with torch.no_grad():
            eps_map = torch.zeros_like(x)
            x0_map = torch.zeros_like(x)
            avg_map = torch.zeros_like(x)

            x_slice_lis = []
            indices_map = {}
            for j in range(0, x.shape[2] - lt_sz + 1, sliding_window_size):
                for k in range(0, x.shape[3] - lt_sz + 1, sliding_window_size):
                    # Prompt

                    x_slice = x[:, :, j : j + lt_sz, k : k + lt_sz]
                    x_slice_lis.append(x_slice)
                    indices_map[j, k] = len(x_slice_lis) - 1

            x_slice_lis = torch.vstack(x_slice_lis)

            cond_out = []
            uncond_out = []
            for idx_20x in range(0, x_slice_lis.shape[0], batch_size):
                with torch.cuda.amp.autocast():
                    combined_cond = torch.vstack([img_cond_lis[idx_20x:idx_20x+batch_size,...], no_cond_lis[idx_20x:idx_20x+batch_size,...]])
                    combined_x = torch.vstack([x_slice_lis[idx_20x:idx_20x+batch_size,...]] * 2)
                    t_cond = torch.tensor(combined_x.shape[0] * [t]).view(-1)
                    combined_out = model.model.diffusion_model(
                        combined_x,
                        t_cond.float().to(device),
                        context=combined_cond,
                    )

                    cond_out_batch, uncond_out_batch = torch.tensor_split(combined_out, 2)

                    cond_out.append(cond_out_batch)
                    uncond_out.append(uncond_out_batch)

            cond_out = torch.cat(cond_out, dim=0)
            uncond_out = torch.cat(uncond_out, dim=0)

            epsilon_combined = (1 + guidance) * cond_out - guidance * uncond_out
            x0_combined = (x_slice_lis / torch.sqrt(atbar)) - (epsilon_combined * torch.sqrt((1 - atbar) / atbar))

            for j in range(0, x.shape[2] - lt_sz + 1, sliding_window_size):
                for k in range(0, x.shape[3] - lt_sz + 1, sliding_window_size):
                    idx = indices_map[j, k]
                    epsilon_slice = epsilon_combined[idx]

                    x0_slice = x0_combined[idx]

                    eps_map[:, :, j : j + lt_sz, k : k + lt_sz] += epsilon_slice
                    x0_map[:, :, j : j + lt_sz, k : k + lt_sz] += x0_slice
                    avg_map[:, :, j : j + lt_sz, k : k + lt_sz] += 1

            x0_pred = x0_map / avg_map
            epsilon = (x.float() - torch.sqrt(atbar) * x0_pred) / torch.sqrt(1 - atbar)

        # Predict next step
        x_prev = torch.sqrt(atbar_prev)*x0_pred + torch.sqrt(1-atbar_prev-beta_tilde)*epsilon + torch.sqrt(beta_tilde)*torch.randn_like(x)
        xt_20x_all_postprocessed = rearrange(x_prev, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', p1=emb_h, p2=emb_w)

    return xt_20x_all_postprocessed