import os
import sys
sys.path.insert(0, "/path/to/zoomldm/root/folder")
import torch
from diffusion import create_diffusion
from models import DiT_models
from diffusion import create_diffusion
# pylint: disable=import-error disable=no-name-in-module
from ldm.data.brca.ssl_hierarchical_fixed import MAG_DICT
from ldm.modules.encoders.modules import EmbeddingViT2_5
from ldm.models.diffusion.plms import PLMSSampler
from scripts.utils import (
    get_model,
    compute_statistics_of_path,
    calculate_activation_statistics,
)
# pylint: enable=import-error enable=no-name-in-module
from train_cdm import my_collate
from torch.utils.data import DataLoader
import tqdm
import numpy as np
from omegaconf import OmegaConf
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from pathlib import Path
import uuid
import shutil
import logging
from datetime import datetime
import re
from PIL import Image

def main(args):

    model_cdm_str = re.search(".*-(DiT-\w+)/.*", args.cdm_ckpt_path).group(1)

    logdir = Path(args.logdir) / model_cdm_str
    logdir.mkdir(exist_ok=True, parents=True)
    now = datetime.now().strftime("%m-%dT%H-%M")
    logging.basicConfig(
        filename=(logdir / f"{args.magnification}_{now}.log"),
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )

    logging.info(args)

    device = torch.device(0)

    model_path = Path(args.ldm_path)
    # pylint: disable=too-many-function-args
    model_ldm = get_model(model_path, device, "last.ckpt")
    # pylint: enable=too-many-function-args
    model_ldm.cond_stage_model.p_uncond = 0

    sampler_ldm = PLMSSampler(model_ldm)


    model_cdm = DiT_models[model_cdm_str]().to(device)
    sd = torch.load(args.cdm_ckpt_path, map_location="cpu")
    model_cdm.load_state_dict(sd['ema'])
    model_cdm.eval()
    
    # diffusion = create_diffusion(timestep_respacing="200", predict_xstart=True)
    diffusion = create_diffusion(timestep_respacing="200", diffusion_steps=1000, noise_schedule="squaredcos_cap_v2", predict_xstart=True)  # embeddings: 1000 steps, cosine noise schedule, x0_pred
    # diffusion = create_diffusion(timestep_respacing="1000", diffusion_steps=1000, noise_schedule="squaredcos_cap_v2", predict_xstart=True)  # embeddings: 1000 steps, cosine noise schedule, x0_pred

    batch_size = 64

    syn_images = []

    m2, s2 = compute_statistics_of_path(args.fid_stats_path)

    unique_id = str(uuid.uuid4())[:8]
    out_dir = logdir / f"images_{args.magnification}"
    out_dir.mkdir(exist_ok=True, parents=True)
    j = 0

    for _ in tqdm.tqdm(range(args.n_images // batch_size)):

        y = torch.tensor([MAG_DICT[args.magnification]] * batch_size).to(device).to(torch.long)
        z = torch.randn(batch_size, 512, 65, device=device)
        model_kwargs = dict(y=y)
        samples_cdm = diffusion.ddim_sample_loop(model_cdm.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, device=device)

        with torch.no_grad(), model_ldm.ema_scope(), torch.autocast(dtype=torch.float16, device_type="cuda"):

            cc = samples_cdm.transpose(1,2)
            uc = torch.zeros_like(cc)

            samples_ddim, _ = sampler_ldm.sample(
                S=args.ddim_steps,
                conditioning=cc,
                batch_size=batch_size,
                shape=[3,64,64],
                verbose=False,
                unconditional_guidance_scale=args.guidance_str,
                unconditional_conditioning=uc,
            )

            x_samples_ddim = model_ldm.decode_first_stage(samples_ddim)
        
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples_ddim = (x_samples_ddim * 255).to(torch.uint8)
        x_samples_ddim = x_samples_ddim.cpu().numpy()

        for img in x_samples_ddim:
            unique_id = str(uuid.uuid4())[:8]
            img = Image.fromarray(img.transpose(1, 2, 0))
            img.save(out_dir / f"{unique_id}.jpg")
            j += 1

    syn_images = []
    for img in out_dir.iterdir():
        if ".jpg" in img.name:
            syn_images.append(np.array(Image.open(img)))

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()
    m1, s1 = calculate_activation_statistics(syn_images, inception, device)

    fid = calculate_frechet_distance(m1, s1, m2, s2)
    logging.info(f"For {args.ldm_path}, {model_cdm_str} FID = {fid}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_images", type=int, default=10000)
    parser.add_argument("--guidance_str", type=float, default=1.75)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ldm_path", type=str, default="/path/to/zoomldm/checkpoint/folder")
    parser.add_argument("--fid_stats_path", type=str, default="/path/to/fid_stats.npz")
    parser.add_argument("--cdm_ckpt_path", type=str, required=True)
    parser.add_argument("--logdir", type=str, default="fid")
    parser.add_argument("--magnification", type=str, default="20x")
    args = parser.parse_args()
    main(args)