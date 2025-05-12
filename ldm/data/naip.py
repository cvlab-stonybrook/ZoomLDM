from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import h5py
import io
import torch.nn.functional as F
import torch
from einops import rearrange

MAG_DICT = {
    "1x": 0,
    "2x": 1,
    "3x": 2,
    "4x": 3,
}

MAG_NUM_IMGS = {
    "1x": 365119,
    "2x": 94263,
    "3x": 25690,
    "4x": 8772,
}



class NAIPDataset(Dataset):
    def __init__(self, config=None):
        self.root = Path(config.get("root"))
        self.mag = config.get("mag", None)

        self.keys = list(MAG_DICT.keys())
        self.feat_target_size = config.get("feat_target_size", -1)
        self.return_image = config.get("return_image", False)
        self.normalize_ssl = config.get("normalize_ssl", False)


    def __len__(self):
        if self.mag:
            return MAG_NUM_IMGS[self.mag]
        return sum(MAG_NUM_IMGS.values())

    def __getitem__(self, idx):
        if self.mag:
            mag_choice = self.mag
        else:
            mag_choice = np.random.choice(self.keys)
            # pick a random index
            idx = np.random.randint(0, MAG_NUM_IMGS[mag_choice])

        folder_path = self.root / f"{mag_choice}/"

        vae_feat = np.load(folder_path / f"{idx}_vae.npy").astype(np.float16)

        ssl_feat = np.load(folder_path / f"{idx}_dino_grid.npy").astype(np.float16)

        h = np.sqrt(ssl_feat.shape[0]).astype(int)

        ssl_feat = torch.tensor(rearrange(ssl_feat, "(h1 h2) dim -> dim h1 h2", h1 = h))

        # resize ssl_feat
        if self.feat_target_size != -1 and h > self.feat_target_size:
            shape = (self.feat_target_size, self.feat_target_size)
            ssl_feat = F.adaptive_avg_pool2d(ssl_feat, shape)

        # normalize ssl_feat
        if self.normalize_ssl:
            mean = ssl_feat.mean(axis=0, keepdims=True)
            std = ssl_feat.std(axis=0, keepdims=True)
            ssl_feat = (ssl_feat - mean) / (std + 1e-8)


        #### load image
        if self.return_image:
            image = Image.open(folder_path / f"{idx}.jpg")
            image = np.array(image).astype(np.uint8)

        else:
            image = np.ones((1, 1, 1, 3), dtype=np.float16)

        return {
            "image": image,
            "vae_feat": vae_feat,
            "ssl_feat": ssl_feat,
            "idx": idx,
            "mag": MAG_DICT[mag_choice],
            "img_path": str(folder_path / f"{idx}.jpg"),
        }
