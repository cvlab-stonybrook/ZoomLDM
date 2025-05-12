from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import h5py
import io
import torch.nn.functional as F
import torch


MAG_DICT = {
    "20x": 0,
    "10x": 1,
    "5x": 2,
    "2_5x": 3,
    "1_25x": 4,
    "0_625x": 5,
    "0_3125x": 6,
    "0_15625x": 7,
}

MAG_NUM_IMGS = {
    "20x": 12_509_760,
    "10x": 3_036_288,
    "5x": 752_000,
    "2_5x": 187_280,
    "1_25x": 57_090,
    "0_625x": 20_679,
    "0_3125x": 7_923,
    "0_15625x": 2489,
}


class TCGADataset(Dataset):
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
        return MAG_NUM_IMGS["20x"]

    def __getitem__(self, idx):
        if self.mag:
            mag_choice = self.mag
        else:
            mag_choice = np.random.choice(self.keys)
            # pick a random index
            idx = np.random.randint(0, MAG_NUM_IMGS[mag_choice])

        ##### load VAE feat
        folder = str(idx // 1_000_000)
        folder_path = self.root / f"{mag_choice}/{folder}"

        try:
            vae_feat = np.load(folder_path / f"{idx}_vae.npy").astype(np.float16)
            if vae_feat.shape != (3, 64, 64):
                ### TEMPORARY FIX ###
                raise Exception(f"vae shape {vae_feat.shape} for idx {idx}")

        except:
            idx = np.random.randint(len(self))
            return self.__getitem__(idx)

        ###### load SSL feature
        ssl_feat = np.load(folder_path / f"{idx}_uni_grid.npy").astype(np.float16)

        if len(ssl_feat.shape) == 1:
            ssl_feat = ssl_feat[:, None]
        h = np.sqrt(ssl_feat.shape[1]).astype(int)

        ssl_feat = torch.tensor(ssl_feat.reshape((-1, h, h)))

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
            image = np.load(folder_path / f"{idx}_img.npy")
            image = Image.open(io.BytesIO(image))
            image = np.array(image).astype(np.uint8)

        else:
            image = np.ones((1, 1, 1, 3), dtype=np.float16)

        return {
            "image": image,
            "vae_feat": vae_feat,
            "ssl_feat": ssl_feat,
            "idx": idx,
            "mag": MAG_DICT[mag_choice],
        }
