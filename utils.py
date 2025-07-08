import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import defaultdict

def show(img_lis, names):
    fig, axs = plt.subplots(nrows=len(img_lis), ncols=1, squeeze=False, figsize=(20,4))
    for ax,name, img in zip(axs, names, img_lis):
        ax[0].imshow(np.asarray(img))
        ax[0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        ax[0].set_ylabel(name)
    fig.tight_layout()
    

def collate_fn(batch):
    outputs = defaultdict(list)
    images_processed = []
    ssl_features_processed = []
    ssl_features_unpooled_processed = []
    mags_processed = []

    for item in batch:
        pil_img = item['image'].convert("RGB")
        outputs['image'].append(torch.from_numpy(np.array(pil_img)))

        for k in ['ssl_feat','ssl_feat_unpooled', 'mag']:
            if k in item:
                outputs[k].append(torch.tensor(item[k]))

    for k,v in outputs.items():
        outputs[k] = torch.stack(v)

    return outputs