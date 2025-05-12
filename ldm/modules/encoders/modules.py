import torch
import torch.nn as nn
import torch.nn.functional as F


from torchvision.models.vision_transformer import Encoder

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):
    def encode(self, x):
        return x


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key="class", ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes+1, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0.0 and not disable_dropout:
            mask = 1.0 - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1 - mask) * torch.ones_like(c) * (self.n_classes - 1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc}
        return uc


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class EmbeddingViT2(nn.Module):

    """
    1. more transformer blocks
    2. correct padding : non zero embeddings at the center, instead of beginning
    3. classifier guidance null token replacement AFTER transformer instead of before
    """

    def __init__(
        self,
        feat_key="feat",
        mag_key="mag",
        input_channels=1024,
        hidden_channels=512,
        vit_mlp_dim=2048,
        output_channels=512,
        seq_length=64,
        mag_levels=8,
        num_layers=12,
        num_heads=8,
        p_uncond=0,
        ckpt_path=None,
        ignore_keys=[],
    ):
        super(EmbeddingViT2, self).__init__()

        self.mag_embedding = nn.Embedding(mag_levels, hidden_channels)
        self.feat_key = feat_key
        self.mag_key = mag_key
        self.hidden_channels = hidden_channels

        self.dim_reduce = nn.Linear(input_channels, hidden_channels)

        self.pad_token = nn.Parameter(torch.randn(1, 1, hidden_channels))
        self.encoder = Encoder(
            seq_length=seq_length + 1,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_channels,
            mlp_dim=vit_mlp_dim,
            dropout=0,
            attention_dropout=0,
        )
        self.final_proj = nn.Linear(hidden_channels, output_channels)
        self.p_uncond = p_uncond

        # if ckpt_path is not None:
        #     self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def forward(self, batch):
        x = batch[self.feat_key]
        int_mag = batch[self.mag_key]

        # Process inputs
        x = self.process_input_batch(x)  # Shape: [batch_size, 64, hidden_channels]

        mag_embed = self.mag_embedding(int_mag).unsqueeze(1)  # Shape: [batch_size, 1, hidden_channels]
        x = torch.cat((mag_embed, x), dim=1)  # Shape: [batch_size, 65, hidden_channels]

        x = self.encoder(x)

        x = self.final_proj(x)  # Shape: [batch_size, 65, output_channels]

        return x

    def process_input_batch(self, x):
        if isinstance(x, torch.Tensor):
            x = list(x)
        if isinstance(x, list):
            return torch.stack([self.process_single_input(item) for item in x])
        else:
            return self.process_single_input(x).unsqueeze(0)

    def process_single_input(self, x):
        # Ensure x is 3D: [channels, height, width]
        if x.dim() == 2:
            x = x.unsqueeze(0)

        c, h, w = x.shape

        n = h * w

        x = x.view(c, -1).transpose(0, 1)
        x = self.dim_reduce(x)

        if h == w == 1:
            # center the token
            mask = torch.ones(64, device=x.device)
            mask[32] = 0

        elif h < 8 or w < 8:
            # pad x to 64 tokens, keep the original tokens at the center

            x = F.pad(x, (0, 0, 32 - n // 2, 32 - n // 2))
            mask = torch.ones(64, device=x.device)
            mask[32 - n // 2 : 32 + n // 2] = 0

        else:
            # we used avg pooling in the dataloader
            return x

        x = x * (1 - mask.unsqueeze(1)) + self.pad_token * mask.unsqueeze(1)
        return x.squeeze()  # Return as [64, hidden_channels]

    def encode(self, batch):
        c = self.forward(batch)
        # replace features with zeros with probability p_uncond
        if self.p_uncond > 0.0:
            mask = 1.0 - torch.bernoulli(torch.ones(len(c)) * self.p_uncond)
            mask = mask[:, None, None].to(c.device)
            c = mask * c
        return c



class EmbeddingViT2_5(EmbeddingViT2):

    """
    v2 but layer norm at the end
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        hidden_channels = kwargs.get("hidden_channels")

        self.layer_norm = nn.LayerNorm(hidden_channels)

    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]

        sd_cond_stage = {k.replace("cond_stage_model.", ""):v for k,v in sd.items() if "cond_stage_model" in k}

        self.load_state_dict(sd_cond_stage, strict=True)
        print(f"Restored from {path}")


    def forward(self, batch):
        x = batch[self.feat_key]
        int_mag = batch[self.mag_key]

        # Process inputs
        x = self.process_input_batch(x)  # Shape: [batch_size, 64, hidden_channels]

        mag_embed = self.mag_embedding(int_mag).unsqueeze(1)  # Shape: [batch_size, 1, hidden_channels]
        x = torch.cat((mag_embed, x), dim=1)  # Shape: [batch_size, 65, hidden_channels]

        x = self.encoder(x)

        x = self.final_proj(x)
        x = self.layer_norm(x)

        return x
