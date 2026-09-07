import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import SiglipVisionModel

from utils import Constants


eps = Constants.eta


class MLPAttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens):
        # tokens: [B, N_patches, D]

        weights = self.score(tokens).squeeze(-1)  # [B,N]
        weights = torch.softmax(weights, dim=1)

        pooled = torch.sum(
            tokens * weights.unsqueeze(-1),
            dim=1,
        )

        return self.norm(pooled)


class SigLIPEncoderImg(nn.Module):

    def __init__(
        self,
        ndim_w,
        ndim_z,
        dist,
        model_name="google/siglip-base-patch16-256",
    ):
        super().__init__()

        self.dist = dist

        # Vision tower only.
        self.backbone = SiglipVisionModel.from_pretrained(
            model_name
        )

        hidden_dim = self.backbone.config.hidden_size
        self.image_size = self.backbone.config.image_size

        # Different pooling for private/shared latent.
        self.pool_w = MLPAttentionPool(hidden_dim)
        self.pool_z = MLPAttentionPool(hidden_dim)

        # Posterior parameters for w
        self.fc_mu_w = nn.Linear(hidden_dim, ndim_w)
        self.fc_scale_w = nn.Linear(hidden_dim, ndim_w)

        # Posterior parameters for z
        self.fc_mu_z = nn.Linear(hidden_dim, ndim_z)
        self.fc_scale_z = nn.Linear(hidden_dim, ndim_z)

    def preprocess(self, x):

        if x.ndim != 4 or x.size(1) != 3:
            raise ValueError(
                f"SigLIP expects [B,3,H,W], got {x.shape}"
            )

        x = x.float()

        if x.shape[-2:] != (
            self.image_size,
            self.image_size,
        ):
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )

        # Input dataset: [0,1]
        # SigLIP normalization: mean=.5 std=.5
        x = (x - 0.5) / 0.5

        return x

    def forward(self, x):

        x = self.preprocess(x)

        output = self.backbone(
            pixel_values=x,
            return_dict=True,
        )

        # [B, N_patches, hidden_dim]
        tokens = output.last_hidden_state

        # Separate learned pooling
        h_w = self.pool_w(tokens)
        h_z = self.pool_z(tokens)

        mu_w = self.fc_mu_w(h_w)
        mu_z = self.fc_mu_z(h_z)

        raw_scale_w = self.fc_scale_w(h_w)
        raw_scale_z = self.fc_scale_z(h_z)

        if self.dist == "Normal":

            scale_w = (
                F.softplus(raw_scale_w) + eps
            )

            scale_z = (
                F.softplus(raw_scale_z) + eps
            )

        else:

            scale_w = (
                F.softmax(raw_scale_w, dim=-1)
                * raw_scale_w.size(-1)
                + eps
            )

            scale_z = (
                F.softmax(raw_scale_z, dim=-1)
                * raw_scale_z.size(-1)
                + eps
            )

        # Exactly the same interface as old EncoderImg.
        return (
            torch.cat([mu_w, mu_z], dim=-1),
            torch.cat([scale_w, scale_z], dim=-1),
        )