# models/vae_TCGA.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from models.base_vae import VAE

_EPS = 1e-6


class Enc(nn.Module):
    """Simple MLP encoder for tabular TCGA views.

    Produces concatenated (mu, lv) for w and z latents to match the base VAE API
    used by the PolyMNIST encoders.
    """

    def __init__(self, input_dim: int, ndim_w: int, ndim_z: int, dist: str = 'Normal'):
        super().__init__()
        self.dist = dist
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)

        # Separate heads for w and z parts
        self.fc_mu_w = nn.Linear(64, ndim_w)
        self.fc_lv_w = nn.Linear(64, ndim_w)
        self.fc_mu_z = nn.Linear(64, ndim_z)
        self.fc_lv_z = nn.Linear(64, ndim_z)

    def forward(self, x):
        # x: (B, D)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))

        mu_w = self.fc_mu_w(h)
        lv_w = self.fc_lv_w(h)
        mu_z = self.fc_mu_z(h)
        lv_z = self.fc_lv_z(h)

        if self.dist == 'Normal':
            lv_w = F.softplus(lv_w) + _EPS
            lv_z = F.softplus(lv_z) + _EPS
        else:
            # For Laplace parameterization, follow PolyMNIST convention: scale > 0
            lv_w = F.softmax(lv_w, dim=-1) * lv_w.size(-1) + _EPS
            lv_z = F.softmax(lv_z, dim=-1) * lv_z.size(-1) + _EPS

        mu = torch.cat([mu_w, mu_z], dim=-1)
        lv = torch.cat([lv_w, lv_z], dim=-1)
        return mu, lv


class Dec(nn.Module):
    """Simple MLP decoder for tabular TCGA views.

    Takes the joint latent u = [w, z] and outputs (mean, scale/logvar) for the
    reconstruction distribution over the tabular view.
    """

    def __init__(self, ndim: int, dist: str = 'Normal', output_dim: int = 100):
        super().__init__()
        self.dist = dist
        self.fc1 = nn.Linear(ndim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, output_dim)
        self.fc_lv = nn.Linear(128, output_dim)

    def forward(self, u):
        h = F.relu(self.fc1(u))
        h = F.relu(self.fc2(h))
        mean = self.fc_mu(h)
        lv = self.fc_lv(h)

        if self.dist == 'Normal':
            lv = F.softplus(lv) + _EPS
        else:
            lv = F.softplus(lv) + _EPS  # treat as positive scale for Laplace as well

        return mean, lv


class TCGA(VAE):
    """Two-view TCGA VAE using MLP enc/dec (no ResNet, no image sizes).

    Expects params to provide:
      - input_dim: dimensionality of this view (e.g., 100)
      - latent_dim_w, latent_dim_z, latent_dim_u
      - priorposterior in {'Normal','Laplace'}
      - likelihood in {'Normal','Laplace'}
    """

    def __init__(self, params):
        super(TCGA, self).__init__(
            prior_dist=dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,
            likelihood_dist=dist.Normal if params.likelihood == 'Normal' else dist.Laplace,
            post_dist=dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,
            enc=Enc(
                input_dim=params.input_dim,
                ndim_w=params.latent_dim_w,
                ndim_z=params.latent_dim_z,
                dist=params.priorposterior,
            ),
            dec=Dec(
                ndim=params.latent_dim_u,
                dist=params.likelihood,
                output_dim=params.input_dim,
            ),
            params=params,
        )