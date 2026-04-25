
import sys
import os
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from .base_vae import VAE

# Add mmvaeplus_from_sbm to sys.path to allow imports from mmplus_model_cel
current_dir = os.path.dirname(os.path.abspath(__file__))
sbm_dir = os.path.join(current_dir, 'mmvaeplus_from_sbm')
if sbm_dir not in sys.path:
    sys.path.append(sbm_dir)

try:
    from mmplus_model_cel import CelebEncImg, CelebDecImg, CelebEncMask, CelebDecMask, CelebEncAtt, CelebDecAtt
except ImportError:
    # Fallback if the path manipulation doesn't work as expected (e.g. during development)
    from .mmvaeplus_from_sbm.mmplus_model_cel import CelebEncImg, CelebDecImg, CelebEncMask, CelebDecMask, CelebEncAtt, CelebDecAtt

class CelebA_Image(VAE):
    """ Unimodal VAE subclass for Image modality CelebA experiment """

    def __init__(self, params):
        super(CelebA_Image, self).__init__(
            prior_dist=dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,
            likelihood_dist=dist.Laplace,
            post_dist=dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,
            enc=CelebEncImg(params.latent_dim_w, params.latent_dim_z),
            dec=CelebDecImg(params.latent_dim_w + params.latent_dim_z),
            params=params
        )
        self.modelName = 'celeba_img'
        self.dataSize = torch.Size([3, 128, 128])
        self.llik_scaling = 1.
        self.params = params

class CelebA_Mask(VAE):
    """ Unimodal VAE subclass for Mask modality CelebA experiment """

    def __init__(self, params):
        super(CelebA_Mask, self).__init__(
            prior_dist=dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,
            likelihood_dist=dist.Laplace,
            post_dist=dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,
            enc=CelebEncMask(params.latent_dim_w, params.latent_dim_z),
            dec=CelebDecMask(params.latent_dim_w + params.latent_dim_z),
            params=params
        )
        self.modelName = 'celeba_mask'
        self.dataSize = torch.Size([1, 128, 128])
        self.llik_scaling = 1.
        self.params = params

class CelebA_Attr(VAE):
    """ Unimodal VAE subclass for Attribute modality CelebA experiment """

    def __init__(self, params):
        super(CelebA_Attr, self).__init__(
            prior_dist=dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,
            likelihood_dist=dist.Bernoulli, # Attributes are binary
            post_dist=dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,
            enc=CelebEncAtt(params.latent_dim_w, params.latent_dim_z),
            dec=CelebDecAtt(params.latent_dim_w + params.latent_dim_z),
            params=params
        )
        self.modelName = 'celeba_attr'
        self.dataSize = torch.Size([18]) # 18 selected attributes
        self.llik_scaling = 1.
        self.params = params
