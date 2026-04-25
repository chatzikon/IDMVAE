# PolyMNIST unimodal VAE model specification
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import save_image, make_grid
from .base_vae import VAE
from .encoder_decoder_blocks.resnet_polyMNIST import Enc, Dec

IMG_SIZE = 64
RESNET_S0 = 8
RESNET_NF = 64

class PolyMNIST(VAE):
    """ Unimodal VAE subclass for PolyMNIST experiment """

    def __init__(self, params):
        super(PolyMNIST, self).__init__(
            prior_dist=dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,
            likelihood_dist=dist.Normal if params.likelihood == 'Normal' else dist.Laplace,
            post_dist=dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,
            enc=Enc(params.latent_dim_w, params.latent_dim_z, dist=params.priorposterior, size=IMG_SIZE, s0=RESNET_S0, nf=RESNET_NF),
            dec=Dec(params.latent_dim_u, dist=params.likelihood, size=IMG_SIZE, s0=RESNET_S0, nf=RESNET_NF),
            params=params
        )

        self.modelName = 'polymnist-split'
        self.llik_scaling = 1.
        self.datadir = params.datadir
        self.params = params

    def generate_unconditional_random_to_tensor(self, N):
        """
        Unconditional random generation.
        Returns:
                Tensor of unconditional random generations.
        """
        samples = super(PolyMNIST, self).generate_unconditional_random(N)
        samples = samples.data.cpu()

        return make_grid(samples, nrow=N)

    def generate_unconditional_to_tensor(self, N):
        """
        Unconditional generation.
        Returns:
                Tensor of unconditional generations.
        """
        samples = super(PolyMNIST, self).generate_unconditional(N)
        samples = samples.data.cpu()

        return make_grid(samples, nrow=N)

    def generate_unconditional_random_for_fid_calculation(self, savePath, num_samples, tranche):
        """
        Unconditional random generation for FID calculation. (Split in tranches for memory issues)
        Args:
            savePath: Path of directory where to save images
            num_samples: Num_samples to generate
            tranche: Tranche of images currently generated

        """
        N = num_samples
        samples = super(PolyMNIST, self).generate_unconditional_random(N)
        samples = samples.data.cpu()
        for image in range(samples.size(0)):
            save_image(samples[image, :, :, :], '{}/random/m{}/{}_{}.png'.format(savePath, self.modal, tranche, image))

    def self_and_cross_modal_reconstruct_for_fid(self, data, savePath, i):
        """
        Conditional generation for FID calculation.
        Args:
            data: input
            savePath: Path of directory where to save images
            i: index naming

        """
        recon = super(PolyMNIST, self).reconstruct(data)
        recon = recon.squeeze(0).cpu()
        for image in range(recon.size(0)):
            save_image(recon[image, :, :, :], '{}/m{}/m{}/{}_{}.png'.format(savePath, self.modal,self.modal, image, i))

    def reconstruct_to_tensor(self, data, N=10):
        """
        Test-time reconstruction.
        Returns:
                Tensor of reconstructions.
        """
        recon = super(PolyMNIST, self).reconstruct(data[:N])
        recon = recon.squeeze(0)
        comp = torch.cat([data[:N], recon]).data.cpu()
        return make_grid(comp, nrow=N)
