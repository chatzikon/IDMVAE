import torch
import torch.nn.functional as F
from .idmvae import IDMVAE
from .vae_CelebA_modality import CelebA_Image, CelebA_Mask, CelebA_Attr
from utils import ContrastiveLoss
from eval_functions import (
    setup_pretrained_denoiser,
    has_pretrained_denoiser,
    run_pretrained_denoiser,
)

class CelebA_IDMVAE(IDMVAE):
    """
    IDMVAE subclass for CelebA Mask Experiment (Image, Mask, Attributes)
    """
    def __init__(self, params):
        super(CelebA_IDMVAE, self).__init__(params, CelebA_Image, CelebA_Mask, CelebA_Attr)

        self.modelName = 'IDMVAE_CelebA'
        self.eps = 1e-6

        self.mi_shared = [ContrastiveLoss(tau=1, normalize=True) for _ in self.vaes]
        self.mi_private = [ContrastiveLoss(tau=1, normalize=True) for _ in self.vaes]
        self.contrast_mi = [ContrastiveLoss(tau=1, normalize=True) for _ in self.vaes]
        self.params = params
        denoiser_device = next(self.pretrained_vae.parameters()).device
        (
            self.denoiser_model,
            self.denoiser_diffusion,
            self.denoiser_device,
            self.denoiser_condition_label,
        ) = setup_pretrained_denoiser(self.params, denoiser_device)

        self.enable_denoiser_outputs = self._has_denoiser()

    def _has_denoiser(self):
        return has_pretrained_denoiser(self.denoiser_model, self.denoiser_diffusion)

    def _run_denoiser(self, noisy_latents):
        return run_pretrained_denoiser(
            self.denoiser_model,
            self.denoiser_diffusion,
            self.denoiser_device,
            self.denoiser_condition_label,
            noisy_latents,
        )

    def _encode_pixels_to_latents(self, pixel_batch):
        if pixel_batch.dim() != 4 or pixel_batch.size(1) not in (1, 3):
            return None
        if pixel_batch.size(1) == 1:
            pixel_batch = pixel_batch.repeat(1, 3, 1, 1)
        device = next(self.pretrained_vae.parameters()).device
        x = pixel_batch.to(device)
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = x * 2.0 - 1.0
        with torch.no_grad():
            enc = self.pretrained_vae.encode(x).latent_dist
            latents = enc.sample() * 0.18215
        return latents.cpu()

    def _denoise_recon_matrix(self, recons):
        if not self._has_denoiser():
            return None
        num_modalities = len(recons)
        denoised = [[None for _ in range(num_modalities)] for _ in range(num_modalities)]
        for r, row in enumerate(recons):
            for o, tensor in enumerate(row):
                if tensor is None or o not in (0, 1):
                    continue
                raw = tensor.squeeze(0)
                if raw.dim() != 4:
                    continue
                c = raw.size(1)
                out_h, out_w = raw.size(2), raw.size(3)
                if c == 4:
                    refined = self._run_denoiser(raw)
                    if refined is not None:
                        denoised[r][o] = refined.unsqueeze(0)
                    continue
                if c in (1, 3):
                    raw_3ch = raw.repeat(1, 3, 1, 1) if c == 1 else raw
                    latents = self._encode_pixels_to_latents(raw_3ch)
                    if latents is None:
                        continue
                    refined = self._run_denoiser(latents)
                    if refined is None:
                        continue
                    pixels = self._decode_latents_to_pixels(refined).cpu()
                    if out_h != 256 or out_w != 256:
                        pixels = F.interpolate(pixels, size=(out_h, out_w), mode='bilinear', align_corners=False)
                    if c == 1:
                        pixels = pixels.mean(dim=1, keepdim=True)
                    denoised[r][o] = pixels.unsqueeze(0)
        return denoised
