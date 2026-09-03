from .idmvae_polyMNIST import PolyMNIST_5modalities as IDMVAE_PolyMNIST_5modalities
from .idmvae_CUB import CUB_Image_Captions as IDMVAE_CUB_Image_Captions
from .idmvae_UCF import UCF_Image_Captions as IDMVAE_UCF_Image_Captions

from .idmvae_CelebAMask import CelebA_IDMVAE
from .vae_polyMNIST_single_modality import PolyMNIST  # setup for mmvae baseline

__all__ = [
    "IDMVAE_PolyMNIST_5modalities",
    "IDMVAE_CUB_Image_Captions",
    "IDMVAE_UCF_Image_Captions",
    "CelebA_IDMVAE",
    "PolyMNIST",
]
