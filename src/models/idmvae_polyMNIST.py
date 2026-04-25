# PolyMNIST experiment IDMVAE model specifications
from .idmvae import IDMVAE
from .vae_polyMNIST_single_modality import PolyMNIST
from utils import ContrastiveLoss

from utils import Constants
eps = Constants.eta


class PolyMNIST_5modalities(IDMVAE):
    """
    IDMVAE subclass for PolyMNIST Experiment
    """
    def __init__(self, params):
        super(PolyMNIST_5modalities, self).__init__(params, PolyMNIST, PolyMNIST, PolyMNIST, PolyMNIST, PolyMNIST)

        self.modelName = 'IDMVAE_PolyMNIST'
        print("PolyMNIST data version: quadrant_pt")

        for idx, vae in enumerate(self.vaes):
            vae.modelName = 'VAE_PolyMNIST_m' + str(idx)

        self.mi_shared = [ContrastiveLoss(tau=1, normalize=True) for _ in self.vaes]
        self.mi_private = [ContrastiveLoss(tau=1, normalize=True) for _ in self.vaes]
        self.contrast_mi = [ContrastiveLoss(tau=1, normalize=True) for _ in self.vaes]

        self.params = params
        self.eps = eps
