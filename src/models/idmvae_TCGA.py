import torch
from .idmvae import IDMVAE
from .vae_TCGA import TCGA
from utils import ContrastiveLoss, Constants

# Numerical epsilon for priors
eps = Constants.eta


class IDMVAE_TCGA(IDMVAE):
    """
    IDMVAE subclass for TCGA multi-omics experiment
    Modalities: mRNAseq, RPPA, Methylation, miRNAseq
    """

    def __init__(self, params):
        super(IDMVAE_TCGA, self).__init__(params, TCGA, TCGA)

        # Model identifiers
        self.modelName = "IDMVAE_TCGA"
        self.eps = eps
        self.params = params


        # One MI estimator per modality for shared vs private
        self.mi_shared = [ContrastiveLoss(tau=1, normalize=True) for _ in self.vaes]
        self.mi_private = [ContrastiveLoss(tau=1, normalize=True) for _ in self.vaes]
        # One MI estimator per modality for CrossMI loss
        self.contrast_mi = [ContrastiveLoss(tau=1, normalize=True) for _ in self.vaes]
