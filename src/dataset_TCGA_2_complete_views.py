# dataset_TCGA.py

import numpy as np
import torch


class TCGA2CompleteViews(torch.utils.data.Dataset):

    def __init__(self, npz_path: str):
        """
        Args:
          - npz_path: path to a .npz file that contains at least:
               * "mRNAseq"        : (N_samples, 100) array
               * "RPPA"           : (N_samples, 100) array
               * "Methylation"    : (N_samples, 100) array
               * "miRNAseq"       : (N_samples, 100) array
               * "mask"           : (N_samples, 4) array of 0/1 floats
               * "label"          : (N_samples,)   array of floats in {0,1,-1}
                                   (here we assume it’s already only the 1-year column)
        """
        raw = np.load(npz_path)

        # Load all four modalities as (N, 100), cast to float32
        mRNA = raw["view0"].astype(np.float32)  # shape (N, 100)
        # RPPA = raw["RPPA"].astype(np.float32)  # shape (N, 100)
        # Methylation = raw["Methylation"].astype(np.float32)
        miRNA = raw["view3"].astype(np.float32)

        # Stack them into a single array of shape (N, 2, 100)
        #   modalities[b, v, :] = b-th sample’s v-th view (100 dims)
        modalities = np.stack([mRNA, miRNA], axis=1)

        # Load the 1-year mortality label: assume raw["label"] is already shape (N,)
        labels_full = raw["label"].astype(np.float32)  # shape (N,)
        self.X = modalities  # shape (N_valid, 4, 100)
        self.y = labels_full  # shape (N_valid,)

        # Reshape y → (N_valid, 1) so that later we return a (1,) tensor
        self.y = self.y.reshape(-1, 1)  # shape (N_valid, 1)

        # Double-check our shapes line up
        assert self.X.shape[0] == self.y.shape[0], "Mismatch between X and y lengths"
        self.N = self.X.shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x_all = self.X[idx].copy()  # shape (4, 100)
        y = self.y[idx]  # shape (1,)

        # Convert each of the 2 “rows” into a separate torch.FloatTensor(100,)
        views_list = [torch.from_numpy(x_all[v]) for v in range(2)]
        label_tensor = torch.from_numpy(y)  # shape (1,)

        return views_list, label_tensor