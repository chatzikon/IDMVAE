import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import Constants

eps = Constants.eta

def actvn(x):
    out = torch.nn.functional.leaky_relu(x, 2e-1)
    return out


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


# Classes
class Enc(nn.Module):
    """ Encoder Image PolyMNIST Resnet """

    def __init__(self, ndim_w, ndim_z, dist='Normal', size=64, s0=8, nf=64):  # size=28, s0=7
        super().__init__()
        self.dist = dist
        self.nf = nf  # nfilter
        nf_max = self.nf_max = 512  # nfilter_max
        self.s0 = s0

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks_w = [
            ResnetBlock(nf, nf)
        ]

        blocks_z = [
            ResnetBlock(nf, nf)
        ]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks_w += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]
            blocks_z += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img_w = nn.Conv2d(3, 1*nf, 3, padding=1)
        self.resnet_w = nn.Sequential(*blocks_w)
        self.fc_mu_w = nn.Linear(self.nf0*s0*s0, ndim_w)
        self.fc_lv_w = nn.Linear(self.nf0*s0*s0, ndim_w)

        self.conv_img_z = nn.Conv2d(3, 1 * nf, 3, padding=1)
        self.resnet_z = nn.Sequential(*blocks_z)
        self.fc_mu_z = nn.Linear(self.nf0 * s0 * s0, ndim_z)
        self.fc_lv_z = nn.Linear(self.nf0 * s0 * s0, ndim_z)

    def forward(self, x):
        out_w = self.conv_img_w(x)
        out_w = self.resnet_w(out_w)
        out_w = out_w.view(out_w.size()[0], self.nf0*self.s0*self.s0)
        lv_w = self.fc_lv_w(out_w) # Shape: [batch_size, ndim_w]

        out_z = self.conv_img_z(x)
        out_z = self.resnet_z(out_z)
        out_z = out_z.view(out_z.size()[0], self.nf0 * self.s0 * self.s0)
        lv_z = self.fc_lv_z(out_z)

        if self.dist == 'Normal':
            return torch.cat((self.fc_mu_w(out_w), self.fc_mu_z(out_z)), dim=-1), \
                   torch.cat((F.softplus(lv_w) + eps,
                              F.softplus(lv_z) + eps), dim=-1)
        else:
            return torch.cat((self.fc_mu_w(out_w), self.fc_mu_z(out_z)), dim=-1), \
                   torch.cat((F.softmax(lv_w, dim=-1) * lv_w.size(-1) + eps,
                              F.softmax(lv_z, dim=-1) * lv_z.size(-1) + eps), dim=-1)


class Dec(nn.Module):
    """ Decoder Image PolyMNIST Resnet """

    def __init__(self, ndim, dist='Laplace', size=64, s0=8, nf=64): # size=28, s0=7
        super().__init__()
        self.dist = dist

        # NOTE: I've set below variables according to Kieran's suggestions
        self.nf = nf  # nfilter
        nf_max = self.nf_max = 512  # nfilter_max
        self.s0 = s0

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        self.fc = nn.Linear(ndim, self.nf0*s0*s0)

        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2**(nlayers-i), nf_max)
            nf1 = min(nf * 2**(nlayers-i-1), nf_max)
            blocks += [
                ResnetBlock(nf0, nf1),
                nn.Upsample(scale_factor=2)
            ]

        blocks += [
            ResnetBlock(nf, nf),
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img_mean = nn.Conv2d(nf, 3, 3, padding=1)
        self.conv_img_scale = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, u): # (K, B, W+Z)
        out = self.fc(u).view(-1, self.nf0, self.s0, self.s0) # (K*B, nf_max, s0, s0)
        out = self.resnet(out) # (K*B, nf, H, V)
        out_mean = self.conv_img_mean(actvn(out)) # (K*B, C, H, V)
        out_mean = out_mean.view(*u.size()[:2], *out_mean.size()[1:]) # (K, B, C, H, V)

        if self.dist == 'Normal':
            return F.sigmoid(out_mean), torch.tensor(0.75).to(u.device)  # F.softplus(out_scale) + eps
        else:
            return F.sigmoid(out_mean), torch.tensor(0.75).to(u.device)  # mean, length scale
