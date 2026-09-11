import numpy as np
import torch.nn as nn


class DecoderImg(nn.Module):
    """ Generate an image given a sample from the latent space. """

    def __init__(self, ndim, img_size=64, out_channels=3):
        super().__init__()

        s0 = self.s0 = 2
        nf = self.nf = 64
        nf_max = self.nf_max = 1024

        nlayers = int(np.log2(img_size / s0))

        self.nf0 = min(
            nf_max,
            nf * 2 ** nlayers
        )

        self.fc = nn.Linear(
            ndim,
            self.nf0 * s0 * s0
        )

        blocks = []

        for i in range(nlayers):

            nf0 = min(
                nf * 2 ** (nlayers - i),
                nf_max
            )

            nf1 = min(
                nf * 2 ** (nlayers - i - 1),
                nf_max
            )

            blocks += [
                ResnetBlock(nf0, nf1),
                nn.Upsample(scale_factor=2)
            ]

        blocks += [
            ResnetBlock(nf, nf),
        ]

        self.resnet = nn.Sequential(*blocks)

        self.conv_img = nn.Conv2d(
            nf,
            out_channels,
            3,
            padding=1
        )

    def forward(self, u):

        out = self.fc(u).view(
            -1,
            self.nf0,
            self.s0,
            self.s0
        )

        out = self.resnet(out)

        out = self.conv_img(
            actvn(out)
        )

        out = out.view(
            *u.size()[:2],
            *out.size()[1:]
        )

        # mean, length scale
        return (
            out,
            torch.tensor(0.01).to(u.device)
        )