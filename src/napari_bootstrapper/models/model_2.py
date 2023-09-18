import os
import json
import torch
from funlib.learn.torch.models import UNet, ConvPass

class Model_2(torch.nn.Module):

    def __init__(
            self):

        super().__init__()

        ds_fact = [(1, 2, 2), (1, 2, 2)]

        ksd = [
            [(3, 3, 3), (3, 3, 3)],
            [(1, 3, 3), (1, 3, 3)],
            [(1, 3, 3), (1, 3, 3)],
        ]

        ksu = [
            [(1, 3, 3), (1, 3, 3)],
            [(3, 3, 3), (3, 3, 3)],
        ]
        
        self.unet = UNet3d(
            in_channels=6,
            num_fmaps=12,
            fmap_inc_factor=5,
            downsample_factors=ds_fact,
            kernel_size_down=ksd,
            kernel_size_up=ksu,
            padding="valid",
            constant_upsample=True,
        )

        self.affs_head = ConvPass(num_fmaps, 3, [[1, 1, 1]], activation='Sigmoid')

    def forward(self, input):

        z = self.unet(input)
        affs = self.affs_head(z)

        return affs


class WeightedMSELoss(torch.nn.Module):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, pred, target, weights):

        scale = (weights * (pred - target) ** 2)

        if len(torch.nonzero(scale)) != 0:

            mask = torch.masked_select(scale, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:

            loss = torch.mean(scale)

        return loss
