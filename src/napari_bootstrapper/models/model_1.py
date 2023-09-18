import torch
from funlib.learn.torch.models import UNet, ConvPass

from pathlib import Path
from typing import List

import attrs
from attrs.validators import instance_of


def to_path(path):
    if path is None:
        return

    return Path(path)


@attrs.define
class ModelConfig2D:
    """Model configuration.

    Parameters:

        num_fmaps:

            The number of feature maps in the first level of the U-Net.

        fmap_inc_factor:

            The factor by which to increase the number of feature maps between
            levels of the U-Net.

        downsampling_factors:

            A list of downsampling factors, each given per dimension (e.g.,
            [[2,2], [3,3]] would correspond to two downsample layers, one with
            an isotropic factor of 2, and another one with 3). This parameter
            will also determine the number of levels in the U-Net.

        checkpoint (optional, default ``None``):

            A path to a checkpoint of the network. Needs to be set for networks
            that are used for prediction. If set during training, the
            checkpoint will be used to resume training, otherwise the network
            will be trained from scratch.


    """

    num_fmaps: int = attrs.field(validator=instance_of(int))
    fmap_inc_factor: int = attrs.field(validator=instance_of(int))
    downsampling_factors: List[List[int]] = attrs.field(
        default=[
            [2, 2],
            [2, 2],
            [2, 2],
        ]
    )
    checkpoint: Path = attrs.field(default=None, converter=to_path)


class Model2D(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            num_fmaps,
            fmap_inc_factor,
            downsampling_factors,
        ):
        super().__init__()

        num_levels = len(downsampling_factors) + 1
        ksd = [[(3, 3), (3, 3)]] * num_levels
        ksu = [[(3, 3), (3, 3)]] * (num_levels - 1)

        self.unet = UNet(
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsampling_factors,
            kernel_size_down=ksd,
            kernel_size_up=ksu,
            constant_upsample=True,
        )

        self.head = ConvPass(num_fmaps, out_channels, [[1, 1]], activation='Sigmoid')

    def forward(self, input):

        z = self.unet(input)

        pred = self.head(z)

        return pred


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
