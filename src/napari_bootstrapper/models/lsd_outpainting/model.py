from .unet import UNet2d, ConvPass
import torch


class LsdModel(torch.nn.Module):

    def __init__(
            self,
            output_shapes):

        super().__init__()

        num_fmaps = sum(output_shapes)

        if num_fmaps < 5:
            num_fmaps = 5

        ds_fact = [(2, 2), (2, 2), (2, 2)]
        num_levels = len(ds_fact) + 1
        ksd = [[(3, 3), (3, 3)]] * num_levels
        ksu = [[(3, 3), (3, 3)]] * (num_levels - 1)

        self.unet = UNet2d(
            in_channels=1,
            num_fmaps=num_fmaps,
            fmap_inc_factor=5,
            downsample_factors=ds_fact,
            kernel_size_down=ksd,
            kernel_size_up=ksu,
            padding="valid",
            constant_upsample=True,
        )

        self.lsd_head = ConvPass(num_fmaps, output_shapes[0], [[1, 1]], activation='Sigmoid')

    def forward(self, input):

        z = self.unet(input)

        lsds = self.lsd_head(z)

        return lsds


class WeightedMSELoss(torch.nn.Module):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def _calc_loss(self, pred, target, weights):

        scale = (weights * (pred - target) ** 2)

        if len(torch.nonzero(scale)) != 0:

            mask = torch.masked_select(scale, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:

            loss = torch.mean(scale)

        return loss

    def forward(
            self,
            lsds_prediction,
            lsds_target,
            lsds_weights):

        lsds_loss = self._calc_loss(lsds_prediction, lsds_target, lsds_weights)

        return lsds_loss
