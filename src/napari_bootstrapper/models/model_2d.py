import torch

from .unet import ConvPass as ConvPass2D
from .unet import UNet as UNet2D


class Model(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        stack_infer=True,
        num_fmaps=12,
        num_fmaps_out=12,
        fmap_inc_factor=5,
        downsample_factors=((2, 2), (2, 2), (2, 2)),
        kernel_size_down=None,
        kernel_size_up=None,
        model_type="2d_mtlsd",
        **kwargs,
    ):

        super().__init__()

        self.stack_infer = stack_infer
        self.model_type = model_type
        self.aff_nbhd = kwargs.get("aff_neighborhood", [[-1, 0], [0, -1]])

        num_levels = len(downsample_factors)
        if kernel_size_down is None:
            kernel_size_down = [[[3, 3], [3, 3]]] * (num_levels + 1)
        if kernel_size_up is None:
            kernel_size_up = [[[3, 3], [3, 3]]] * num_levels

        self.unet = UNet2D(
            in_channels=in_channels,
            num_fmaps=num_fmaps,
            fmap_inc_factor=fmap_inc_factor,
            downsample_factors=downsample_factors,
            kernel_size_down=kernel_size_down,
            kernel_size_up=kernel_size_up,
            constant_upsample=True,
            padding="valid",
            num_fmaps_out=num_fmaps_out,
        )

        # Create output heads based on model type
        self.has_lsd = model_type in ["2d_lsd", "2d_mtlsd"]
        self.has_aff = model_type in ["2d_affs", "2d_mtlsd"]

        if self.has_lsd:
            self.lsd_head = ConvPass2D(
                num_fmaps_out, 6, [[1, 1]], activation="Sigmoid"
            )

        if self.has_aff:
            self.aff_head = ConvPass2D(
                num_fmaps_out, len(self.aff_nbhd), [[1, 1]], activation="Sigmoid"
            )

    def forward(self, x):
        if len(x.size()) == 5:
            # reshape from (n,c,d,h,w) to (n,c*d,h,w)
            n, c, d, h, w = x.size()
            x = x.view(n, c * d, h, w)

        z = self.unet(x)

        outputs = []

        if self.has_lsd:
            lsds = self.lsd_head(z)
            if self.stack_infer:
                lsds = torch.unsqueeze(lsds, -3)
            outputs.append(lsds)

        if self.has_aff:
            affs = self.aff_head(z)
            if self.stack_infer:
                affs = torch.unsqueeze(affs, -3)
            outputs.append(affs)

        return tuple(outputs)
