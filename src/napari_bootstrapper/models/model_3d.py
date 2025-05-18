import torch

from .unet_3d import ConvPass as ConvPass3D
from .unet_3d import UNet as UNet3D


class AffsUNet(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        num_fmaps=8,
        fmap_inc_factor=3,
        downsample_factors=((1, 2, 2), (1, 2, 2), (1, 2, 2)),
        kernel_size_down=(
            ((1, 3, 3), (1, 3, 3)),
            ((2, 3, 3), (2, 3, 3)),
            ((3, 3, 3), (3, 3, 3)),
            ((3, 3, 3), (3, 3, 3)),
        ),
        kernel_size_up=(
            ((3, 3, 3), (3, 3, 3)),
            ((2, 3, 3), (2, 3, 3)),
            ((1, 3, 3), (1, 3, 3)),
        ),
        model_type="3d_affs_from_2d_mtlsd",
    ):

        super().__init__()

        self.model_type = model_type

        if model_type == "3d_affs_from_2d_mtlsd":
            assert in_channels == 8, "in_channels must be 8=(6+2) for 2d_mtlsd"
            self.process_inputs = lambda *inputs: torch.cat(inputs, dim=1)
        elif model_type == "3d_affs_from_2d_lsd":
            assert in_channels == 6, "in_channels must be 6 for 2d_lsd"
            self.process_inputs = lambda *inputs: inputs[0]
        elif model_type == "3d_affs_from_2d_affs":
            assert in_channels == 2, "in_channels must be 2 for 2d_affs"
            self.process_inputs = lambda *inputs: inputs[0]
        else:
            raise ValueError("Invalid model_type")

        self.unet = UNet3D(
            in_channels=in_channels,
            num_fmaps=num_fmaps,
            fmap_inc_factor=fmap_inc_factor,
            downsample_factors=downsample_factors,
            kernel_size_down=kernel_size_down,
            kernel_size_up=kernel_size_up,
            constant_upsample=True,
            padding="valid",
        )

        self.affs_head = ConvPass3D(
            num_fmaps, 6, [[1, 1, 1]], activation="Sigmoid"
        )

    def forward(self, *inputs):

        z = self.process_inputs(*inputs)
        z = self.unet(z)
        affs = self.affs_head(z)

        return affs
