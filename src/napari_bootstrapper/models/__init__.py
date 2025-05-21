import torch

from .model_2d import Model as Model2D
from .model_3d import AffsUNet


PRETRAINED_3D_MODEL_URLS = {
    "3d_affs_from_2d_affs": "https://github.com/ucsdmanorlab/bootstrapper/releases/download/v0.3.0/3d_affs_from_2d_affs.zip",
    "3d_affs_from_2d_lsd": "https://github.com/ucsdmanorlab/bootstrapper/releases/download/v0.3.0/3d_affs_from_2d_lsd.zip",
    "3d_affs_from_2d_mtlsd": "https://github.com/ucsdmanorlab/bootstrapper/releases/download/v0.3.0/3d_affs_from_2d_mtlsd.zip",
    "3d_affs_from_3d_lsd": "https://github.com/ucsdmanorlab/bootstrapper/releases/download/v0.3.0/3d_affs_from_3d_lsd.zip",
}


class WeightedMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def _calc_loss(self, pred, target, weights):
        scale = weights * (pred - target) ** 2

        if len(torch.nonzero(scale)) != 0:
            mask = torch.masked_select(scale, torch.gt(weights, 0))
            loss = torch.mean(mask)
        else:
            loss = torch.mean(scale)

        return loss


class LsdLoss(WeightedMSELoss):
    def forward(self, lsds_prediction, lsds_target, lsds_weights):
        return self._calc_loss(lsds_prediction, lsds_target, lsds_weights)


class AffsLoss(WeightedMSELoss):
    def forward(self, affs_prediction, affs_target, affs_weights):
        return self._calc_loss(affs_prediction, affs_target, affs_weights)


class MtlsdLoss(WeightedMSELoss):
    def forward(
        self,
        lsds_prediction,
        lsds_target,
        lsds_weights,
        affs_prediction,
        affs_target,
        affs_weights,
    ):
        lsds_loss = self._calc_loss(lsds_prediction, lsds_target, lsds_weights)
        affs_loss = self._calc_loss(affs_prediction, affs_target, affs_weights)
        return lsds_loss + affs_loss


def get_loss(model_type):
    if model_type == "2d_lsd":
        return LsdLoss()
    elif model_type == "2d_affs":
        return AffsLoss()
    elif model_type == "2d_mtlsd":
        return MtlsdLoss()
    elif "3d_affs_from_" in model_type:
        return AffsLoss()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_2d_model(
    model_type,
    num_channels,
    stack_infer=True,
    num_fmaps=12,
    fmap_inc_factor=5,
):
    return Model2D(
        in_channels=num_channels,
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        model_type=model_type,
    )


def get_3d_model(
    model_type,
    num_channels,
    num_fmaps=8,
    fmap_inc_factor=3,
):
    return AffsUNet(
        in_channels=num_channels,
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        model_type=model_type,
    )
