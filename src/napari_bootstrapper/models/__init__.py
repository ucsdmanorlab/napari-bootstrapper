import torch

from .model_2d import Model as Model2D
from .model_3d import AffsUNet

PRETRAINED_3D_MODEL_URLS = {
    "3d_affs_from_2d_affs": "https://github.com/ucsdmanorlab/bootstrapper/releases/download/v0.3.2/3d_affs_from_2d_affs.zip",
    "3d_affs_from_2d_lsd": "https://github.com/ucsdmanorlab/bootstrapper/releases/download/v0.3.2/3d_affs_from_2d_lsd.zip",
    "3d_affs_from_2d_mtlsd": "https://github.com/ucsdmanorlab/bootstrapper/releases/download/v0.3.2/3d_affs_from_2d_mtlsd.zip",
}

DEFAULT_2D_MODEL_CONFIG = {
    "batch_size": 8,
    "learning_rate": 1e-4,
    "max_iterations": 5000,
    "num_workers": 8,
    "save_snapshots_every": 1000,
    "net": {
        "adj_slices": 3,
        "num_fmaps": 12,
        "num_fmaps_out": 12,
        "fmap_inc_factor": 5,
        "input_shape": [196, 196],
        "output_shape": [104, 104],
        "shape_increase": [120, 120],
    },
    "task": {
        "lsd_sigma": 10,
        "lsd_downsample": 2,
        "aff_neighborhood": [
            [-1, 0],
            [0, -1],
            [-9, 0],
            [0, -9],
            [-27, 0],
            [0, -27],
        ],
        "aff_grow_boundary": 0,
    },
}

DEFAULT_3D_MODEL_CONFIG = {
    "batch_size": 1,
    "learning_rate": 0.5e-4,
    "max_iterations": 10000,
    "num_workers": 8,
    "save_snapshots_every": 1000,
    "net": {
        "num_fmaps": 9,
        "num_fmaps_out": 18,
        "fmap_inc_factor": 3,
        "input_shape": [24, 148, 148],
        "output_shape": [4, 56, 56],
        "shape_increase": [12, 120, 120],
    },
    "task": {
        "lsd_sigma": 10,
        "lsd_downsample": 2,
        "in_aff_neighborhood": [
            [-1, 0],
            [0, -1],
            [-9, 0],
            [0, -9],
            [-27, 0],
            [0, -27],
        ],
        "aff_grow_boundary": 1,
        "aff_neighborhood": [
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
            [-2, 0, 0],
            [0, -9, 0],
            [0, 0, -9],
            [-3, 0, 0],
            [0, -27, 0],
            [0, 0, -27],
        ],
    },
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
    num_fmaps=12,
    num_fmaps_out=12,
    fmap_inc_factor=5,
    **kwargs,
):
    return Model2D(
        in_channels=num_channels,
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        model_type=model_type,
        num_fmaps_out=num_fmaps_out,
        **kwargs,
    )


def get_3d_model(
    model_type,
    num_channels,
    num_fmaps=12,
    num_fmaps_out=18,
    fmap_inc_factor=2,
    **kwargs,
):
    return AffsUNet(
        in_channels=num_channels,
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        model_type=model_type,
        num_fmaps_out=num_fmaps_out,
        **kwargs,
    )
