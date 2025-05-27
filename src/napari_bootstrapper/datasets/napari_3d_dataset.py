import gunpowder as gp
import numpy as np
from torch.utils.data import IterableDataset

from ..gp.add_obfuscated_2d_lsds import AddObfuscated2DLSDs
from ..gp.calc_max_padding import calc_max_padding
from ..gp.create_labels import CreateLabels
from ..gp.custom_grow_boundary import CustomGrowBoundary
from ..gp.custom_intensity_augment import CustomIntensityAugment
from ..gp.obfuscate_affs import ObfuscateAffs
from ..gp.smooth_augment import SmoothAugment


class Napari3DDataset(IterableDataset):
    def __init__(
        self,
        model_type: str,
        input_shape: list[int],
        output_shape: list[int],
        lsd_sigma: int = 20,
        lsd_downsample: int = 4,
        in_aff_neighborhood: list[list[int]] | None = None,
        aff_grow_boundary: int = 1,
        aff_neighborhood: list[list[int]] | None = None,
    ):
        self.model_type = model_type
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.voxel_size = 1, 1, 1
        self.offset = 0, 0, 0

        self.lsd_sigma = lsd_sigma
        self.lsd_downsample = lsd_downsample
        self.in_neighborhood = (
            [[0, *x] for x in [[-1, 0], [0, -1]]]
            if in_aff_neighborhood is None
            else [[0, *x] for x in in_aff_neighborhood]
        )
        self.aff_grow_boundary = aff_grow_boundary
        self.aff_neighborhood = (
            [
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
                [-2, 0, 0],
                [0, -8, 0],
                [0, 0, -8],
            ]
            if aff_neighborhood is None
            else aff_neighborhood
        )

        self.voxel_size = gp.Coordinate(self.voxel_size)
        self.offset = gp.Coordinate(self.offset)
        self.context = calc_max_padding(
            self.output_shape, self.voxel_size, self.lsd_sigma
        )

        # get unet num_channels from model type
        if model_type == "3d_affs_from_2d_affs":
            self.num_channels = len(self.in_neighborhood)
        elif model_type == "3d_affs_from_2d_lsd":
            self.num_channels = 6
        elif model_type == "3d_affs_from_2d_mtlsd":
            self.num_channels = 6 + len(self.in_neighborhood)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.__setup_pipeline()

    def __setup_pipeline(self):
        self.labels = gp.ArrayKey("SYNTHETIC_LABELS")
        self.input_lsds = gp.ArrayKey("INPUT_2D_LSDS")
        self.input_affs = gp.ArrayKey("INPUT_2D_AFFS")
        self.gt_affs = gp.ArrayKey("GT_AFFS")
        self.pred_affs = gp.ArrayKey("PRED_AFFS")
        self.affs_weights = gp.ArrayKey("AFFS_WEIGHTS")

        self.pipeline = (
            CreateLabels(
                self.labels, shape=self.input_shape, voxel_size=self.voxel_size
            )
            + gp.Pad(self.labels, None, mode="reflect")
            + gp.DeformAugment(
                control_point_spacing=(20, 20),
                jitter_sigma=(3.0, 3.0),
                spatial_dims=2,
                subsample=1,
                scale_interval=(0.9, 1.1),
                graph_raster_voxel_size=self.voxel_size[1:],
                p=0.5,
            )
            + gp.ShiftAugment(prob_slip=0.2, prob_shift=0.2, sigma=5)
            + gp.SimpleAugment(transpose_only=[1, 2])
        )

        if (
            self.model_type == "3d_affs_from_2d_lsd"
            or self.model_type == "3d_affs_from_2d_mtlsd"
        ):
            self.pipeline += (
                AddObfuscated2DLSDs(
                    self.labels,
                    self.input_lsds,
                    sigma=(0, self.lsd_sigma, self.lsd_sigma),
                    downsample=self.lsd_downsample,
                )
                + gp.NoiseAugment(self.input_lsds, mode="gaussian", p=0.33)
                + CustomIntensityAugment(
                    self.input_lsds,
                    0.9,
                    1.1,
                    -0.1,
                    0.1,
                    z_section_wise=True,
                    p=0.5,
                )
                + SmoothAugment(self.input_lsds, p=0.5)
                + gp.DefectAugment(
                    self.input_lsds,
                    prob_missing=0.15,
                    prob_low_contrast=0.05,
                    prob_deform=0.0,
                    axis=1,
                )
            )
        if (
            self.model_type == "3d_affs_from_2d_affs"
            or self.model_type == "3d_affs_from_2d_mtlsd"
        ):
            self.pipeline += (
                CustomGrowBoundary(self.labels, only_xy=True)
                + gp.AddAffinities(
                    affinity_neighborhood=self.in_neighborhood,
                    labels=self.labels,
                    affinities=self.input_affs,
                    dtype=np.float32,
                )
                + ObfuscateAffs(self.input_affs)
                + gp.NoiseAugment(self.input_affs, mode="poisson", p=0.33)
                + CustomIntensityAugment(
                    self.input_affs,
                    0.9,
                    1.1,
                    -0.1,
                    0.1,
                    z_section_wise=True,
                    p=0.5,
                )
                + SmoothAugment(self.input_affs, p=0.5)
                + gp.DefectAugment(
                    self.input_affs,
                    prob_missing=0.15,
                    prob_low_contrast=0.05,
                    prob_deform=0.0,
                    axis=1,
                )
            )

        self.pipeline += (
            gp.GrowBoundary(
                self.labels, steps=self.aff_grow_boundary, only_xy=True
            )
            + gp.AddAffinities(
                affinity_neighborhood=self.aff_neighborhood,
                labels=self.labels,
                affinities=self.gt_affs,
                dtype=np.float32,
            )
            + gp.BalanceLabels(self.gt_affs, self.affs_weights)
        )

    def __iter__(self):
        return iter(self.__yield_sample())

    def __yield_sample(self):
        with gp.build(self.pipeline):
            while True:
                request = gp.BatchRequest()
                request.add(self.labels, self.input_shape)

                if "lsd" in self.model_type:
                    request.add(self.input_lsds, self.input_shape)
                if "2d_affs" in self.model_type or "mtlsd" in self.model_type:
                    request.add(self.input_affs, self.input_shape)

                request.add(self.gt_affs, self.output_shape)
                request.add(self.affs_weights, self.output_shape)

                sample = self.pipeline.request_batch(request)

                gt_affs_data = sample[self.gt_affs].data.copy()
                affs_weights_data = sample[self.affs_weights].data.copy()
                if "2d_lsd" in self.model_type:
                    input_lsds_data = sample[self.input_lsds].data.copy()
                    yield input_lsds_data, gt_affs_data, affs_weights_data
                elif "2d_affs" in self.model_type:
                    input_affs_data = sample[self.input_affs].data.copy()
                    yield input_affs_data, gt_affs_data, affs_weights_data
                elif "2d_mtlsd" in self.model_type:
                    input_lsds_data = sample[self.input_lsds].data.copy()
                    input_affs_data = sample[self.input_affs].data.copy()
                    yield input_lsds_data, input_affs_data, gt_affs_data, affs_weights_data
                else:
                    raise ValueError(f"Unknown model type: {self.model_type}")
