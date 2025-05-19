import gunpowder as gp
import numpy as np
from napari.layers import Image, Labels
from torch.utils.data import IterableDataset

from ..gp.add_2d_lsds import Add2DLSDs
from ..gp.calc_max_padding import calc_max_padding
from ..gp.napari_image_source import NapariImageSource
from ..gp.napari_labels_source import NapariLabelsSource
from ..gp.smooth_augment import SmoothAugment


class Napari2DDataset(IterableDataset):
    def __init__(
        self,
        image_layer: Image,
        labels_layer: Labels,
        mask_layer: Labels,
        model_type: str,
        lsd_sigma: int = 20,
        lsd_downsample: int = 4,
        aff_neighborhood: list[list[int]] = [[-1, 0], [0, -1]],
        aff_grow_boundary: int = 1,
    ):
        self.model_type = model_type
        self.image_layer = image_layer
        self.labels_layer = labels_layer
        self.mask_layer = mask_layer

        self.input_shape = 3, 212, 212  # adjacent sections as extra channels
        self.output_shape = 1, 120, 120
        self.voxel_size = 1, 1, 1
        self.offset = 0, 0, 0

        self.lsd_sigma = (0, lsd_sigma, lsd_sigma)
        self.lsd_downsample = lsd_downsample
        self.aff_neighborhood = [
            [0, *x] for x in aff_neighborhood
        ]
        self.aff_grow_boundary = aff_grow_boundary

        self.voxel_size = gp.Coordinate(self.voxel_size)
        self.offset = gp.Coordinate(self.offset)
        self.shape = gp.Coordinate(self.image_layer.data.shape)
        context = (
            calc_max_padding(self.output_shape, self.voxel_size, self.lsd_sigma)
            if self.model_type != "2d_affs"
            else (gp.Coordinate(self.input_shape) - gp.Coordinate(self.output_shape)) // 2
        )
        self.context = (0, context[1], context[2]) # ensure 2D padding

        # get unet num_channels from shape
        if len(self.shape) == 4:
            num_channels = self.shape[0]
        elif len(self.shape) == 3:
            num_channels = 1
        else:
            raise ValueError("Image must be 3D")
        self.num_channels = num_channels * self.input_shape[0]

        self.__setup_pipeline()

    def __setup_pipeline(self):
        self.raw = gp.ArrayKey("RAW")
        self.labels = gp.ArrayKey("LABELS")
        self.mask = gp.ArrayKey("MASK")

        self.gt_lsds = gp.ArrayKey("GT_LSDS")
        self.lsds_weights = gp.ArrayKey("LSDS_WEIGHTS")
        self.gt_affs = gp.ArrayKey("GT_AFFS")
        self.gt_affs_mask = gp.ArrayKey("GT_AFFS_MASK")
        self.affs_weights = gp.ArrayKey("AFFS_WEIGHTS")

        raw_spec = gp.ArraySpec(
            roi=gp.Roi(self.offset, self.voxel_size * self.shape),
            dtype=self.image_layer.data.dtype,
            interpolatable=True,
            voxel_size=self.voxel_size,
        )
        labels_spec = gp.ArraySpec(
            roi=gp.Roi(self.offset, self.voxel_size * self.shape),
            dtype=self.labels_layer.data.dtype,
            interpolatable=False,
            voxel_size=self.voxel_size,
        )
        mask_spec = gp.ArraySpec(
            roi=gp.Roi(self.offset, self.voxel_size * self.shape),
            dtype=self.mask_layer.data.dtype,
            interpolatable=False,
            voxel_size=self.voxel_size,
        )

        self.pipeline = (
            (
                NapariImageSource(
                    image=self.image_layer,
                    key=self.raw,
                    spec=raw_spec,
                ),
                NapariLabelsSource(
                    labels=self.labels_layer,
                    key=self.labels,
                    spec=labels_spec,
                ),
                NapariLabelsSource(
                    labels=self.mask_layer,
                    key=self.mask,
                    spec=mask_spec,
                ),
            )
            + gp.MergeProvider()
            + gp.AsType(self.labels, np.uint32)
            + gp.Pad(self.raw, None)
            + gp.Pad(self.labels, self.context)
            + gp.Pad(self.mask, self.context)
            + gp.RandomLocation(mask=self.mask, min_masked=0.001)
            + gp.Normalize(self.raw)
            + gp.SimpleAugment(transpose_only=[1, 2])
            + gp.DeformAugment(
                control_point_spacing=(10, 10),
                jitter_sigma=(3.0, 3.0),
                spatial_dims=2,
                subsample=1,
                scale_interval=(0.9, 1.1),
                graph_raster_voxel_size=self.voxel_size[1:],
                p=0.5,
            )
            + gp.NoiseAugment(self.raw, p=0.5)
            + gp.IntensityAugment(
                self.raw,
                scale_min=0.9,
                scale_max=1.1,
                shift_min=-0.1,
                shift_max=0.1,
                z_section_wise=True,
                p=0.5,
            )
            + SmoothAugment(self.raw, p=0.5)
            + gp.DefectAugment(
                self.raw,
                prob_missing=0.0 if self.input_shape[0] == 1 else 0.05,
            )
            + gp.IntensityScaleShift(self.raw, 2, -1)
        )

        if self.model_type == "2d_lsd":
            self.pipeline += Add2DLSDs(
                self.labels,
                self.gt_lsds,
                unlabelled=self.mask,
                lsds_mask=self.lsds_weights,
                sigma=self.lsd_sigma,
                downsample=self.lsd_downsample,
            )
        elif self.model_type == "2d_affs":
            self.pipeline += (
                gp.GrowBoundary(
                    self.labels, self.mask, steps=self.aff_grow_boundary, only_xy=True
                )
                + gp.AddAffinities(
                    self.aff_neighborhood,
                    self.labels,
                    self.gt_affs,
                    unlabelled=self.mask,
                    affinities_mask=self.gt_affs_mask,
                    dtype=np.float32,
                )
                + gp.BalanceLabels(
                    self.gt_affs, self.affs_weights, mask=self.gt_affs_mask
                )
            )
        elif self.model_type == "2d_mtlsd":
            self.pipeline += (
                Add2DLSDs(
                    self.labels,
                    self.gt_lsds,
                    unlabelled=self.mask,
                    lsds_mask=self.lsds_weights,
                    sigma=self.lsd_sigma,
                    downsample=self.lsd_downsample,
                )
                + gp.GrowBoundary(
                    self.labels, self.mask, steps=self.aff_grow_boundary, only_xy=True
                )
                + gp.AddAffinities(
                    self.aff_neighborhood,
                    self.labels,
                    self.gt_affs,
                    unlabelled=self.mask,
                    affinities_mask=self.gt_affs_mask,
                    dtype=np.float32,
                )
                + gp.BalanceLabels(
                    self.gt_affs, self.affs_weights, mask=self.gt_affs_mask
                )
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def __iter__(self):
        return iter(self.__yield_sample())

    def __yield_sample(self):
        with gp.build(self.pipeline):
            while True:
                request = gp.BatchRequest()
                request.add(self.raw, self.input_shape)
                request.add(self.labels, self.output_shape)
                request.add(self.mask, self.output_shape)

                if "lsd" in self.model_type:
                    request.add(self.gt_lsds, self.output_shape)
                    request.add(self.lsds_weights, self.output_shape)
                if "affs" in self.model_type or "mtlsd" in self.model_type:
                    request.add(self.gt_affs, self.output_shape)
                    request.add(self.affs_weights, self.output_shape)

                sample = self.pipeline.request_batch(request)

                raw_data = sample[self.raw].data.copy()
                
                if self.model_type == "2d_lsd":
                    gt_lsds_data = sample[self.gt_lsds].data.copy()
                    lsds_weights_data = sample[self.lsds_weights].data.copy()
                    yield raw_data, gt_lsds_data, lsds_weights_data
                elif self.model_type == "2d_affs":
                    gt_affs_data = sample[self.gt_affs].data.copy()
                    affs_weights_data = sample[self.affs_weights].data.copy()
                    yield raw_data, gt_affs_data, affs_weights_data
                elif self.model_type == "2d_mtlsd":
                    gt_lsds_data = sample[self.gt_lsds].data.copy()
                    lsds_weights_data = sample[self.lsds_weights].data.copy()
                    gt_affs_data = sample[self.gt_affs].data.copy()
                    affs_weights_data = sample[self.affs_weights].data.copy()
                    yield raw_data, gt_lsds_data, lsds_weights_data, gt_affs_data, affs_weights_data
                else:
                    raise ValueError(f"Unknown model type: {self.model_type}")