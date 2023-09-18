import math
from typing import Tuple, List

import numpy as np
import gunpowder as gp
from napari.layers import Image, Labels
from torch.utils.data import IterableDataset

from lsd.train.gp import AddLocalShapeDescriptor

from ..gp.nodes.napari_source_2d import NapariSource2D
from ..gp.nodes.unlabel import Unlabel
from ..gp.nodes.random_noise import RandomNoiseAugment
from ..gp.nodes.smooth_array import SmoothArray


class NapariDataset2D(IterableDataset):  # type: ignore
    def __init__(
        self,
        raw: Image,
        labels: Labels,
        voxel_size: List[int],
        min_masked: float,
        control_point_spacing: int,
        control_point_jitter: float,
    ):
        """A dataset that serves random samples from a zarr container.

        Args:

            layer:

                The napari layer to use.
                The data should have shape `(s, c, [t,] [z,] y, x)`, where
                `s` = # of samples, `c` = # of channels, `t` = # of frames, and
                `z`/`y`/`x` are spatial extents. The dataset should have an
                `"axis_names"` attribute that contains the names of the used
                axes, e.g., `["s", "c", "y", "x"]` for a 2D dataset.

            control_point_spacing:

                The distance in pixels between control points used for elastic
                deformation of the raw data.

            control_point_jitter:

                How much to jitter the control points for elastic deformation
                of the raw data, given as the standard deviation of a normal
                distribution with zero mean.
        """

        self.raw_layer = raw
        self.labels_layer = labels
        self.voxel_size = voxel_size
        self.min_masked = min_masked
        self.control_point_spacing = control_point_spacing
        self.control_point_jitter = control_point_jitter

        # get number of channels of 3D image data
        self.num_channels = self.raw_layer.data.shape[0] if len(self.raw_layer.data.shape) == 4 else 1 #TODO: fix 

        #TODO: account for raw.shape != labels.shape, offset.
        self.available_sections = list(np.where(np.any(self.labels_layer.data, axis=(-2,-1)))[0])

        self.__setup_pipeline()

    def __iter__(self):
        return iter(self.__yield_sample())

    def __calc_max_padding(
        self, output_size, voxel_size, sigma, mode="shrink"
    ):
        method_padding = gp.Coordinate((sigma * 2,) * 2)
        
        diag = np.sqrt(output_size[0] ** 2 + output_size[1] ** 2)

        max_padding = gp.Roi(
            (gp.Coordinate([i / 2 for i in [diag, diag]]) + method_padding),
            (0,) * 2,   
        ).snap_to_grid(voxel_size, mode=mode)
        
        return max_padding.get_begin()

    def __setup_pipeline(self):
        self.raw = gp.ArrayKey("RAW")
        self.labels = gp.ArrayKey("LABELS")
        self.unlabelled = gp.ArrayKey("UNLABELLED")
       
        #TODO: add affs as an option for pred 
        self.gt_lsds = gp.ArrayKey("GT_LSDS")
        self.lsds_mask = gp.ArrayKey("LSDS_MASK")

        #TODO: have option to set input_shape
        input_shape = gp.Coordinate((196, 196))
        output_shape = gp.Coordinate((104, 104))

        self.voxel_size = gp.Coordinate(self.voxel_size)

        input_size = input_shape * self.voxel_size
        output_size = output_shape * self.voxel_size
        
        self.datasets = [
            (self.raw, input_size),
            (self.labels, output_size),
            (self.unlabelled, output_size),
            (self.gt_lsds, output_size),
            (self.lsds_mask, output_size),
        ]

        self.request = gp.BatchRequest()
        for ds_name, size in self.datasets:
            self.request.add(ds_name, size)

        #TODO: have option to set sigma
        sigma = int(self.voxel_size[-1] * 10)
        
        labels_padding = self.__calc_max_padding(
            output_size, self.voxel_size, sigma
        )

        self.pipeline = (
            tuple(
            (
                NapariSource2D(
                    self.raw_layer, 
                    self.raw, 
                    section, 
                ), 
                NapariSource2D(
                    self.labels_layer, 
                    self.labels, 
                    section, 
                ) 
            )
            + gp.MergeProvider()
            + Unlabel(self.labels, self.unlabelled)
            + gp.Normalize(self.raw)
            + gp.Pad(self.raw, None)
            + gp.Pad(self.labels, labels_padding)
            + gp.Pad(self.unlabelled, labels_padding)
            + gp.RandomLocation(mask=self.unlabelled, min_masked=self.min_masked)
            for section in self.available_sections)
            + gp.RandomProvider()
            + gp.ElasticAugment(
                control_point_spacing=(self.control_point_spacing,) * 2,
                jitter_sigma=(self.control_point_jitter,) * 2,
                rotation_interval=(0, math.pi / 2),
                scale_interval=(0.75, 1.25),
                subsample=4,
                spatial_dims=2,
            )
            + gp.SimpleAugment()
            + RandomNoiseAugment(self.raw)
            + gp.IntensityAugment(self.raw, 0.9, 1.1, -0.1, 0.1)
            + SmoothArray(self.raw, (0.0,1.0))
            + AddLocalShapeDescriptor(
                self.labels,
                self.gt_lsds,
                unlabelled=self.unlabelled,
                lsds_mask=self.lsds_mask,
                sigma=sigma,
                downsample=2)  #TODO: set = 2 if possible.)
        )
            
        if self.num_channels == 1:
           self.pipeline += gp.Unsqueeze([self.raw])
            
    def __yield_sample(self):
        """An infinite generator of crops."""

        with gp.build(self.pipeline):
            while True:
                sample = self.pipeline.request_batch(self.request)
                yield sample[self.raw].data, sample[self.gt_lsds].data, sample[self.lsds_mask].data

    def get_num_channels(self):
       return self.num_channels
