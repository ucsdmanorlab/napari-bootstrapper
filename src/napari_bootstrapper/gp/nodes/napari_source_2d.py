from typing import Optional, Union

import numpy as np
import gunpowder as gp
from gunpowder.array_spec import ArraySpec
from gunpowder.profiling import Timing
from napari.layers import Image, Labels


class NapariSource2D(gp.BatchProvider):
    """
    A gunpowder interface to a napari Image or Labels layer
    Args:
        layer (Image or Labels):
            The napari Image to pull data from
        key (``gp.ArrayKey``):
            The key to provide data into
        section (``int``)
    """

    def __init__(
        self, layer: Union[Image, Labels], key: gp.ArrayKey, section: int, spec: Optional[ArraySpec] = None
    ):
        if spec is None:
            self.array_spec = self._read_metadata(layer)
        else:
            self.array_spec = spec

        if len(layer.data.shape) == 4:
            self.layer = gp.Array(
                layer.data[:,section], self.array_spec
            )
        elif len(layer.data.shape) == 3:
            self.layer = gp.Array(
                layer.data[section], self.array_spec
            )
        else:
            raise AssertionError(f"Layer data has to have 3 or 4 dimensions, Layer data shape: {data_shape}")

        self.key = key

    def setup(self):
        self.provides(self.key, self.array_spec.copy())

    def provide(self, request):
        output = gp.Batch()

        timing_provide = Timing(self, "provide")
        timing_provide.start()

        output[self.key] = self.layer.crop(request[self.key].roi)

        timing_provide.stop()

        output.profiling_stats.add(timing_provide)

        return output

    def _read_metadata(self, layer):
        # offset assumed to be in world coordinates and 3d

        if isinstance(layer, Image):
            dtype = np.uint8
            interpolatable=True

        elif isinstance(layer, Labels):
            dtype = np.uint32
            interpolatable=False

        else:
            raise TypeError("layer must be of type Image or Labels")

        data_shape = layer.data.shape

        # data has to be 3D
        if (len(data_shape) == 4 and data_shape[1] == 1) or (len(data_shape) == 3 and data_shape[0] == 1) or len(data_shape) == 2:
            raise AssertionError(f"Layer data has to be 3D, Layer data shape: {data_shape}")
        
        ndims = 2 # spatial dims

        offset = gp.Coordinate(layer.metadata.get("offset", (0,) * ndims))
        voxel_size = gp.Coordinate(
            layer.metadata.get("resolution", (1,) * ndims)
        )

        # if offset and resolution acquired from layer metadata
        if len(offset) == 3:
            offset = gp.Coordinate(offset[1:])
        if len(voxel_size) == 3:
            voxel_size = gp.Coordinate(voxel_size[1:])

        shape = gp.Coordinate(layer.data.shape[-offset.dims :])

        return gp.ArraySpec(
            roi=gp.Roi(offset, voxel_size * shape),
            dtype=dtype,
            interpolatable=interpolatable,
            voxel_size=voxel_size,
        )
