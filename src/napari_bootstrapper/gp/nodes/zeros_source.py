import gunpowder as gp
import numpy as np


class ZerosSource(gp.BatchProvider):
    def __init__(self, datasets, shape=None, dtype=np.uint64, array_specs=None):
        self.datasets = datasets

        if array_specs is None:
            self.array_specs = {}
        else:
            self.array_specs = array_specs

        self.shape = gp.Coordinate(shape) if shape is not None else gp.Coordinate((200, 200, 200))
        self.dtype = dtype

        # number of spatial dimensions
        self.ndims = None

    def setup(self):
        for array_key, ds_name in self.datasets.items():
            if array_key in self.array_specs:
                spec = self.array_specs[array_key].copy()
            else:
                spec = gp.ArraySpec()

            if spec.voxel_size is None:
                voxel_size = gp.Coordinate((1,) * len(self.shape))
                spec.voxel_size = voxel_size

            self.ndims = len(spec.voxel_size)

            if spec.roi is None:
                offset = gp.Coordinate((0,) * self.ndims)
                spec.roi = gp.Roi(offset, self.shape * spec.voxel_size)

            if spec.dtype is not None:
                assert spec.dtype == self.dtype
            else:
                spec.dtype = self.dtype

            if spec.interpolatable is None:
                spec.interpolatable = spec.dtype in [
                    np.float,
                    np.float32,
                    np.float64,
                    np.float10,
                    np.uint8,  # assuming this is not used for labels
                ]

            self.provides(array_key, spec)

    def provide(self, request):
        batch = gp.Batch()

        for array_key, request_spec in request.array_specs.items():
            voxel_size = self.spec[array_key].voxel_size

            # scale request roi to voxel units
            dataset_roi = request_spec.roi / voxel_size

            # shift request roi into dataset
            dataset_roi = (
                dataset_roi - self.spec[array_key].roi.get_offset() / voxel_size
            )


            # create array spec
            array_spec = self.spec[array_key].copy()
            array_spec.roi = request_spec.roi

            # add array to batch
            batch.arrays[array_key] = gp.Array(
                np.zeros(self.shape, self.dtype)[dataset_roi.to_slices()], array_spec
            )

        return batch
