import random

import gunpowder as gp
import numpy as np


class CustomIntensityAugment(gp.BatchFilter):
    """Randomly scale and shift the values of an array.
    Works on arrays with num_channels > 1.

    Args:

        array (:class:`ArrayKey`):

            The intensity array to modify.

        scale_min (``float``):
        scale_max (``float``):
        shift_min (``float``):
        shift_max (``float``):

            The min and max of the uniformly randomly drawn scaling and
            shifting values for the intensity augmentation. Intensities are
            changed as::

                a = a.mean() + (a-a.mean())*scale + shift

        z_section_wise (``bool``):

            Perform the augmentation z-section wise. Requires 3D arrays and
            assumes that z is the first dimension.

        clip (``bool``):

            Set to False if modified values should not be clipped to [0, 1]
            Disables range check!

        p (``float``, optional):

            Probability applying the augmentation. Default is 1.0 (always
            apply). Should be a float value between 0 and 1. Lowering this value
            could be useful for computational efficiency and increasing
            augmentation space.
    """

    def __init__(
        self,
        array,
        scale_min,
        scale_max,
        shift_min,
        shift_max,
        z_section_wise=False,
        clip=True,
        p=1.0,
    ):
        self.array = array
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.shift_min = shift_min
        self.shift_max = shift_max
        self.z_section_wise = z_section_wise
        self.clip = clip
        self.p = p

    def setup(self):
        self.enable_autoskip()
        self.updates(self.array, self.spec[self.array])

    def skip_node(self, request):
        return random.random() > self.p

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.array] = request[self.array].copy()
        return deps

    def process(self, batch, request):
        raw = batch.arrays[self.array]

        assert (
            not self.z_section_wise or raw.spec.roi.dims == 3
        ), "If you specify 'z_section_wise', I expect 3D data."
        assert raw.data.dtype == np.float32 or raw.data.dtype == np.float64, (
            "Intensity augmentation requires float types for the raw array (not "
            + str(raw.data.dtype)
            + "). Consider using Normalize before."
        )
        if self.clip:
            assert (
                raw.data.min() >= 0 and raw.data.max() <= 1
            ), "Intensity augmentation expects raw values in [0,1]. Consider using Normalize before."

        if self.z_section_wise:
            for z in range(
                (raw.spec.roi / self.spec[self.array].voxel_size).shape[0]
            ):
                if len(raw.data.shape) == 3:
                    raw.data[z] = self.__augment(
                        raw.data[z],
                        np.random.uniform(
                            low=self.scale_min, high=self.scale_max
                        ),
                        np.random.uniform(
                            low=self.shift_min, high=self.shift_max
                        ),
                    )
                else:
                    raw.data[:, z, :, :] = self.__augment(
                        raw.data[:, z, :, :],
                        np.random.uniform(
                            low=self.scale_min, high=self.scale_max
                        ),
                        np.random.uniform(
                            low=self.shift_min, high=self.shift_max
                        ),
                    )
        else:
            raw.data = self.__augment(
                raw.data,
                np.random.uniform(low=self.scale_min, high=self.scale_max),
                np.random.uniform(low=self.shift_min, high=self.shift_max),
            )

        # clip values, we might have pushed them out of [0,1]
        if self.clip:
            raw.data[raw.data > 1] = 1
            raw.data[raw.data < 0] = 0

    def __augment(self, a, scale, shift):
        return a.mean() + (a - a.mean()) * scale + shift
