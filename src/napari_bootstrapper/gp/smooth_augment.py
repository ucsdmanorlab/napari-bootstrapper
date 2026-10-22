import gunpowder as gp
import numpy as np
import random
from scipy.ndimage import gaussian_filter
import itertools


class SmoothAugment(gp.BatchFilter):
    """Randomly blur an intensity array with a Gaussian filter.

    Args:

        array (:class:`ArrayKey`):

            The intensity array to modify.

        blur_min (``float``):
        blur_max (``float``):

            The min and max of the Gaussian sigma. SmoothAugment draws a
            sigma for each slab uniformly at random between these two values.
        slab (``tuple`` of ``int``, optional):
            The shape of the slabs that SmoothAugment blurs one at a time.
            A -1 stands for the full size of the array along that axis.
            For example, a slab of::

                (2, -1, -1, -1)

            blurs each slice ``[0:2,:]``, ``[2:4,:]``, ... of 4D data
            separately.
    """

    def __init__(self, array, blur_min=0.5, blur_max=1.5, slab=None, p=1.0):
        self.array = array
        self.blur_min = blur_min
        self.blur_max = blur_max
        self.slab = slab
        self.p = p
        assert self.blur_max >= self.blur_min

    def setup(self):
        self.enable_autoskip()
        self.updates(self.array, self.spec[self.array])

    def skip_node(self, request):
        return random.random() > self.p

    def process(self, batch, request):

        array_data = batch[self.array].data

        if self.slab is not None:
            slab = self.slab
        else:
            slab = [-1] * len(array_data.shape)

        slab = tuple(m if s == -1 else s for m, s in zip(array_data.shape, slab))

        slab_ranges = (range(0, m, s) for m, s in zip(array_data.shape, slab))

        for start in itertools.product(*slab_ranges):
            slices = tuple(
                slice(start[d], start[d] + slab[d]) for d in range(len(slab))
            )
            sigma = random.uniform(self.blur_min, self.blur_max)
            array_data[slices] = gaussian_filter(
                array_data[slices], sigma=sigma
            ).astype(array_data.dtype)

        batch[self.array].data = array_data
