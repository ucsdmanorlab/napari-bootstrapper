import gunpowder as gp
import numpy as np
import random
from skimage.exposure import equalize_adapthist as clahe
import itertools


class ClaheAugment(gp.BatchFilter):
    """Randomly apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        array (:class`ArrayKey`):
            The intensity array to modify.
        clip_limit_range (tuple of float):
            The min and max of the clip limit. ClaheAugment draws the clip
            limit uniformly at random between these two values.
        signal_threshold (float):
            ClaheAugment applies CLAHE to a slice only if the slice's range
            (max - min) is above this value. It skips the other slices to
            avoid amplifying noise.
        p (float):
            The probability of applying this augmentation.
    """
    
    def __init__(self, array, clip_limit_range=(0.01, 0.1), signal_threshold=0.01, p=1.0):
        self.array = array
        self.clip_limit_range = clip_limit_range
        self.signal_threshold = signal_threshold
        self.p = p
        assert self.clip_limit_range[1] >= self.clip_limit_range[0]

    def setup(self):
        self.enable_autoskip()
        self.updates(self.array, self.spec[self.array])

    def skip_node(self, request):
        return random.random() > self.p

    def process(self, batch, request):
        array_data = batch[self.array].data
        clip_limit = random.uniform(self.clip_limit_range[0], self.clip_limit_range[1])
        kernel_size = array_data.shape[-2:]
        
        batch[self.array].data = self._augment(
            array_data, clip_limit, kernel_size
        )

    def _augment(self, data, clip_limit, kernel_size):
        if data.ndim == 2:
            if abs(data.max() - data.min()) > self.signal_threshold:
                return clahe(data, clip_limit=clip_limit, kernel_size=kernel_size).astype(data.dtype)
            else:
                return data
        
        elif data.ndim > 2:
            result = np.empty_like(data)
            for i in range(data.shape[0]):
                result[i] = self._augment(data[i], clip_limit, kernel_size)
            return result
        
        else:
            raise RuntimeError(f"ClaheAugment requires at least 2D arrays, got {data.ndim}D")