import gunpowder as gp
import numpy as np
from gunpowder.array_spec import ArraySpec
from gunpowder.profiling import Timing


class NpArraySource(gp.BatchProvider):
    """
    A gunpowder source that provides a numpy array in the given spec
    Args:
        spec (ArraySpec):
            The spec to populate with ones (virtually)
        key (``gp.ArrayKey``):
            The key to provide data into
    """

    def __init__(self, array: np.ndarray, spec: ArraySpec, key: gp.ArrayKey):
        self.array = gp.Array(array, spec=spec)
        self.key = key

    def setup(self):
        self.provides(self.key, self.array.spec.copy())

    def provide(self, request):
        output = gp.Batch()

        timing_provide = Timing(self, "provide")
        timing_provide.start()

        output[self.key] = self.array.crop(request[self.key].roi)

        timing_provide.stop()
        output.profiling_stats.add(timing_provide)

        return output
