import gunpowder as gp
import numpy as np
from gunpowder.array_spec import ArraySpec
from napari.layers import Labels


class NapariLabelsSource(gp.BatchProvider):
    """
    A gunpowder node to pull data from a napari Labels
    Args:
        labels (Labels):
            The napari labels layer to pull data from
        key (``gp.ArrayKey``):
            The key to provide data into
    """

    def __init__(self, labels: Labels, key: gp.ArrayKey, spec: ArraySpec):
        self.array_spec = spec

        self.labels = gp.Array(
            data=labels.data,
            spec=spec,
        )
        self.key = key

    def setup(self):
        self.provides(self.key, self.array_spec.copy())

    def provide(self, request):
        output = gp.Batch()
        output[self.key] = self.labels.crop(request[self.key].roi)
        return output
