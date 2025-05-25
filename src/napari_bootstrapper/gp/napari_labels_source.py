import gunpowder as gp
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

    def __init__(
        self,
        labels: Labels,
        key: gp.ArrayKey,
        spec: ArraySpec,
        channels_dim: int | None = None,
    ):
        self.array_spec = spec
        self.channels_dim = channels_dim

        self.labels = gp.Array(
            data=(
                labels.data
                if (self.channels_dim is None or len(labels.data.shape) == 3)
                else labels.data.max(axis=self.channels_dim)
            ),
            spec=spec,
        )
        self.key = key

    def setup(self):
        self.provides(self.key, self.array_spec.copy())

    def provide(self, request):
        output = gp.Batch()
        output[self.key] = self.labels.crop(request[self.key].roi)
        return output
