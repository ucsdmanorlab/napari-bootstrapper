import gunpowder as gp
import numpy as np


class Unlabel(gp.BatchFilter):
    def __init__(self, labels, unlabelled):
        self.labels = labels
        self.unlabelled = unlabelled

    def setup(self):
        up_spec = self.spec[self.labels].copy()
        up_spec.dtype = np.uint8
        #self.provides(self.unlabelled, self.spec[self.labels].copy())
        self.provides(self.unlabelled, up_spec)

    def prepare(self, request):
        deps = gp.BatchRequest()
        labels_spec = request[self.unlabelled].copy()
        labels_spec.dtype = np.uint32 #request[self.labels].dtype
        deps[self.labels] = labels_spec #request[self.unlabelled].copy()

        return deps

    def process(self, batch, request):
        labels = batch[self.labels].data

        unlabelled = (labels > 0).astype(np.uint8)

        spec = batch[self.labels].spec.copy()
        spec.roi = request[self.unlabelled].roi.copy()
        spec.dtype = np.uint8

        batch = gp.Batch()

        batch[self.unlabelled] = gp.Array(unlabelled, spec)

        return batch
