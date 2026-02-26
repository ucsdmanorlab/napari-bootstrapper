import gunpowder as gp
import numpy as np


class BumpBackground(gp.BatchFilter):
    '''Bump background ID to max_id+1. '''

    def __init__(self,labels):
        self.labels = labels

    def process(self, batch, request):
        label_data = batch.arrays[self.labels].data
        dtype = label_data.dtype

        label_data[label_data == 0] = np.amax(np.unique(label_data)) + 1
        batch.arrays[self.labels].data = label_data.astype(dtype)


class UnbumpBackground(gp.BatchFilter):
    '''UnBump background ID back to 0. '''

    def __init__(self,labels):
        self.labels = labels

    def process(self, batch, request):
        label_data = batch.arrays[self.labels].data
        dtype = label_data.dtype

        uniques = np.unique(label_data)

        label_data[label_data == np.amax(uniques)] = 0

        batch.arrays[self.labels].data = label_data.astype(dtype)
