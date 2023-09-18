import gunpowder as gp
import numpy as np
import random
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
    generate_binary_structure
)
from skimage.morphology import disk
from skimage.measure import label


class CreateLabels(gp.BatchFilter):
    def __init__(
        self,
        labels,
        anisotropy
    ):

        self.labels = labels
        self.anisotropy = anisotropy + 1

    def process(self, batch, request):

        labels = batch[self.labels].data
        labels = np.concatenate([labels,]*self.anisotropy)
        shape = labels.shape

        # different numbers simulate more or less objects
        num_points = random.randint(25,50)

        for n in range(num_points):
            z = random.randint(1, labels.shape[0] - 1)
            y = random.randint(1, labels.shape[1] - 1)
            x = random.randint(1, labels.shape[2] - 1)

            labels[z, y, x] = 1
        
        structs = [generate_binary_structure(2, 2), disk(random.randint(1,5))]

        #different numbers will simulate larger or smaller objects
        for z in range(labels.shape[0]):
            
            dilations = random.randint(1, 10)
            struct = random.choice(structs)

            dilated = binary_dilation(
                labels[z], structure=struct, iterations=dilations
            )

            labels[z] = dilated.astype(labels.dtype)

        #relabel
        labels = label(labels)

        #expand labels
        distance = labels.shape[0]

        distances, indices = distance_transform_edt(
            labels == 0, return_indices=True
        )

        expanded_labels = np.zeros_like(labels)

        dilate_mask = distances <= distance

        masked_indices = [
            dimension_indices[dilate_mask] for dimension_indices in indices
        ]

        nearest_labels = labels[tuple(masked_indices)]

        expanded_labels[dilate_mask] = nearest_labels

        labels = expanded_labels

        #change background
        labels[labels == 0] = np.max(labels) + 1

        #relabel
        labels = label(labels)

        batch[self.labels].data = labels[::self.anisotropy].astype(np.uint64)
