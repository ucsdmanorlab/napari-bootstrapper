import random

import numpy as np

from .add_2d_lsds import Add2DLSDs, compute_lsds


class AddObfuscated2DLSDs(Add2DLSDs):
    def _descriptor(self, labels, crop):
        descriptors = []
        for z in range(crop[0].start, crop[0].stop):
            labels_sec = np.copy(labels[z])
            if np.random.random() > 0.2:
                labels_sec = self._random_merge(labels_sec)
            descriptors.append(
                compute_lsds(
                    labels_sec,
                    self.sigma[1:],
                    self.voxel_size[1:],
                    self.downsample,
                    crop[1:],
                )
            )
        return np.stack(descriptors, axis=1)

    def _random_merge(self, array, num_pairs_to_merge=4):

        unique_ids = np.unique(array)

        if len(unique_ids) < 2:
            return array

        np.random.shuffle(unique_ids)

        max_pairs = len(unique_ids) // 2
        pairs_to_merge = min(num_pairs_to_merge, max_pairs)

        for _ in range(random.randrange(pairs_to_merge)):
            label1, label2 = np.random.choice(unique_ids, 2, replace=False)
            array[array == label2] = label1
            unique_ids = unique_ids[unique_ids != label2]

        return array
