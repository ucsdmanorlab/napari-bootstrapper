import numpy as np
import waterz
from scipy.ndimage import (
    label,
    maximum_filter,
    measurements,
    distance_transform_edt,
    gaussian_filter,
)
from skimage.filters import median, threshold_otsu
from skimage.io import imread, imsave
from skimage.measure import label as relabel
from skimage.segmentation import watershed
import zarr


def get_segmentation(affinities, fragments, threshold):

    # sometimes weird things happen when returning frags without copying
    test = fragments.copy()
    thresholds = [threshold]

    segmentations = waterz.agglomerate(
        affs=affinities.astype(np.float32),
        fragments=test,
        thresholds=thresholds,
    )

    segmentation = next(segmentations)

    return segmentation


def watershed_from_affinities(
    affs,
    max_affinity_value=1.0,
    min_seed_distance=10,
):

    mean_affs = 0.5 * (affs[1] + affs[2])
    depth = mean_affs.shape[0]

    fragments = np.zeros(mean_affs.shape, dtype=np.uint64)

    id_offset = 0

    for z in range(depth):

        # could also try > mean
        boundary_mask = np.mean(affs[:, z], axis=0) > threshold_otsu(affs)
        boundary_distances = distance_transform_edt(boundary_mask)

        ret = watershed_from_boundary_distance(
            boundary_distances,
            boundary_mask=boundary_mask,
            id_offset=id_offset,
            min_seed_distance=min_seed_distance,
        )

        fragments[z] = ret[0]
        id_offset = ret[1]

    ret = (fragments, id_offset)

    return ret


def watershed_from_boundary_distance(
    boundary_distances, boundary_mask=None, id_offset=0, min_seed_distance=10
):

    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered == boundary_distances
    seeds, n = label(maxima)

    print(f"Found {n} fragments")

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds != 0] += id_offset

    # have also tried with a boundary mask, same issues
    fragments = watershed(
        boundary_distances.max() - boundary_distances,
        seeds,
        # mask=boundary_mask
    )

    ret = (fragments.astype(np.uint64), n + id_offset)

    return ret
