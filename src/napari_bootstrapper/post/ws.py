import numpy as np
import waterz
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    label,
    maximum_filter,
)
from skimage.segmentation import watershed as skimage_watershed


def watershed_from_boundary_distance(
    boundary_distances,
    boundary_mask,
    return_seeds=False,
    id_offset=0,
    min_seed_distance=10,
):

    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered == boundary_distances

    seeds, n = label(maxima)

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds != 0] += id_offset

    fragments = skimage_watershed(
        boundary_distances.max() - boundary_distances,
        seeds,
        mask=boundary_mask,
    )
    # fragments = cwatershed(boundary_distances.max() - boundary_distances, seeds)

    ret = (fragments.astype(np.uint64), n + id_offset)
    if return_seeds:
        ret = ret + (seeds.astype(np.uint64),)

    return ret


def watershed_from_affinities(
    affs,
    max_affinity_value=1.0,
    fragments_in_xy=False,
    return_seeds=False,
    min_seed_distance=10,
    noise_eps=None,
    sigma=None,
    bias=None,
    threshold=0.5,
):
    """Extract initial fragments from affinities using a watershed
    transform. Returns the fragments and the maximal ID in it.
    Returns:
        (fragments, max_id)
        or
        (fragments, max_id, seeds) if return_seeds == True"""

    sigma = (0, *sigma) if sigma is not None else None

    # add some random noise to affs (this is particularly necessary if your affs are
    #  stored as uint8 or similar)
    # If you have many affinities of the exact same value the order they are processed
    # in may be fifo, so you can get annoying streaks.

    ### tmp comment out ###

    shift = np.zeros_like(affs)

    if noise_eps is not None:
        shift += np.random.randn(*affs.shape) * noise_eps

    #######################

    # add smoothed affs, to solve a similar issue to the random noise. We want to bias
    # towards processing the central regions of objects first.

    ### tmp comment out ###

    if sigma is not None:
        shift += gaussian_filter(affs, sigma=sigma) - affs

    #######################
    if bias is not None:
        if bias is float:
            bias = [bias] * affs.shape[0]
        else:
            assert len(bias) == affs.shape[0]
        shift += np.array([bias]).reshape(
            (-1, *((1,) * (len(affs.shape) - 1)))
        )

    affs = affs + shift

    if fragments_in_xy:

        mean_affs = 0.5 * (affs[-1] + affs[-2])  # affs are (c,z,y,x)
        # mean_affs = (1 / 3) * (
        #     affs[0] + affs[1] + affs[2]
        # )  # todo: other affinities? *0.5
        depth = mean_affs.shape[0]

        fragments = np.zeros(mean_affs.shape, dtype=np.uint64)
        if return_seeds:
            seeds = np.zeros(mean_affs.shape, dtype=np.uint64)

        id_offset = 0
        for z in range(depth):

            boundary_mask = mean_affs[z] > 0.5 * max_affinity_value
            boundary_distances = distance_transform_edt(boundary_mask)

            ret = watershed_from_boundary_distance(
                boundary_distances,
                boundary_mask,
                return_seeds=return_seeds,
                id_offset=id_offset,
                min_seed_distance=min_seed_distance,
            )

            fragments[z] = ret[0]
            if return_seeds:
                seeds[z] = ret[2]

            id_offset = ret[1]

        ret = (fragments, id_offset)
        if return_seeds:
            ret += (seeds,)

    else:

        boundary_mask = np.mean(affs, axis=0) > 0.5 * max_affinity_value
        boundary_distances = distance_transform_edt(boundary_mask)

        ret = watershed_from_boundary_distance(
            boundary_distances,
            boundary_mask,
            return_seeds,
            min_seed_distance=min_seed_distance,
        )

        fragments = ret[0]

    # return fragments
    thresholds = [threshold]
    segmentations = waterz.agglomerate(
        affs=affs.astype(np.float32),
        fragments=fragments.copy(),
        thresholds=thresholds,
    )

    segmentation = next(segmentations)
    return segmentation
