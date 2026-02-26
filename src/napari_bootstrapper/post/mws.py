import logging

import numba
import mwatershed as mws
import numpy as np
from scipy.ndimage import gaussian_filter, label, maximum_filter, measurements
from scipy.ndimage.morphology import distance_transform_edt
from skimage.measure import label as relabel
from skimage.morphology import remove_small_objects

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@numba.njit(parallel=True)
def replace_values(arr, src, dst):
    shape = arr.shape
    arr = arr.ravel()
    label_map = {src[i]: dst[i] for i in range(len(src))}
    relabeled_arr = np.zeros_like(arr)

    for i in numba.prange(arr.shape[0]):  # type: ignore[non-iterable]
        relabeled_arr[i] = label_map.get(arr[i], arr[i])

    return relabeled_arr.reshape(shape)


def mwatershed_from_affinities(
    affs: np.ndarray,
    neighborhood: list[list[int]],
    bias: list[float],
    sigma: list[int] | None = None,
    filter_fragments: float | None = None,
    min_seed_distance: int | None = None,
    seed_eps: float | None = None,
    noise_eps: float | None = None,
    strides: list[list[int]] | None = None,
    randomized_strides: bool = False,
    remove_debris: int | None = None,
):
    # return fragments_data
    fragments_data = compute_fragments(affs, sigma=sigma, bias=bias, noise_eps=noise_eps, min_seed_distance=min_seed_distance, seed_eps=seed_eps, neighborhood=neighborhood, strides=strides, randomized_strides=randomized_strides)

    # filter fragments
    if filter_fragments > 0:
        filter_avg_fragments(affs, fragments_data, filter_fragments)

    # remove small debris
    if remove_debris > 0:
        fragments_dtype = fragments_data.dtype
        fragments_data = fragments_data.astype(np.int64)
        fragments_data = remove_small_objects(
            fragments_data, min_size=remove_debris
        )
        fragments_data = fragments_data.astype(fragments_dtype)

    return fragments_data


def filter_avg_fragments(affs, fragments_data, filter_value):
    # tmp (think about this)
    average_affs = np.mean(affs[0:3], axis=0)

    filtered_fragments = []

    fragment_ids = np.unique(fragments_data)

    for fragment, mean in zip(
        fragment_ids, measurements.mean(average_affs, fragments_data, fragment_ids)
    ):
        if mean < filter_value:
            filtered_fragments.append(fragment)

    filtered_fragments = np.array(filtered_fragments, dtype=fragments_data.dtype)
    replace = np.zeros_like(filtered_fragments)
    replace_values(fragments_data, filtered_fragments, replace)


def get_seeds(
    boundary_distances,
    min_seed_distance=10,
):
    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered == boundary_distances

    seeds, n = label(maxima)

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64)

    return seeds

def compute_fragments(affs_data, sigma=None, bias=None, noise_eps=None, min_seed_distance=None, seed_eps=None, neighborhood=None, strides=None, randomized_strides=False):
    if sigma is not None:
        # add 0 for channel dim
        sigma = (0, *sigma)
    else:
        sigma = None

    # add some random noise to affs (this is particularly necessary if your affs are
    #  stored as uint8 or similar)
    # If you have many affinities of the exact same value the order they are processed
    # in may be fifo, so you can get annoying streaks.

    shift = np.zeros_like(affs_data)

    if noise_eps is not None:
        shift += np.random.randn(*affs_data.shape) * noise_eps

    #######################

    # add smoothed affs, to solve a similar issue to the random noise. We want to bias
    # towards processing the central regions of objects first.

    if sigma is not None:
        shift += gaussian_filter(affs_data, sigma=sigma) - affs_data

    #######################
    shift += np.array([bias]).reshape(
        (-1, *((1,) * (len(affs_data.shape) - 1)))
    )

    if min_seed_distance is not None:
        boundary_mask = np.mean(affs_data, axis=0) > 0.5
        boundary_distances = distance_transform_edt(boundary_mask)

        seeds = get_seeds(
            boundary_distances,
            min_seed_distance=min_seed_distance,
        ).astype(np.uint64)

        seeds[~boundary_mask] = 0

        if seed_eps is not None:
            D = distance_transform_edt(seeds == 0)
            shift -= seed_eps * D
    else:
        seeds = None

    fragments_data = mws.agglom(
        (affs_data + shift).astype(np.float64),
        offsets=neighborhood,
        strides=strides,
        seeds=seeds,
        randomized_strides=randomized_strides,
    )

    return fragments_data