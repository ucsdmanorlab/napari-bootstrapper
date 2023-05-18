import numpy as np
import waterz
from scipy.ndimage import label, \
        maximum_filter, \
        measurements, \
        distance_transform_edt, \
        gaussian_filter
from skimage.filters import median, threshold_otsu
from skimage.io import imread, imsave
from skimage.segmentation import watershed
import zarr

def watershed_from_boundary_distance(
        boundary_distances,
        boundary_mask,
        id_offset=0,
        min_seed_distance=10):

    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered==boundary_distances
    seeds, n = label(maxima)

    print(f"Found {n} fragments")

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds!=0] += id_offset

    fragments = watershed(
        boundary_distances.max() - boundary_distances,
        seeds,
        #mask=boundary_mask
    )

    fragments.astype(np.uint64)

    return fragments


def watershed_from_lsds(
        lsds,
        max_affinity_value=1.0,
        min_seed_distance=10):

    y_distances = np.linalg.norm(lsds[:,1:] - lsds[:,:-1], axis=0)
    x_distances = np.linalg.norm(lsds[:,:,1:] - lsds[:,:,:-1], axis=0)

    shape = lsds.shape[1]-1

    y = maximum_filter(y_distances[:,0:shape], footprint=np.ones([2,2]))
    x = maximum_filter(x_distances[0:shape,:], footprint=np.ones([2,2]))

    affs = 1 - np.stack([y, x, np.zeros([shape, shape])])

    affs = (affs - affs.min()) / (affs.max() - affs.min())

    affs = gaussian_filter(affs, sigma=1)

    boundary_mask = np.mean(affs, axis=0)>threshold_otsu(affs)
    boundary_distances = distance_transform_edt(boundary_mask)

    fragments = watershed_from_boundary_distance(
            boundary_distances,
            boundary_mask,
            min_seed_distance=min_seed_distance).astype(np.uint64)

    return affs, fragments, boundary_mask, boundary_distances


def get_segmentation(affinities, frags, threshold, labels_mask=None):

    thresholds = [threshold]

    test = frags.copy()

    generator = waterz.agglomerate(
        affs=affinities.astype(np.float32),
        fragments=test,
        thresholds=thresholds
    )

    seg = next(generator)

    return seg
