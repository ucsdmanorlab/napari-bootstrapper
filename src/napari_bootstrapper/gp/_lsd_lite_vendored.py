"""Private copies of the lsd-lite LSD functions that Add2DLSDs uses.

lsd-lite 1.0.1 on PyPI computes each label over the whole section. The
copies below come from a later lsd-lite commit that computes each label
inside its bounding box, which gives the same values in less time.
"""

import functools
import logging
from collections.abc import Sequence

import numpy as np
from funlib.geometry import Coordinate
from numpy.lib.stride_tricks import as_strided
from scipy.ndimage import gaussian_filter

# The functions below are private copies of get_lsds and its helpers in
# lsd_lite/lsds.py of lsd-lite (https://github.com/vaxenburg/lsd-lite),
# commit 63540f1. The copies add a leading underscore to get_lsds,
# outer_product and upsample, and change nothing else. The lsd-lite
# license:
#
# MIT License
#
# Copyright (c) 2025 Funke lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

logger = logging.getLogger(__name__)


def _get_lsds(
    segmentation: np.ndarray,
    sigma: float | Sequence[float],
    voxel_size: Sequence[int] | None = None,
    labels: np.ndarray | Sequence[int] | None = None,
    downsample: int = 1,
    use_paper_ordering: bool = True,
):
    """
    Compute local shape descriptors for the given segmentation.

    Args:

        segmentation (``np.array`` of ``int``):

            A label array to compute the local shape descriptors for.

        sigma (``tuple`` of ``float``):

            The radius to consider for the local shape descriptor.

        voxel_size (``tuple`` of ``int``, optional):

            The voxel size of ``segmentation``. Defaults to 1.

        labels (array-like of ``int``, optional):

            Restrict the computation to the given labels. Defaults to all
            labels inside the ``roi`` of ``segmentation``.

        downsample (``int``, optional):

            Compute the local shape descriptor on a downsampled volume for
            faster processing. Defaults to 1 (no downsampling).
    """

    assert all([(s // downsample) * downsample == s for s in segmentation.shape]), (
        f"Segmentation shape {segmentation.shape} must be divisible by "
        f"downsample factor {downsample}."
    )

    dims = len(segmentation.shape)
    if isinstance(sigma, (int, float)):
        sigma = (sigma,) * dims

    assert len(sigma) == dims, (
        f"Sigma {sigma} must have the same number of dimensions as "
        f"segmentation. shape: {segmentation.shape}."
    )

    if voxel_size is None:
        voxel_size = Coordinate((1,) * dims)
    else:
        voxel_size = Coordinate(voxel_size)

    assert voxel_size.dims == dims, (
        f"Voxel size {voxel_size} must have the same number of dimensions as "
        f"segmentation. shape: {segmentation.shape}."
    )

    if labels is None:
        labels = np.unique(segmentation)

    # get sub-sampled shape, roi, voxel size and sigma
    df = downsample
    logger.debug("Downsampling segmentation %s with factor %f", segmentation.shape, df)

    sub_segmentation = segmentation[tuple(slice(None, None, df) for _ in range(dims))]

    sub_shape = sub_segmentation.shape
    sub_voxel_size = tuple(v * df for v in voxel_size)
    sub_sigma_voxel = tuple(s / v for s, v in zip(sigma, sub_voxel_size))

    grid = np.meshgrid(
        *[
            np.arange(0, sub_shape[dim] * sub_voxel_size[dim], sub_voxel_size[dim])
            for dim in range(dims)
        ],
        indexing="ij",
    )
    coords = np.array(grid, dtype=np.float32)

    # normalize stats
    # get max possible mean offset for normalization
    # farthest voxel in context is 3*sigma away, but due to Gaussian
    # weighting, sigma itself is probably a better upper bound
    max_distance = np.array([s for s in sigma], dtype=np.float32)

    # prepare full-res descriptor volumes for roi
    channels = 10 if dims == 3 else 6
    descriptors = np.zeros((channels,) + segmentation.shape, dtype=np.float32)

    aggregate = functools.partial(
        gaussian_filter,
        mode="constant",
        cval=0.0,
        truncate=3.0,
    )

    for label in labels:
        if label == 0:
            continue

        # Restrict the calculation to the label's bounding box, not the full volume.
        # This is exact: the descriptor is only written inside the label, and the
        # zero-padded Gaussians there see nothing but the label's own mask. The box
        # is rounded out to whole coarse voxels so it covers the label after upsampling.
        full_mask = segmentation == label
        bbox = _bounding_box(full_mask)
        if bbox is None:
            continue  # label not present in this block
        sub_box = tuple(slice(s.start // df, (s.stop - 1) // df + 1) for s in bbox)
        full_box = tuple(slice(s.start * df, s.stop * df) for s in sub_box)

        sub_mask: np.ndarray = sub_segmentation[sub_box] == label
        box_coords = coords[(slice(None),) + sub_box]
        masked_coords = box_coords * sub_mask

        # simply a mask convolved with a Gaussian
        mass = aggregate(
            sub_mask.astype(np.float32),
            sigma=sub_sigma_voxel,
        )
        mass_mask = mass == 0
        mass[mass_mask] = 1

        # offsets (meshgrid convolved with Gaussian, divided by mass, minus
        # meshgrid)
        center_of_mass = (
            np.array(
                [
                    aggregate(
                        masked_coords[d],
                        sigma=sub_sigma_voxel,
                    )
                    for d in range(dims)
                ]
            )
            / mass
        )
        mean_offset = center_of_mass - box_coords
        mean_offset = (
            mean_offset / max_distance.reshape((-1,) + (1,) * dims) * 0.5 + 0.5
        )

        # covariance
        coords_outer = _outer_product(masked_coords)
        center_of_mass_outer = _outer_product(center_of_mass)

        # get indices of upper triangle of covariance matrix
        rows, cols = np.triu_indices(dims)
        entries = (rows * dims + cols).tolist()

        # sort them s.t. the diagonal entries come first. the first `dims` are
        # the diagonals
        entries = sorted(
            entries, key=lambda x: x % (dims + 1) * (dims + 1) + x // (dims + 1)
        )

        covariance = (
            np.array([aggregate(coords_outer[d], sub_sigma_voxel) for d in entries])
            / mass
        )
        covariance -= center_of_mass_outer[entries]

        # normalize variance and pearson correlations
        variance = covariance[:dims]  # diagonals
        variance = np.maximum(variance, 1e-3)  # avoid division by zero

        # normalize pearson coefficients by axis variance
        # also divide by 2 and add 0.5 to get them into [0, 1]
        pearson = covariance[dims:]  # off-diagonals
        for ind, entry in enumerate(entries[dims:]):
            i, j = entry // dims, entry % dims
            pearson[ind] /= 2 * np.sqrt(variance[i] * variance[j])
            pearson[ind] += 0.5

        # normalize variance by axis sigma
        for d in range(dims):
            variance[d] /= sigma[d] ** 2

        # reset 0 mass regions (this was set to 1 above to avoid nans)
        mass = mass * ~mass_mask

        if dims == 3 and use_paper_ordering:
            # rearrange for backwards compatibility
            pearson = pearson[[0, 2, 1]]

        descriptor = np.concatenate((mean_offset, variance, pearson, mass[None, :]))
        descriptor = _upsample(descriptor, df)
        descriptors[(slice(None),) + full_box] += descriptor * full_mask[full_box][None, ...]

    np.clip(descriptors, 0.0, 1.0, out=descriptors)

    return descriptors


def _bounding_box(mask: np.ndarray) -> tuple[slice, ...] | None:
    """Tight slices around the ``True`` voxels of ``mask``; ``None`` if there are none."""

    slices = []
    for axis in range(mask.ndim):
        other_axes = tuple(a for a in range(mask.ndim) if a != axis)
        hit = np.flatnonzero(np.any(mask, axis=other_axes))
        if hit.size == 0:
            return None
        slices.append(slice(int(hit[0]), int(hit[-1]) + 1))
    return tuple(slices)


def _outer_product(array):
    """Computes the unique values of the outer products of the first dimension
    of ``array``. If ``array`` has shape ``(k, d, h, w)``, for example, the
    output will be of shape ``(k*(k+1)/2, d, h, w)``.
    """

    k = array.shape[0]
    outer = np.einsum("i...,j...->ij...", array, array)
    return outer.reshape((k**2,) + array.shape[1:])


def _upsample(array, f):
    shape = array.shape
    stride = array.strides

    if len(array.shape) == 4:
        sh = (shape[0], shape[1], f, shape[2], f, shape[3], f)
        st = (stride[0], stride[1], 0, stride[2], 0, stride[3], 0)
    else:
        sh = (shape[0], shape[1], f, shape[2], f)
        st = (stride[0], stride[1], 0, stride[2], 0)

    view = as_strided(array, sh, st)

    ll = [shape[0]]
    [ll.append(shape[i + 1] * f) for i, j in enumerate(shape[1:])]

    return view.reshape(ll)
