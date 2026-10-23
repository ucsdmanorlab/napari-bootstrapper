import gunpowder as gp
import numpy as np

from ._lsd_lite_vendored import _get_lsds


def compute_lsds(segmentation, sigma, voxel_size, downsample, crop):
    descriptor = _get_lsds(
        _pad_high_end_to_downsample_multiple(segmentation, downsample),
        sigma,
        voxel_size,
        labels=np.unique(segmentation[crop]),
        downsample=downsample,
    )[(slice(None),) + crop]
    _set_old_lsd_mass_placeholder(descriptor, segmentation[crop])
    return descriptor


def _pad_high_end_to_downsample_multiple(segmentation, downsample):
    return np.pad(
        segmentation, [(0, -s % downsample) for s in segmentation.shape]
    )


def _set_old_lsd_mass_placeholder(descriptor, segmentation):
    descriptor[-1][(segmentation != 0) & (descriptor[-1] == 0)] = 1.0


class Add2DLSDs(gp.BatchFilter):
    """Compute six Gaussian LSD channels per section, with sigma in world units.

    The loss mask excludes background voxels and voxels where labels_mask or
    unlabelled is zero.
    """

    def __init__(
        self,
        segmentation,
        descriptor,
        lsds_mask=None,
        labels_mask=None,
        unlabelled=None,
        sigma=5.0,
        downsample=1,
    ):
        self.segmentation = segmentation
        self.descriptor = descriptor
        self.lsds_mask = lsds_mask
        self.labels_mask = labels_mask
        self.unlabelled = unlabelled
        self.sigma = (
            tuple(sigma)
            if isinstance(sigma, (tuple, list))
            else (0, sigma, sigma)
        )
        self.downsample = downsample

    def setup(self):
        spec = self.spec[self.segmentation].copy()
        spec.dtype = np.float32
        self.voxel_size = spec.voxel_size
        self.context = tuple(s * 3 for s in self.sigma)
        self.provides(self.descriptor, spec)
        if self.lsds_mask:
            self.provides(self.lsds_mask, spec.copy())

    def prepare(self, request):
        deps = gp.BatchRequest()
        if self.descriptor not in request:
            return deps

        context_roi = request[self.descriptor].roi.grow(
            self.context, self.context
        )
        context_roi = context_roi.snap_to_grid(self.voxel_size, mode="shrink")
        if self.segmentation in request:
            context_roi = context_roi.union(request[self.segmentation].roi)
        deps[self.segmentation] = request[self.descriptor].copy()
        deps[self.segmentation].roi = context_roi
        for mask in (self.labels_mask, self.unlabelled):
            if mask:
                deps[mask] = deps[self.segmentation].copy()
        return deps

    def process(self, batch, request):
        if self.descriptor not in request:
            return

        segmentation = batch[self.segmentation]
        descriptor_roi = request[self.descriptor].roi
        crop = (
            (descriptor_roi - segmentation.spec.roi.offset) / self.voxel_size
        ).to_slices()
        descriptor = self._descriptor(segmentation.data, crop)
        spec = self.spec[self.descriptor].copy()
        spec.roi = descriptor_roi.copy()
        outputs = gp.Batch()
        outputs[self.descriptor] = gp.Array(descriptor, spec)

        if self.lsds_mask and self.lsds_mask in request:
            mask = (segmentation.data[crop] != 0).astype(np.float32)
            for key in (self.labels_mask, self.unlabelled):
                if key:
                    mask *= batch[key].data[crop]
            mask = np.stack([mask] * descriptor.shape[0]).astype(np.float32)
            outputs[self.lsds_mask] = gp.Array(mask, spec.copy())
        return outputs

    def _descriptor(self, segmentation, crop):
        return np.stack(
            [
                compute_lsds(
                    segmentation[z],
                    self.sigma[1:],
                    self.voxel_size[1:],
                    self.downsample,
                    crop[1:],
                )
                for z in range(crop[0].start, crop[0].stop)
            ],
            axis=1,
        )
