import gunpowder as gp
import logging
import math
import numpy as np
import random
import torch
import zarr
from lsd.train.gp import AddLocalShapeDescriptor
from lsd.train import LsdExtractor
from funlib.learn.torch.models import UNet, ConvPass
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
    generate_binary_structure,
)
from skimage.measure import label as relabel

logging.basicConfig(level=logging.INFO)

torch.backends.cudnn.benchmark = True


# todo: clean up - move to functions file and move custom nodes to gp folder,
# have cleaner importing etc. should probably just make the pipeline a class
# that we can better control from the gui...


def calc_max_padding(output_size, voxel_size, sigma, mode="shrink"):

    method_padding = gp.Coordinate((sigma * 3,) * 3)

    diag = np.sqrt(output_size[1] ** 2 + output_size[2] ** 2)

    max_padding = gp.Roi(
        (
            gp.Coordinate([i / 2 for i in [output_size[0], diag, diag]])
            + method_padding
        ),
        (0,) * 3,
    ).snap_to_grid(voxel_size, mode=mode)

    return max_padding.get_begin()


class CreatePoints(gp.BatchFilter):
    def __init__(
        self,
        labels,
    ):

        self.labels = labels

    def process(self, batch, request):

        labels = batch[self.labels].data
        shape = labels.shape

        spec = batch[self.labels].spec

        # for efficiency
        num_points = 25

        for n in range(num_points):
            z = random.randint(1, labels.shape[0] - 1)
            y = random.randint(1, labels.shape[1] - 1)
            x = random.randint(1, labels.shape[2] - 1)

            labels[z, y, x] = 1

        batch[self.labels].data = labels


class DilatePoints(gp.BatchFilter):
    def __init__(self, labels, dilations=2):

        self.labels = labels

    def process(self, batch, request):

        labels = batch[self.labels].data

        struct = generate_binary_structure(2, 2)

        dilations = random.randint(1, 10)

        for z in range(labels.shape[0]):

            dilated = binary_dilation(
                labels[z], structure=struct, iterations=dilations
            )

            labels[z] = dilated.astype(labels.dtype)

        batch[self.labels].data = labels


class Relabel(gp.BatchFilter):
    def __init__(self, labels):

        self.labels = labels

    def process(self, batch, request):

        labels = batch[self.labels].data

        relabeled = relabel(labels, connectivity=1).astype(labels.dtype)

        batch[self.labels].data = relabeled


class ExpandLabels(gp.BatchFilter):
    def __init__(self, labels, background=0):
        self.labels = labels
        self.background = background

    def process(self, batch, request):

        labels_data = batch[self.labels].data
        distance = labels_data.shape[0]

        distances, indices = distance_transform_edt(
            labels_data == self.background, return_indices=True
        )

        expanded_labels = np.zeros_like(labels_data)

        dilate_mask = distances <= distance

        masked_indices = [
            dimension_indices[dilate_mask] for dimension_indices in indices
        ]

        nearest_labels = labels_data[tuple(masked_indices)]

        expanded_labels[dilate_mask] = nearest_labels

        batch[self.labels].data = expanded_labels


class ChangeBackground(gp.BatchFilter):
    def __init__(self, labels):

        self.labels = labels

    def process(self, batch, request):

        labels = batch[self.labels].data

        labels[labels == 0] = np.max(labels) + 1

        batch[self.labels].data = labels


class SmoothLSDs(gp.BatchFilter):
    def __init__(self, lsds):
        self.lsds = lsds

    def process(self, batch, request):

        lsds = batch[self.lsds].data

        sigma = random.uniform(0.5, 2.0)

        for z in range(lsds.shape[1]):
            lsds_sec = lsds[:, z]

            lsds[:, z] = np.array(
                [
                    gaussian_filter(lsds_sec[i], sigma=sigma)
                    for i in range(lsds_sec.shape[0])
                ]
            ).astype(lsds_sec.dtype)

        batch[self.lsds].data = lsds


class WeightedMSELoss(torch.nn.MSELoss):
    def __init__(self) -> None:
        super(WeightedMSELoss, self).__init__()

    def forward(self, prediction, target, weights):

        scaled = weights * (prediction - target) ** 2

        if len(torch.nonzero(scaled)) != 0:
            mask = torch.masked_select(scaled, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:
            loss = torch.mean(scaled)

        return loss


class CustomLSDs(AddLocalShapeDescriptor):
    def __init__(self, segmentation, descriptor, *args, **kwargs):

        super().__init__(segmentation, descriptor, *args, **kwargs)

        self.extractor = LsdExtractor(
            self.sigma[0:2], self.mode, self.downsample
        )

    def process(self, batch, request):

        labels = batch[self.segmentation].data

        spec = batch[self.segmentation].spec.copy()

        spec.dtype = np.float32

        descriptor = np.zeros(shape=(6, *labels.shape))

        for z in range(labels.shape[0]):
            labels_sec = labels[z]

            descriptor_sec = self.extractor.get_descriptors(
                segmentation=labels_sec, voxel_size=spec.voxel_size[1:]
            )

            descriptor[:, z] = descriptor_sec

        batch = gp.Batch()
        batch[self.descriptor] = gp.Array(descriptor.astype(spec.dtype), spec)

        return batch


def fake_lsds_pipeline(
    iterations, voxel_size, save_every, checkpoint_basename
):

    print("starting fake lsds...")

    zeros = gp.ArrayKey("ZEROS")
    gt_lsds = gp.ArrayKey("GT_LSDS")
    gt_affs = gp.ArrayKey("GT_AFFS")
    pred_affs = gp.ArrayKey("PRED_AFFS")
    affs_weights = gp.ArrayKey("AFFS_WEIGHTS")

    voxel_size = gp.Coordinate(voxel_size)

    input_shape = gp.Coordinate((10, 96, 96))
    output_shape = gp.Coordinate((4, 56, 56))

    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    request = gp.BatchRequest()

    request.add(zeros, input_size)
    request.add(gt_lsds, input_size)
    request.add(gt_affs, output_size)
    request.add(pred_affs, output_size)
    request.add(affs_weights, output_size)

    context = calc_max_padding(output_size, voxel_size, sigma=80)

    num_fmaps = 12

    # this lightweight net performs well and is almost as fast as the 2d
    # network. Could test different approaches though...
    ds_fact = [(2, 2, 2), (1, 2, 2)]

    ksd = [
        [(2, 3, 3), (2, 3, 3)],
        [(1, 3, 3), (1, 3, 3)],
        [(1, 3, 3), (1, 3, 3)],
    ]

    ksu = [
        [(1, 3, 3), (1, 3, 3)],
        [(2, 3, 3), (2, 3, 3)],
    ]

    unet = UNet(
        in_channels=6,
        num_fmaps=num_fmaps,
        fmap_inc_factor=3,
        downsample_factors=ds_fact,
        kernel_size_down=ksd,
        kernel_size_up=ksu,
        padding="valid",
        constant_upsample=True,
    )

    model = torch.nn.Sequential(
        unet,
        ConvPass(
            in_channels=num_fmaps,
            out_channels=3,
            kernel_sizes=[(1, 1, 1)],
            activation="Sigmoid",
        ),
    )

    neighborhood = [[0, 0, -1], [0, -1, 0], [-1, 0, 0]]

    loss = WeightedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    source = gp.ZarrSource(
        "test_data.zarr",
        {
            zeros: "zeros",  # just a zeros dataset, since we need a source
        },
        {
            zeros: gp.ArraySpec(interpolatable=False),
        },
    )

    source += gp.Pad(zeros, context * 2)

    pipeline = source

    # randomly sample some points and write them into our zeros array as ones
    pipeline += CreatePoints(zeros)

    # grow the boundaries
    pipeline += DilatePoints(zeros)

    # relabel connected components
    pipeline += Relabel(zeros)

    # expand the labels outwards into the background
    pipeline += ExpandLabels(zeros)

    # there will still be some background, change this to max id + 1
    pipeline += ChangeBackground(zeros)

    # relabel ccs again to deal with incorrectly connected background
    pipeline += Relabel(zeros)

    pipeline += gp.SimpleAugment(transpose_only=[1, 2])

    pipeline += gp.ElasticAugment(
        control_point_spacing=[8, 40, 40],
        jitter_sigma=[0, 2, 2],
        rotation_interval=[0, math.pi / 2.0],
        prob_slip=0.05,
        prob_shift=0.05,
        max_misalign=10,
        subsample=8,
    )

    # do this on non eroded labels - that is what predicted lsds will look like
    pipeline += CustomLSDs(zeros, gt_lsds, sigma=80, downsample=2)

    # smooth the batch by different sigmas to simulate noisy predictions
    pipeline += SmoothLSDs(gt_lsds)

    pipeline += gp.NoiseAugment(gt_lsds)

    # now we erode - we want the gt affs to have a pixel boundary
    pipeline += gp.GrowBoundary(zeros, steps=1, only_xy=True)

    pipeline += gp.AddAffinities(
        affinity_neighborhood=neighborhood,
        labels=zeros,
        affinities=gt_affs,
        dtype=np.float32,
    )

    pipeline += gp.BalanceLabels(gt_affs, affs_weights)

    pipeline += gp.Stack(1)

    pipeline += gp.PreCache(cache_size=40, num_workers=10)

    pipeline += gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs={"input": gt_lsds},
        loss_inputs={0: pred_affs, 1: gt_affs, 2: affs_weights},
        outputs={0: pred_affs},
        save_every=save_every,
        # checkpoint_basename='checkpoints/fake_lsds',
        checkpoint_basename=checkpoint_basename,
        log_dir="logs/fake_lsds_log",
    )

    pipeline += gp.Squeeze([gt_lsds, gt_affs, pred_affs])

    pipeline += gp.Snapshot(
        dataset_names={
            gt_lsds: "gt_lsds",
            gt_affs: "gt_affs",
            pred_affs: "pred_affs",
        },
        output_dir="snapshots/fake_lsds_snapshots",
        output_filename="batch_{iteration}.zarr",
        every=100,
    )

    # pipeline += gp.PrintProfilingStats(every=10)

    with gp.build(pipeline):
        for i in range(iterations):
            pipeline.request_batch(request)


if __name__ == "__main__":

    fake_lsds_pipeline(1000)
