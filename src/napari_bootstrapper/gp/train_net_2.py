import gunpowder as gp
import os
import json
import logging
import math
import numpy as np
import random
import torch
import zarr
from lsd.train.gp import AddLocalShapeDescriptor

from funlib.learn.torch.models import UNet, ConvPass
from autoseg.utils import ZerosSource, CreateLabels, SmoothArray, CustomLSDs

logging.basicConfig(level=logging.INFO)

torch.backends.cudnn.benchmark = True


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


def fake_lsds_pipeline(
        iterations,
        voxel_size,
        save_every,
        checkpoint_basename):

    num_fmaps = 12

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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-4)

    zeros = gp.ArrayKey("ZEROS")
    gt_lsds = gp.ArrayKey("GT_LSDS")
    gt_affs = gp.ArrayKey("GT_AFFS")
    pred_affs = gp.ArrayKey("PRED_AFFS")
    affs_weights = gp.ArrayKey("AFFS_WEIGHTS")

    voxel_size = gp.Coordinate(voxel_size)
    anisotropy = int((voxel_size[0] / voxel_size[1]) - 1) # 0 is isotropic

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

    source = ZerosSource(
        {
            zeros: "zeros",  # just a zeros dataset, since we need a source
        },
        shape=input_shape,
        array_specs={
            zeros: gp.ArraySpec(interpolatable=False, voxel_size=voxel_size),
        },
    )

    source += gp.Pad(zeros, None)

    pipeline = source

    pipeline += CreateLabels(zeros,anisotropy)

    pipeline += gp.SimpleAugment(transpose_only=[1, 2])

    pipeline += gp.ElasticAugment(
        control_point_spacing=[voxel_size[1], voxel_size[0], voxel_size[0]],
        jitter_sigma=[2*int(not bool(anisotropy)), 2, 2],
        scale_interval=(0.75,1.25),
        rotation_interval=[0,math.pi/2.0],
        subsample=4,
    )

    # do this on non eroded labels - that is what predicted lsds will look like
    pipeline += CustomLSDs(
        zeros, gt_lsds, sigma=int(10*voxel_size[-1]), downsample=2
    )

    # add random noise
    pipeline += gp.NoiseAugment(gt_lsds)
    
    pipeline += gp.IntensityAugment(gt_lsds, 0.9, 1.1, -0.1, 0.1)

    # smooth the batch by different sigmas to simulate noisy predictions
    pipeline += SmoothArray(gt_lsds,(0.0,1.0))

    # now we erode - we want the gt affs to have a pixel boundary
    pipeline += gp.GrowBoundary(zeros, steps=1, only_xy=bool(anisotropy))

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
        save_every=1000,
        checkpoint_basename=checkpoint_basename
    )

    pipeline += gp.Squeeze([gt_lsds,gt_affs,pred_affs])
    
    pipeline += gp.Snapshot(
        dataset_names={
            zeros: "labels",
            gt_lsds: "gt_lsds",
            gt_affs: "gt_affs",
            pred_affs: "pred_affs",
        },
        output_filename="batch_{iteration}.zarr",
        output_dir="snapshots/fake_lsds_snapshots",
        every=save_every,
    )

    with gp.build(pipeline):
        for i in range(iterations):
            pipeline.request_batch(request)
