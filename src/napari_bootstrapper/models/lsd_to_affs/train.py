from .model import AffsModel, WeightedMSELoss

from autoseg.utils import ZerosSource, CreateLabels, SmoothArray, CustomLSDs
from lsd.train.gp import AddLocalShapeDescriptor
import gunpowder as gp

import os
import json
import logging
import math
import numpy as np
import random
import torch
import zarr

logging.basicConfig(level=logging.INFO)

torch.backends.cudnn.benchmark = True


def train(
        iterations,
        voxel_size,
        save_every,
        checkpoint_basename,
        num_workers):

    model = AffsModel()
    model.train()

    loss = WeightedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-4)

    zeros = gp.ArrayKey("ZEROS")
    gt_lsds = gp.ArrayKey("GT_LSDS")
    gt_affs = gp.ArrayKey("GT_AFFS")
    pred_affs = gp.ArrayKey("PRED_AFFS")
    affs_weights = gp.ArrayKey("AFFS_WEIGHTS")

    voxel_size = gp.Coordinate(voxel_size)
    anisotropy = int((voxel_size[0] / voxel_size[1]) - 1) # 0 is isotropic
    sigma = int(10*voxel_size[-1])

    input_shape = (10, 148, 148)
    output_shape = (6, 108, 108)

    input_shape = gp.Coordinate(input_shape)
    output_shape = gp.Coordinate(output_shape)
    
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
        zeros, gt_lsds, sigma=sigma, downsample=2
    )

    # add random noise
    pipeline += gp.NoiseAugment(gt_lsds)
    
    pipeline += gp.IntensityAugment(gt_lsds, 0.9, 1.1, -0.1, 0.1)

    # smooth the batch by different sigmas to simulate noisy predictions
    pipeline += SmoothArray(gt_lsds, (0.0, 0.1))

    # now we erode - we want the gt affs to have a pixel boundary
    pipeline += gp.GrowBoundary(zeros, steps=1, only_xy=bool(anisotropy))

    pipeline += gp.AddAffinities(
        affinity_neighborhood=[
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ],
        labels=zeros,
        affinities=gt_affs,
        dtype=np.float32,
    )

    pipeline += gp.BalanceLabels(gt_affs, affs_weights)

    pipeline += gp.Stack(1)

    pipeline += gp.PreCache(num_workers=num_workers)

    pipeline += gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs={"input": gt_lsds},
        loss_inputs={0: pred_affs, 1: gt_affs, 2: affs_weights},
        outputs={0: pred_affs},
        save_every=save_every,
        #log_dir=os.path.join(setup_dir,'log'),
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
        output_dir=os.path.join('snapshots/lsd_to_affs_snapshots'),
        every=1000,
    )

    with gp.build(pipeline):
        for i in range(iterations):
            batch = pipeline.request_batch(request)
            print(f"Train affs: iteration={batch.iteration} loss={batch.loss}")

        print("Training affinities complete!")

if __name__ == "__main__":

    train(
        501,
        [50,8,8],
        500,
        "lsd_to_affs",
        10)
