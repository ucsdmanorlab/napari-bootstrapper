from .model import LsdModel, WeightedMSELoss

from lsd.train.gp import AddLocalShapeDescriptor

import os
import sys
import json
import math
import logging
import numpy as np
import gunpowder as gp
import torch
from scipy.ndimage import gaussian_filter


logging.basicConfig(level=logging.INFO)


def calc_max_padding(output_size, voxel_size, sigma, mode="shrink"):

    method_padding = gp.Coordinate((sigma * 2,) * 2)

    diag = np.sqrt(output_size[0] ** 2 + output_size[1] ** 2)

    max_padding = gp.Roi(
        (
            gp.Coordinate([i / 2 for i in [diag, diag]])
            + method_padding
        ),
        (0,) * 2,
    ).snap_to_grid(voxel_size, mode=mode)

    return max_padding.get_begin()


def train(
        zarr_container,
        raw_ds,
        labels_ds,
        unlabelled_ds,
        max_iteration,
        voxel_size,
        min_masked,
        save_every,
        batch_size,
        num_workers,
        checkpoint_basename):

    model = LsdModel()
    model.train()

    loss = WeightedMSELoss()
    
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.5e-4)

    input_shape = (196, 196)
    output_shape = (104, 104)

    input_shape = gp.Coordinate(input_shape)
    output_shape = gp.Coordinate(output_shape)

    voxel_size = gp.Coordinate(voxel_size)
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size
    sigma = int(10*voxel_size[-1])
    labels_padding = calc_max_padding(output_size, voxel_size, sigma)

    raw = gp.ArrayKey('RAW')
    labels = gp.ArrayKey('LABELS')
    unlabelled = gp.ArrayKey('UNLABELLED')

    gt_lsds = gp.ArrayKey('GT_LSDS')
    pred_lsds = gp.ArrayKey('PRED_LSDS')
    lsds_weights = gp.ArrayKey('LSDS_WEIGHTS')

    request = gp.BatchRequest()

    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(unlabelled, output_size)
    request.add(gt_lsds, output_size)
    request.add(pred_lsds, output_size)
    request.add(lsds_weights, output_size)

    available_sections = [x for x in os.listdir(os.path.join(zarr_container,labels_ds)) if '.' not in x]
    print(f"Available sections to train on: {available_sections}")

    source = tuple(
        gp.ZarrSource(
            zarr_container,
            {
                raw: os.path.join(raw_ds,str(i)),
                labels: os.path.join(labels_ds,str(i)),
                unlabelled: os.path.join(unlabelled_ds,str(i)),
            },
            {
                raw: gp.ArraySpec(interpolatable=True),
                labels: gp.ArraySpec(interpolatable=False),
                unlabelled: gp.ArraySpec(interpolatable=False),
            }) +
        gp.Normalize(raw) +
        gp.Pad(raw, None) +
        gp.Pad(labels, labels_padding) +
        gp.Pad(unlabelled, labels_padding) +
        gp.RandomLocation(mask=unlabelled,min_masked=min_masked)
        for i in available_sections
    )

    pipeline = source
    pipeline += gp.RandomProvider()

    pipeline += gp.ElasticAugment(
        control_point_spacing=(voxel_size[0] * 5,) * 2,
        jitter_sigma=(2,2),
        scale_interval=(0.75,1.25),
        rotation_interval=[0,math.pi/2.0],
        subsample=4)

    pipeline += gp.SimpleAugment()

    pipeline += gp.NoiseAugment(raw)

    pipeline += gp.IntensityAugment(
        raw,
        scale_min=0.9,
        scale_max=1.1,
        shift_min=-0.1,
        shift_max=0.1)

    pipeline += AddLocalShapeDescriptor(
            labels,
            gt_lsds,
            unlabelled=unlabelled,
            lsds_mask=lsds_weights,
            sigma=sigma,
            downsample=1)

    pipeline += gp.IntensityScaleShift(raw, 2,-1)

    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(batch_size)

    pipeline += gp.PreCache(num_workers=num_workers)

    pipeline += gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs={
            'input': raw
        },
        outputs={
            0: pred_lsds,
        },
        loss_inputs={
            0: pred_lsds,
            1: gt_lsds,
            2: lsds_weights,
        },
        save_every=save_every,
        #log_dir=os.path.join(setup_dir,'log'),
        checkpoint_basename=checkpoint_basename)

    pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)

    pipeline += gp.Snapshot(
            dataset_names={
                raw: 'raw',
                gt_lsds: 'gt_lsds',
                pred_lsds: 'pred_lsds',
                lsds_weights: 'lsds_weights',
            },
            output_filename='batch_{iteration}.zarr',
            output_dir=os.path.join('snapshots/lsd_outpainting_snapshots'),
            every=save_every
    )

    with gp.build(pipeline):
        for i in range(max_iteration):
            batch = pipeline.request_batch(request)
            print(f"Training lsds: iteration={batch.iteration} loss={batch.loss}")

        print("Training lsds complete!")


if __name__ == "__main__":

    train(
        sys.argv[1],
        "image",
        "labels",
        "unlabelled",
        501,
        [8,8],
        0.1,
        500,
        5,
        10,
        "lsd_outpainting")
