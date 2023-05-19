import os
import json
import math
import logging
import numpy as np
import gunpowder as gp
import torch
from lsd.train.gp import AddLocalShapeDescriptor
from funlib.learn.torch.models import UNet, ConvPass

logging.basicConfig(level=logging.INFO)

neighborhood = [[-1,0],[0,-1]]



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


def lsd_outpainting_pipeline(
        zarr_container,
        raw_ds,
        labels_ds,
        unlabelled_ds,
        max_iteration,
        voxel_size,
        min_masked,
        save_every,
        batch_size,
        checkpoint_basename):

    available_sections = [x for x in os.listdir(os.path.join(zarr_container,labels_ds)) if '.' not in x]
    print(f"Available sections to train on: {available_sections}")

    num_fmaps = 12

    ds_fact = [(2, 2), (2, 2), (2, 2)]
    num_levels = len(ds_fact) + 1
    ksd = [[(3, 3), (3, 3)]] * num_levels
    ksu = [[(3, 3), (3, 3)]] * (num_levels - 1)

    unet = UNet(
        in_channels=1,
        num_fmaps=num_fmaps,
        fmap_inc_factor=5,
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
            out_channels=6,
            kernel_sizes=[(1, 1)],
            activation="Sigmoid",
        ),
    )

    loss = WeightedMSELoss()

    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.5e-4)

    input_shape = gp.Coordinate((196,196))
    output_shape = gp.Coordinate((104,104))

    voxel_size = gp.Coordinate(voxel_size)
    output_size = output_shape * voxel_size
    input_size = input_shape * voxel_size

    context = (input_size - output_size) // 2

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
        gp.Pad(labels, context) +
        gp.Pad(unlabelled, context) +
        gp.RandomLocation(mask=unlabelled,min_masked=min_masked)
        for i in available_sections
    )

    pipeline = source
    pipeline += gp.RandomProvider()

    pipeline += gp.ElasticAugment(
        control_point_spacing=(50,) * 2,
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
            sigma=int(10*voxel_size[-1]),
            downsample=2)

    pipeline += gp.IntensityScaleShift(raw, 2,-1)

    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(batch_size)

    pipeline += gp.PreCache(num_workers=10, cache_size=40)

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
            pipeline.request_batch(request)
