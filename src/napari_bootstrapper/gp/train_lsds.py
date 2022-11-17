import gunpowder as gp
import logging
import math
import numpy as np
import random
import zarr
import skimage.draw
from lsd.train.gp import AddLocalShapeDescriptor

import torch
from funlib.learn.torch.models import UNet, ConvPass

logging.basicConfig(level=logging.INFO)


class WeightedMSELoss(torch.nn.MSELoss):
    def __init__(self) -> None:
        super(WeightedMSELoss,self).__init__()

    def forward(self, prediction, target, weights):

        scaled = weights*(prediction-target)**2

        if len(torch.nonzero(scaled)) !=0:
            mask = torch.masked_select(scaled,torch.gt(weights,0))
            loss = torch.mean(mask)

        else:
            loss = torch.mean(scaled)

        return loss

def pipeline(iterations):

    raw = gp.ArrayKey('RAW')
    labels = gp.ArrayKey('LABELS')
    gt_lsds = gp.ArrayKey('GT_LSDS')
    gt_lsds_mask = gp.ArrayKey('GT_LSDS_MASK')
    unlabelled = gp.ArrayKey('UNLABELLED')
    pred_lsds = gp.ArrayKey('PRED_LSDS')

    voxel_size = gp.Coordinate((4, 4))

    input_shape = gp.Coordinate((300, 300))
    output_shape = gp.Coordinate((208, 208))

    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    request = gp.BatchRequest()

    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(gt_lsds, output_size)
    request.add(gt_lsds_mask, output_size)
    request.add(unlabelled, output_size)
    request.add(pred_lsds, output_size)

    num_fmaps = 12

    ds_fact = [(2,2),(2,2),(2,2)]
    num_levels = len(ds_fact) + 1
    ksd = [[(3,3), (3,3)]]*num_levels
    ksu = [[(3,3), (3,3)]]*(num_levels - 1)

    unet = UNet(
      in_channels=1,
      num_fmaps=num_fmaps,
      fmap_inc_factor=5,
      downsample_factors=ds_fact,
      kernel_size_down=ksd,
      kernel_size_up=ksu,
      padding='valid',
      constant_upsample=True)

    model = torch.nn.Sequential(
      unet,
      ConvPass(in_channels=num_fmaps, out_channels = 6, kernel_sizes = [(1,1)], activation='Sigmoid'))

    loss = WeightedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

    source = gp.ZarrSource(
    '../training_data/test.zarr',
    {
        raw: 'image',
        labels: 'labels',
        unlabelled: 'unlabelled'
    },
    {
        raw: gp.ArraySpec(interpolatable=True),
        labels: gp.ArraySpec(interpolatable=False),
        unlabelled: gp.ArraySpec(interpolatable=False)
    })

    source += gp.Normalize(raw)

    source += gp.RandomLocation(mask=unlabelled, min_masked=0.1)

    pipeline = source

    pipeline += gp.SimpleAugment()

    pipeline += gp.ElasticAugment(
        control_point_spacing=(64, 64),
        jitter_sigma=(5.0, 5.0),
        rotation_interval=(0, math.pi/2))

    pipeline += gp.IntensityAugment(
        raw,
        scale_min=0.9,
        scale_max=1.1,
        shift_min=-0.1,
        shift_max=0.1)

    pipeline += AddLocalShapeDescriptor(
        labels,
        gt_lsds,
        sigma=80,
        lsds_mask=gt_lsds_mask,
        unlabelled=unlabelled,
        downsample=2)

    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(10)

    pipeline += gp.PreCache(
            cache_size = 40,
            num_workers= 10
            )

    pipeline += gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs = {
            'input': raw
        },
        loss_inputs = {
            0: pred_lsds,
            1: gt_lsds,
            2: gt_lsds_mask
        },
        outputs = {
            0: pred_lsds
        },
        save_every = 500,
        log_dir = 'log'
    )

    pipeline += gp.Snapshot(
        dataset_names={
            raw: 'raw',
            gt_lsds: 'gt_lsds',
            pred_lsds: 'pred_lsds'
        },
        output_filename='batch_{iteration}.zarr',
        every=500
    )

    with gp.build(pipeline):
        for i in range(iterations):
            pipeline.request_batch(request)


if __name__ == '__main__':

    pipeline(3000)
