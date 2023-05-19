import json
import re
import shutil
import zarr
import glob
import gunpowder as gp
import math
import numpy as np
import os
import sys
import torch
import daisy
from funlib.persistence import prepare_ds
from funlib.learn.torch.models import UNet, ConvPass


def lsd_outpaint_predict(
        raw_file,
        raw_dataset,
        checkpoint,
        voxel_size):

    section = int(raw_dataset.split('/')[-1])

    raw = gp.ArrayKey('RAW')
    pred_lsds = gp.ArrayKey('PRED_LSDS')
 
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
    model.eval()
    
    input_shape = gp.Coordinate((300, 300))
    output_shape = gp.Coordinate((208, 208))
    voxel_size = gp.Coordinate(voxel_size[1:])

    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    context = (input_size - output_size) // 2

    scan_request = gp.BatchRequest()

    scan_request.add(raw, input_size)
    scan_request.add(pred_lsds, output_size)

    source = gp.ZarrSource(
                raw_file,
            {
                raw: raw_dataset
            },
            {
                raw: gp.ArraySpec(interpolatable=True)
            })

    with gp.build(source):
        total_input_roi = source.spec[raw].roi
        total_output_roi = source.spec[raw].roi.grow(-context,-context)

    predict = gp.torch.Predict(
            model,
            checkpoint=checkpoint,
            inputs = {
                'input': raw
            },
            outputs = {
                0: pred_lsds,
            })

    scan = gp.Scan(scan_request)

    write = gp.ZarrWrite(
            dataset_names={
                pred_lsds: lsds_out_ds,
            },
            store=out_file)

    pipeline = (
            source +
            gp.Normalize(raw) +
            gp.Pad(raw, None) +
            gp.IntensityScaleShift(raw, 2,-1) +
            gp.Unsqueeze([raw]) +
            gp.Unsqueeze([raw]) +
            predict +
            gp.Squeeze([pred_lsds]) +
            gp.IntensityScaleShift(pred_lsds, 255, 0) +
            write+
            scan)

    predict_request = gp.BatchRequest()

    predict_request[raw] = total_input_roi
    predict_request[pred_lsds] = total_output_roi

    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    return batch[pred_lsds].data, total_output_roi
