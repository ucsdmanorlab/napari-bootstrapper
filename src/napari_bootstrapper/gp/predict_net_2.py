import json
import gunpowder as gp
import math
import numpy as np
import os
import sys
import torch
import logging
import zarr
import daisy
from funlib.persistence import prepare_ds
from funlib.learn.torch.models import UNet, ConvPass

neighborhood = [
        [-1,0,0],
        [0,-1,0],
        [0,0,-1]]

def fake_lsds_predict(
        input_lsds_file, input_lsds_dataset, out_file, out_ds, checkpoint, voxel_size, grow):

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

    model.eval()

    input_lsds = gp.ArrayKey('INPUT_LSDS')
    pred_affs = gp.ArrayKey('PRED_AFFS')

    voxel_size = gp.Coordinate(voxel_size)
    input_shape = gp.Coordinate((10, 96, 96))
    output_shape = gp.Coordinate((4, 56, 56))

    if grow:
        input_shape += gp.Coordinate((16, 96, 96))
        output_shape += gp.Coordinate((16, 96, 96))

    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    context = (input_size - output_size) // 2

    scan_request = gp.BatchRequest()

    scan_request.add(input_lsds, input_size)
    scan_request.add(pred_affs, output_size)

    source = gp.ZarrSource(
                input_lsds_file,
            {
                input_lsds: input_lsds_dataset
            },
            {
                input_lsds: gp.ArraySpec(interpolatable=True)
            })

    with gp.build(source):
        total_input_roi = source.spec[input_lsds].roi
        total_output_roi = source.spec[input_lsds].roi.grow(-context,-context)

    prepare_ds(
            out_file,
            out_ds,
            daisy.Roi(
                total_output_roi.get_offset(),
                total_output_roi.get_shape()
            ),
            voxel_size,
            np.uint8,
            write_size=output_size,
            compressor={"id":"blosc","clevel":3},
            delete=True,
            num_channels=len(neighborhood))

    predict = gp.torch.Predict(
            model,
            checkpoint=checkpoint,
            inputs = {
                'input': input_lsds
            },
            outputs = {
                0: pred_affs,
            })

    scan = gp.Scan(scan_request)

    write = gp.ZarrWrite(
            dataset_names={
                pred_affs: out_ds,
            },
            store=out_file)

    pipeline = (
            source +
            gp.Normalize(input_lsds) +
            gp.Pad(input_lsds, None) +
            gp.Unsqueeze([input_lsds]) +
            predict +
            gp.Squeeze([pred_affs]) +
            gp.IntensityScaleShift(pred_affs,255,0) +
            write+
            scan)

    predict_request = gp.BatchRequest()

    predict_request[input_lsds] = total_input_roi
    predict_request[pred_affs] = total_output_roi

    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    return batch[pred_affs].data, total_output_roi
