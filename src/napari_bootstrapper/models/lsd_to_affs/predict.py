from .model import AffsModel
from funlib.persistence import prepare_ds
from funlib.geometry import Roi, Coordinate
import gunpowder as gp

import json
import math
import numpy as np
import os
import sys
import torch
import logging
import zarr


def predict(
        input_lsds_file, input_lsds_dataset, out_file, out_ds, checkpoint, voxel_size, grow):

    model = AffsModel() 
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

    for i in range(len(voxel_size)):
        assert total_output_roi.get_shape()[i]/voxel_size[i] >= output_shape[i], \
            f"total output (write) ROI cannot be smaller than model's output shape, \noffending index: {i}\ntotal_output_roi: {total_output_roi.get_shape()}, \noutput_shape: {output_shape}, \nvoxel size: {voxel_size}" 
 
    prepare_ds(
            out_file,
            out_ds,
            Roi(
                total_output_roi.get_offset(),
                total_output_roi.get_shape()
            ),
            voxel_size,
            np.uint8,
            write_size=output_size,
            compressor={"id":"blosc","clevel":3},
            delete=True,
            num_channels=3)

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


if __name__ == "__main__":

    input_lsds_file = sys.argv[1]
    input_lsds_dataset = "lsds"
    out_file = sys.argv[1]
    out_ds = "affs"
    checkpoint = sys.argv[2]
    voxel_size = [50, 8, 8]
    grow = True

    predict(
        input_lsds_file, input_lsds_dataset, out_file, out_ds, checkpoint, voxel_size, grow)
