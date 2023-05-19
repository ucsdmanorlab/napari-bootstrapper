from .model import LsdModel

from funlib.persistence import prepare_ds

import json
import sys
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


def predict(
        raw_file,
        raw_dataset,
        checkpoint,
        voxel_size):

    section = int(raw_dataset.split('/')[-1])

    raw = gp.ArrayKey('RAW')
    pred_lsds = gp.ArrayKey('PRED_LSDS')

    model = LsdModel([6])
    model.eval()

    input_shape = gp.Coordinate((300, 300))
    output_shape = gp.Coordinate((208, 208))
    voxel_size = gp.Coordinate(voxel_size[1:])

    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    context = (input_size - output_size) / 2

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
            scan)

    predict_request = gp.BatchRequest()

    predict_request[raw] = total_input_roi
    predict_request[pred_lsds] = total_output_roi

    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    return batch[pred_lsds].data, total_output_roi


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(l, key=alphanum_key)


if __name__ == "__main__":

    raw_file = sys.argv[1]
    raw_ds = "image"
    voxel_size = [50, 8, 8] 
    checkpoint = sys.argv[2]
    
    sections = [x for x in os.listdir(os.path.join(raw_file,raw_ds)) if '.' not in x]
    sections = natural_sort(sections)

    full_lsds = []

    for section in sections: 
        raw_dataset = f'{raw_ds}/{section}'

        lsds, lsds_roi = predict(
            raw_file,
            raw_ds,
            checkpoint,
            voxel_size)

        full_lsds.append(lsds)

    full_lsds = (
        np.array(full_lsds).transpose((1, 0, 2, 3)).astype(np.float32)
    )

    # offset should be 3d for padding
    lsds_offset = [int(available_sections[0])*voxel_size[0]] + list(lsds_roi.offset)
    lsds_roi = Roi(Coordinate(lsds_offset),Coordinate(full_lsds.shape[1:])*voxel_size)

    # write lsds to zarr
    out_lsds = prepare_ds(
            raw_file,
            "lsds",
            lsds_roi,
            voxel_size,
            dtype=np.uint8,
            delete=True,
            num_channels=6)

    out_lsds[lsds_roi] = full_lsds
