import gunpowder as gp
import logging
import math
import numpy as np
import os
import sys
import torch
import zarr
from funlib.learn.torch.models import UNet, ConvPass

logging.basicConfig(level=logging.INFO)

def predict(
        raw_file,
        raw_dataset,
        out_file):

    raw = gp.ArrayKey('RAW')
    pred_lsds = gp.ArrayKey('PRED_LSDS')

    voxel_size = gp.Coordinate((4, 4))

    input_shape = gp.Coordinate((300, 300))
    output_shape = gp.Coordinate((208, 208))

    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    context = (input_size - output_size) / 2

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

    model.eval()

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
        total_output_roi = total_input_roi.grow(-context, -context)

    f = zarr.open(out_file, 'a')

    ds = f.create_dataset(
            'pred_lsds',
            shape=[6]+[i/j for i, j in zip(total_output_roi.get_shape(), voxel_size)]
            )
    ds.attrs['resolution'] = voxel_size
    ds.attrs['offset'] = total_output_roi.get_offset()

    predict = gp.torch.Predict(
        model,
        checkpoint=f'model_checkpoint_3000',
        inputs = {
            'input': raw
        },
        outputs = {
            0: pred_lsds,
        })

    scan = gp.Scan(scan_request)

    write = gp.ZarrWrite(
            dataset_names={
                pred_lsds: 'pred_lsds',
            },
            output_filename=out_file)

    pipeline = (
            source +
            # gp.Pad(raw, context) +
            gp.Normalize(raw) +
            gp.Unsqueeze([raw]) +
            gp.Unsqueeze([raw]) +
            predict +
            gp.Squeeze([pred_lsds]) +
            write+
            scan)

    predict_request = gp.BatchRequest()

    predict_request[raw] = total_input_roi
    predict_request[pred_lsds] = total_output_roi

    with gp.build(pipeline):
        pipeline.request_batch(predict_request)

if __name__ == "__main__":

    raw_file = '../training_data/test.zarr'
    raw_dataset = 'image'
    out_file = 'test_prediction.zarr'

    predict(raw_file, raw_dataset, out_file)
