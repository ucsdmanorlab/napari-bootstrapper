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


def lsd_outpaint_predict(raw_file, raw_dataset, checkpoint):

    raw = gp.ArrayKey("RAW")
    pred_lsds = gp.ArrayKey("PRED_LSDS")

    voxel_size = gp.Coordinate((8, 8))

    g = gp.Coordinate((96, 96))

    input_shape = gp.Coordinate((300, 300)) + g
    output_shape = gp.Coordinate((208, 208)) + g

    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    context = (input_size - output_size) / 2

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

    scan_request = gp.BatchRequest()

    scan_request.add(raw, input_size)
    scan_request.add(pred_lsds, output_size)

    source = gp.ZarrSource(
        raw_file, {raw: raw_dataset}, {raw: gp.ArraySpec(interpolatable=True)}
    )

    with gp.build(source):
        total_input_roi = source.spec[raw].roi
        total_output_roi = total_input_roi.grow(-context, -context)

    predict = gp.torch.Predict(
        model,
        checkpoint=f"checkpoints/lsd_outpainting_checkpoint_{str(checkpoint)}",
        inputs={"input": raw},
        outputs={
            0: pred_lsds,
        },
    )

    scan = gp.Scan(scan_request)

    pipeline = (
        source
        + gp.Normalize(raw)
        + gp.Unsqueeze([raw])
        + gp.Unsqueeze([raw])
        + predict
        + gp.Squeeze([pred_lsds])
        + scan
    )

    predict_request = gp.BatchRequest()

    predict_request[raw] = total_input_roi
    predict_request[pred_lsds] = total_output_roi

    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    return batch[pred_lsds].data, total_output_roi.get_begin()


if __name__ == "__main__":

    raw_file = "test_data.zarr"

    full_lsds = []

    for z in range(100):
        raw_dataset = f"full_raw_2d/{z}"

        lsds, offset = lsd_outpaint_predict(raw_file, raw_dataset)
        full_lsds.append(lsds)

    full_lsds = np.array(full_lsds).transpose((1, 0, 2, 3)).astype(np.float32)

    factor = (1, 1, 2, 2)

    out_file = zarr.open("test_prediction.zarr", "a")

    out_file["lsds"] = full_lsds
    out_file["lsds"].attrs["offset"] = [0] * 3
    out_file["lsds"].attrs["resolution"] = [40, 8, 8]
