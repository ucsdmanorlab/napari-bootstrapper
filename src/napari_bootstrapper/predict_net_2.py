import gunpowder as gp
import logging
import math
import numpy as np
import os
import shutil
import sys
import torch
import zarr
from funlib.learn.torch.models import UNet, ConvPass

logging.basicConfig(level=logging.INFO)


def fake_lsds_predict(
    lsds_file, lsds_dataset, out_file, out_dataset, checkpoint, grow=False
):

    pred_lsds = gp.ArrayKey("PRED_LSDS")
    pred_affs = gp.ArrayKey("PRED_AFFS")

    voxel_size = gp.Coordinate((40, 8, 8))

    input_shape = gp.Coordinate((10, 96, 96))
    output_shape = gp.Coordinate((4, 56, 56))

    if grow:
        input_shape += gp.Coordinate((30, 96, 96))
        output_shape += gp.Coordinate((30, 96, 96))

    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    context = (input_size - output_size) / 2

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

    scan_request = gp.BatchRequest()

    scan_request.add(pred_lsds, input_size)
    scan_request.add(pred_affs, output_size)

    source = gp.ZarrSource(
        lsds_file,
        {pred_lsds: lsds_dataset},
        {pred_lsds: gp.ArraySpec(interpolatable=True)},
    )

    with gp.build(source):
        total_input_roi = source.spec[pred_lsds].roi
        total_output_roi = total_input_roi.grow(-context, -context)

    f = zarr.open(out_file, "a")

    affs_path = os.path.join(out_file, out_dataset)

    if os.path.exists(affs_path):
        print("Out file already contains affs, removing dataset...")
        shutil.rmtree(affs_path)

    ds = f.create_dataset(
        out_dataset,
        shape=[3]
        + [i / j for i, j in zip(total_output_roi.get_shape(), voxel_size)],
    )
    ds.attrs["resolution"] = voxel_size
    ds.attrs["offset"] = total_output_roi.get_offset()

    predict = gp.torch.Predict(
        model,
        checkpoint=f"checkpoints/fake_lsds_checkpoint_{str(checkpoint)}",
        inputs={"input": pred_lsds},
        outputs={
            0: pred_affs,
        },
    )

    scan = gp.Scan(scan_request)

    write = gp.ZarrWrite(
        dataset_names={
            pred_affs: out_dataset,
        },
        output_filename=out_file,
    )

    pipeline = (
        source
        + gp.Unsqueeze([pred_lsds])
        + predict
        + gp.Squeeze([pred_affs])
        + write
        + scan
    )

    predict_request = gp.BatchRequest()

    predict_request[pred_lsds] = total_input_roi
    predict_request[pred_affs] = total_output_roi

    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    return batch[pred_affs].data, total_output_roi.get_begin()


if __name__ == "__main__":

    lsds_file = "test_prediction.zarr"
    out_file = lsds_file
    lsds_dataset = "lsds"
    out_dataset = "affs"

    fake_lsds_predict(lsds_file, lsds_dataset, out_file, out_dataset)
