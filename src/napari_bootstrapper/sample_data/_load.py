"""
This module provides the CREMI C and Fluo-C2DL-Huh7 sample data to napari.

It implements the "sample data" specification at
https://napari.org/stable/plugins/building_a_plugin/guides.html#sample-data
"""

from __future__ import annotations

from pathlib import Path

from tifffile import imread

CREMI_IMAGE = Path(__file__).parent / "cremi_image.tif"
CREMI_LABELS = Path(__file__).parent / "cremi_labels.tif"

FLUO_IMAGE = Path(__file__).parent / "Fluo-C2DL-Huh7-image.tif"
FLUO_LABELS = Path(__file__).parent / "Fluo-C2DL-Huh7-labels.tif"
FLUO_MASK = Path(__file__).parent / "Fluo-C2DL-Huh7-mask.tif"


def make_cremi_sample_data():
    raw = imread(CREMI_IMAGE)
    labels = imread(CREMI_LABELS)
    return [
        (
            raw,
            {
                "name": "raw",
            },
            "image",
        ),
        (
            labels,
            {
                "name": "sparse_labels",
            },
            "labels",
        ),
    ]


def make_fluo_c2dl_huh7_sample_data():
    raw = imread(FLUO_IMAGE)
    labels = imread(FLUO_LABELS)
    mask = imread(FLUO_MASK)
    return [
        (
            raw,
            {
                "name": "raw",
            },
            "image",
        ),
        (
            labels,
            {
                "name": "sparse_labels",
            },
            "labels",
        ),
        (
            mask,
            {
                "name": "training_mask",
            },
            "labels",
        ),
    ]
