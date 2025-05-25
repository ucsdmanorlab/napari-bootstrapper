"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/building_a_plugin/guides.html#sample-data

Replace code below according to your needs.
"""

from __future__ import annotations

from pathlib import Path

from tifffile import imread

CREMI_IMAGE_SAMPLE = Path(__file__).parent / "cremi_image.tif"
CREMI_LABELS_SAMPLE = Path(__file__).parent / "cremi_labels.tif"


def make_cremi_sample_data():
    raw = imread(CREMI_IMAGE_SAMPLE)
    painting = imread(CREMI_LABELS_SAMPLE)
    return [
        (
            raw,
            {
                "name": "raw",
            },
            "image",
        ),
        (
            painting,
            {
                "name": "sparse_labels",
            },
            "labels",
        ),
    ]
