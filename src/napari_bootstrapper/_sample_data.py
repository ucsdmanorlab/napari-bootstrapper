"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/building_a_plugin/guides.html#sample-data

Replace code below according to your needs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CREMI_IMAGE_SAMPLE = Path(__file__).parent / "sample_data" / "image.npy"
CREMI_LABELS_SAMPLE = Path(__file__).parent / "sample_data" / "labels.npy"


def make_sample_data():
    raw = np.load(CREMI_IMAGE_SAMPLE, "r")
    painting = np.load(CREMI_LABELS_SAMPLE, "r")
    raw = raw.astype(np.uint8)
    painting = painting.astype(np.uint32)
    return [
        (
            raw,
            {
                "name": "Raw",
                "metadata": {"axes": ["z", "y", "x"]},
            },
            "image",
        ),
        (
            painting,
            {
                "name": "Sparse Labels",
                "metadata": {"axes": ["z", "y", "x"]},
            },
            "Labels",
        ),
    ]
