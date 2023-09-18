from pathlib import Path

import numpy as np

CREMI_IMAGE_SAMPLE = Path(__file__).parent / "image.npy"
CREMI_LABELS_SAMPLE = Path(__file__).parent / "labels.npy"


def cremi_sample():
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
                "name": "Painted Labels",
                "metadata": {"axes": ["z", "y", "x"]},
            },
            "Labels",
        ),
    ]
