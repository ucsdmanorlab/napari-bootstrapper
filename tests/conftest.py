import os
import random
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest
import torch
from napari.components import ViewerModel

from napari_bootstrapper.widget import Widget


@pytest.fixture(autouse=True)
def deterministic_cpu():
    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    torch.set_num_threads(2)


@pytest.fixture
def widget(qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    widget = Widget(ViewerModel())
    widget.device_combo_box.setCurrentText("cpu")
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def small_2d_config():
    return {
        "batch_size": 1,
        "learning_rate": 1e-3,
        "max_iterations": 1,
        "num_workers": 0,
        "save_snapshots_every": 1,
        "net": {
            "adj_slices": 3,
            "num_fmaps": 2,
            "num_fmaps_out": 2,
            "fmap_inc_factor": 2,
            "downsample_factors": ((2, 2),),
            "kernel_size_down": (((3, 3),), ((3, 3),)),
            "kernel_size_up": (((3, 3),),),
            "input_shape": [24, 24],
            "output_shape": [16, 16],
            "shape_increase": [0, 0],
        },
        "task": {
            "lsd_sigma": 2,
            "lsd_downsample": 2,
            "aff_neighborhood": [[-1, 0], [0, -1]],
            "aff_grow_boundary": 0,
        },
    }


@pytest.fixture
def small_3d_config():
    return {
        "batch_size": 1,
        "learning_rate": 1e-3,
        "max_iterations": 1,
        "num_workers": 0,
        "save_snapshots_every": 1,
        "net": {
            "num_fmaps": 2,
            "num_fmaps_out": 2,
            "fmap_inc_factor": 2,
            "downsample_factors": ((1, 2, 2),),
            "kernel_size_down": (((1, 3, 3),), ((3, 3, 3),)),
            "kernel_size_up": (((1, 3, 3),),),
            "input_shape": [5, 24, 24],
            "output_shape": [3, 16, 16],
            "shape_increase": [0, 0, 0],
        },
        "task": {
            "lsd_sigma": 2,
            "lsd_downsample": 2,
            "in_aff_neighborhood": [[-1, 0], [0, -1]],
            "aff_neighborhood": [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
            "aff_grow_boundary": 1,
        },
    }


@pytest.fixture
def sparse_layers(widget):
    raw = np.random.default_rng(4).random((3, 48, 48), dtype=np.float32)
    labels = np.zeros(raw.shape, dtype=np.uint32)
    labels[1, 10:22, 10:22] = 1
    labels[1, 26:38, 26:38] = 2
    image_layer = widget.viewer.add_image(raw, name="raw")
    labels_layer = widget.viewer.add_labels(labels, name="labels")
    mask_layer = widget.viewer.add_labels(
        (labels != 0).astype(np.uint8), name="mask"
    )
    widget.raw_selector.setCurrentText("raw")
    widget.labels_selector.setCurrentText("labels")
    widget.mask_selector.setCurrentText("mask")
    return image_layer, labels_layer, mask_layer
