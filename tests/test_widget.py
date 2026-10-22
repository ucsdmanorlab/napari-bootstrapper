from copy import deepcopy

import numpy as np
import pytest
import torch
import zarr
from napari.components import ViewerModel
from npe2 import PluginManager

from napari_bootstrapper.widget import Widget


def test_widget_builds_and_updates_model_choices(widget):
    assert widget.device_combo_box.currentText() == "cpu"
    assert widget.start_2d_training_button.isEnabled()
    for task, expected in (
        ("2d_lsd", ["3d_affs_from_2d_lsd"]),
        ("2d_affs", ["3d_affs_from_2d_affs"]),
        (
            "2d_mtlsd",
            [
                "3d_affs_from_2d_affs",
                "3d_affs_from_2d_lsd",
                "3d_affs_from_2d_mtlsd",
            ],
        ),
    ):
        widget.model_2d_type_selector.setCurrentText(task)
        selector = widget.model_3d_type_selector
        assert [
            selector.itemText(i) for i in range(selector.count())
        ] == expected


def test_widget_settings_do_not_change_another_viewer(widget, qtbot):
    original_2d = deepcopy(widget.model_2d_config)
    original_3d = deepcopy(widget.model_3d_config)
    widget.model_2d_config["net"]["input_shape"][0] = 128
    widget.model_3d_config["net"]["input_shape"][0] = 16
    widget.seg_params["connected components"]["threshold"] = 0.9
    other = Widget(ViewerModel())
    qtbot.addWidget(other)
    assert other.model_2d_config == original_2d
    assert other.model_3d_config == original_3d
    assert other.seg_params["connected components"]["threshold"] == 0.5


@pytest.mark.parametrize(
    "sample,count",
    [
        ("napari-bootstrapper-cremi", 2),
        ("napari-bootstrapper-fluo_c2dl_huh7", 3),
    ],
)
def test_bundled_sample_loads_in_viewer(sample, count):
    PluginManager.instance().discover()
    layers = ViewerModel().open_sample("napari-bootstrapper", sample)
    assert len(layers) == count
    raw, labels = layers[:2]
    assert raw.data.shape == labels.data.shape
    assert raw.data.ndim == 3
    assert np.ptp(raw.data) > 0
    assert np.count_nonzero(labels.data) > 0
    assert np.count_nonzero(labels.data) < labels.data.size
    assert layers[0]._type_string == "image"
    assert all(layer._type_string == "labels" for layer in layers[1:])


@pytest.mark.parametrize(
    "task,channels",
    [("2d_affs", (2,)), ("2d_lsd", (6,)), ("2d_mtlsd", (6, 2))],
)
def test_2d_train_and_predict_on_cpu(
    widget, sparse_layers, small_2d_config, task, channels
):
    widget.model_2d_type_selector.setCurrentText(task)
    widget.model_2d_config = small_2d_config
    originals = [layer.data.copy() for layer in sparse_layers]
    assert widget.train("2d").work() is None
    assert len(widget.losses_2d) == 1
    assert np.isfinite(widget.losses_2d[0])
    assert widget.losses_2d[0] > 0
    assert widget.optimizer_2d.state
    assert all(
        state["step"] == 1 for state in widget.optimizer_2d.state.values()
    )

    widget.model_2d.eval()
    image = torch.from_numpy(sparse_layers[0].data[None, :, :24, :24])
    with torch.no_grad():
        outputs = widget.model_2d(image)
    assert len(outputs) == len(channels)
    for output, count in zip(outputs, channels, strict=True):
        assert output.shape == (1, count, 1, 16, 16)
        assert torch.isfinite(output).all()
        assert torch.std(output.flatten(-2), dim=-1).max() > 0
    for layer, original in zip(sparse_layers, originals, strict=True):
        np.testing.assert_array_equal(layer.data, original)

    snapshot = zarr.open(widget.tmp_dir / f"snapshots_{task}.zarr", mode="r")
    assert snapshot["0/outputs_0"].shape == (channels[0], 1, 16, 16)
    assert np.isfinite(snapshot["0/outputs_0"][:]).all()


@pytest.mark.parametrize(
    "task,channels",
    [
        ("3d_affs_from_2d_affs", (2,)),
        ("3d_affs_from_2d_lsd", (6,)),
        ("3d_affs_from_2d_mtlsd", (6, 2)),
    ],
)
def test_3d_train_and_predict_on_cpu(widget, small_3d_config, task, channels):
    widget.model_2d_type_selector.setCurrentText("2d_mtlsd")
    widget.model_3d_type_selector.setCurrentText(task)
    widget.model_3d_config = small_3d_config
    widget.train_3d_model_from_scratch_checkbox.setChecked(True)
    assert widget.train("3d").work() is None
    assert len(widget.losses_3d) == 1
    assert np.isfinite(widget.losses_3d[0])
    assert widget.losses_3d[0] > 0
    assert widget.optimizer_3d.state
    assert all(
        state["step"] == 1 for state in widget.optimizer_3d.state.values()
    )
    widget.model_3d.eval()
    inputs = [torch.rand(1, count, 5, 24, 24) for count in channels]
    with torch.no_grad():
        output = widget.model_3d(*inputs)
    assert output.shape == (1, 3, 3, 16, 16)
    assert torch.isfinite(output).all()
    assert torch.std(output.flatten(-3), dim=-1).max() > 0


def test_widget_trains_saves_and_segments_a_volume(
    widget, sparse_layers, small_2d_config, small_3d_config, qtbot
):
    widget.model_2d_type_selector.setCurrentText("2d_mtlsd")
    widget.model_3d_type_selector.setCurrentText("3d_affs_from_2d_mtlsd")
    widget.model_2d_config = small_2d_config
    widget.model_3d_config = small_3d_config
    widget.train_3d_model_from_scratch_checkbox.setChecked(True)
    for dimension in ("2d", "3d"):
        worker = widget.train(dimension)
        setattr(widget, f"train_{dimension}_worker", worker)
        assert worker.work() is None
        widget.prepare_for_stop_training(dimension)
        assert getattr(
            widget, f"pretrained_{dimension}_model_checkpoint"
        ).is_file()
        delattr(widget, f"model_{dimension}")

    widget.seg_params = deepcopy(widget.seg_params)
    widget.seg_params["mutex watershed"].update(
        neighborhood=small_3d_config["task"]["aff_neighborhood"],
        filter_fragments=0,
        remove_debris=0,
        noise_eps=None,
        randomized_strides=False,
    )
    shape = sparse_layers[0].data.shape
    for method in ("connected components", "mutex watershed"):
        widget.seg_method_selector.setCurrentText(method)
        errors = []
        widget.prepare_for_start_inference()
        widget.inference_worker.errored.connect(errors.append)
        qtbot.waitUntil(
            lambda: bool(errors) or widget.start_inference_button.isEnabled(),
            timeout=30000,
        )
        assert not errors
        segmentation = widget.viewer.layers["segmentation"]
        assert segmentation._type_string == "labels"
        assert segmentation.data.shape == shape
        assert segmentation.data.max() > 0
        assert np.issubdtype(segmentation.data.dtype, np.integer)
        assert widget.lsds_2d.shape == (6, *shape)
        assert widget.affs_2d.shape == (2, *shape)
        assert widget.affs_3d.shape == (3, *shape)
        for array in (widget.lsds_2d, widget.affs_2d, widget.affs_3d):
            assert np.isfinite(array).all()
            assert np.std(array, axis=(-3, -2, -1)).max() > 0


def test_load_model_replaces_the_trained_model(
    widget, sparse_layers, small_2d_config, tmp_path, monkeypatch
):
    widget.model_2d_type_selector.setCurrentText("2d_affs")
    widget.model_2d_config = small_2d_config

    def train():
        widget.train_2d_worker = widget.train("2d")
        assert widget.train_2d_worker.work() is None
        widget.prepare_for_stop_training("2d")
        return deepcopy(widget.model_2d.state_dict())

    first = train()
    checkpoint = tmp_path / "first.pth"
    monkeypatch.setattr(
        "napari_bootstrapper.widget.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(checkpoint), ""),
    )
    widget.save_weights("2d")
    second = train()
    assert any(not torch.equal(first[k], second[k]) for k in first)

    widget.train_2d_model_from_scratch_checkbox.setChecked(False)
    monkeypatch.setattr(
        "napari_bootstrapper.widget.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(checkpoint), ""),
    )
    widget.load_weights("2d")
    widget.reload_model_if_needed("2d", "2d_affs", num_channels=3)
    loaded = widget.model_2d.state_dict()
    assert all(torch.equal(loaded[k], first[k]) for k in first)
