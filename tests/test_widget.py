from collections import Counter
from copy import deepcopy

import numpy as np
import pytest
import torch
import zarr
from napari.components import ViewerModel
from napari.utils.theme import get_theme
from npe2 import PluginManager
from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QColor

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


def test_pickers_list_only_matching_layer_types(widget):
    def items(combo):
        return [combo.itemText(i) for i in range(combo.count())]

    widget.viewer.add_image(np.zeros((3, 8, 8), np.float32), name="raw")
    widget.viewer.add_labels(np.zeros((3, 8, 8), np.uint8), name="labels")
    widget.viewer.add_points(np.zeros((1, 3)), name="points")
    assert items(widget.raw_selector) == ["raw"]
    assert items(widget.labels_selector) == ["labels"]
    assert items(widget.mask_selector) == ["labels"]

    widget.viewer.layers.remove("labels")
    assert items(widget.labels_selector) == []
    assert items(widget.mask_selector) == []
    assert items(widget.raw_selector) == ["raw"]


def test_cancelled_save_writes_nothing(widget, monkeypatch):
    monkeypatch.setattr(
        "napari_bootstrapper.widget.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: ("", ""),
    )
    widget.model_2d = torch.nn.Linear(1, 1)
    widget.optimizer_2d = torch.optim.Adam(widget.model_2d.parameters())
    widget.iterations_2d = [0]
    widget.losses_2d = [1.0]

    def save(*args, **kwargs):
        raise AssertionError("Save wrote a checkpoint after Cancel")

    monkeypatch.setattr("napari_bootstrapper.widget.torch.save", save)
    widget.save_weights("2d")


@pytest.mark.parametrize("theme_id", ["light", "dark"])
def test_widget_follows_the_viewer_theme(widget, theme_id):
    widget.viewer.theme = theme_id
    theme = get_theme(theme_id)
    assert theme.current.as_rgb() in widget.styleSheet()
    background = QColor(theme.foreground.as_hex()).name()
    for plot in (widget.losses_2d_widget, widget.losses_3d_widget):
        assert plot.backgroundBrush().color().name() == background
    assert theme.text.as_hex() in widget._description_label.text()


def test_header_logo_follows_the_viewer_theme(widget):
    logo = widget._logo_label
    assert logo.accessibleName() == "Bootstrapper"
    assert logo.toolTip() == "Bootstrapper"
    images = {}
    for theme_id in ("light", "dark"):
        widget.viewer.theme = theme_id
        pixmap = logo.pixmap()
        assert pixmap is not None
        assert not pixmap.isNull()
        ratio = pixmap.devicePixelRatio()
        assert ratio == widget.devicePixelRatioF()
        assert (pixmap.width() / ratio, pixmap.height() / ratio) == (168, 32)
        image = pixmap.toImage()
        images[theme_id] = image
        colors = [
            image.pixelColor(x, y)
            for x in range(image.width())
            for y in range(image.height())
        ]
        ink = Counter(color.name() for color in colors if color.alpha() == 255)
        # Most of the ink is the letters, which must stand out from the
        # panel in both themes.
        letters = QColor(ink.most_common(1)[0][0])
        background = QColor(get_theme(theme_id).background.as_hex())
        assert abs(letters.lightnessF() - background.lightnessF()) > 0.5
    assert images["light"] != images["dark"]


def test_3d_training_controls_start_hidden(widget):
    shown = (widget.download_3d_model_button, widget.load_3d_model_button)
    hidden = (
        widget.start_3d_training_button,
        widget.train_3d_model_from_scratch_checkbox,
    )
    assert all(button.isVisibleTo(widget) for button in shown)
    assert not any(control.isVisibleTo(widget) for control in hidden)

    widget.train_3d_toggle.click()
    assert all(control.isVisibleTo(widget) for control in hidden)
    assert widget.train_3d_toggle.arrowType() == Qt.DownArrow
    widget.train_3d_toggle.click()
    assert not any(control.isVisibleTo(widget) for control in hidden)

    widget.train_3d_toggle.click()
    widget.stop_3d_training_button.setEnabled(True)
    widget.train_3d_toggle.click()
    assert widget.start_3d_training_button.isVisibleTo(widget)


def test_sections_and_buttons_match_the_readme(widget):
    boxes = (
        widget.data_box,
        widget.model_2d_box,
        widget.model_3d_box,
        widget.seg_box,
    )
    assert [box.accessibleName() for box in boxes] == [
        "1. Data",
        "2. 2D model",
        "3. 3D model",
        "4. Segmentation",
    ]
    assert widget.load_2d_model_button.text() == "Load model"
    assert widget.load_3d_model_button.text() == "Load model"
    for button in (
        widget.advanced_2d_config_button,
        widget.advanced_3d_config_button,
        widget.advanced_seg_config_button,
    ):
        assert button.text() == "Advanced Config"
        assert button.property("compact") is True
    for button in (
        widget.start_2d_training_button,
        widget.download_3d_model_button,
        widget.start_inference_button,
    ):
        assert button.property("primary") is True
    for button in (
        widget.stop_2d_training_button,
        widget.load_3d_model_button,
        widget.start_3d_training_button,
    ):
        assert button.property("primary") is not True


def test_loss_plot_keeps_one_curve(widget):
    widget.on_yield_training("2d", ([0.5, 0.4], [0, 1]))
    widget.on_yield_training("2d", ([0.5, 0.4, 0.3], [0, 1, 2]))
    assert len(widget.losses_2d_widget.listDataItems()) == 1
    assert widget.losses_2d_widget.getAxis("left").autoSIPrefix is False


def test_boxed_fields_share_one_left_edge(widget, qtbot):
    widget.resize(420, 900)
    with qtbot.waitExposed(widget):
        widget.show()
    fields = (
        widget.raw_selector,
        widget.labels_selector,
        widget.make_mask_button,
        widget.mask_selector,
        widget.model_2d_type_selector,
        widget.model_3d_type_selector,
        widget.seg_method_selector,
    )
    edges = {field.mapTo(widget, QPoint(0, 0)).x() for field in fields}
    assert len(edges) == 1
