import numpy as np
import pytest
from napari.utils.theme import get_theme
from qtpy.QtCore import QPoint

from napari_bootstrapper.post import segment_affs


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_cc_joins_the_negative_offset_neighbor(axis):
    affinities = np.zeros((3, 4, 4, 4), dtype=np.float32)
    affinities[axis, 1, 1, 1] = 1
    result = segment_affs(affinities, method="connected components")
    neighbor = [1, 1, 1]
    neighbor[axis] = 0
    assert result[1, 1, 1] > 0
    assert result[tuple(neighbor)] == result[1, 1, 1]
    assert np.count_nonzero(result) == 2


def test_mutex_watershed_filters_background_and_keeps_two_objects():
    labels = np.zeros((3, 6, 9), dtype=np.uint32)
    labels[:, 1:5, 1:4] = 1
    labels[:, 1:5, 5:8] = 2
    affinities = np.zeros((3, *labels.shape), dtype=np.float32)
    affinities[0, 1:] = (labels[1:] == labels[:-1]) & (labels[1:] != 0)
    affinities[1, :, 1:] = (labels[:, 1:] == labels[:, :-1]) & (
        labels[:, 1:] != 0
    )
    affinities[2, :, :, 1:] = (labels[:, :, 1:] == labels[:, :, :-1]) & (
        labels[:, :, 1:] != 0
    )
    affinities[:, labels == 2] *= 0.6
    params = {
        "mutex watershed": {
            "neighborhood": [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
            "bias": [-0.5, -0.5, -0.5],
            "filter_fragments": 0.1,
            "remove_debris": 0,
        }
    }
    result = segment_affs(affinities, params=params)
    assert result.shape == labels.shape
    assert np.issubdtype(result.dtype, np.integer)
    np.testing.assert_array_equal(result[labels == 0], 0)
    first, second = result[1, 2, 2], result[1, 2, 6]
    assert first > 0 and second > 0 and first != second
    np.testing.assert_array_equal(result[labels == 1], first)
    np.testing.assert_array_equal(result[labels == 2], second)
    params["mutex watershed"]["filter_fragments"] = 0.5
    filtered = segment_affs(affinities, params=params)
    np.testing.assert_array_equal(filtered[labels != 1], 0)
    assert np.all(filtered[labels == 1] > 0)


@pytest.fixture
def proofreading(qtbot):
    from napari.components import ViewerModel

    from napari_bootstrapper.post._merge_split_widget import (
        ProofreadingWidget,
    )

    viewer = ViewerModel()
    viewer.add_image(np.zeros((6, 9, 16, 16), dtype=np.float32), name="lsds")
    segmentation = np.zeros((9, 16, 16), dtype=np.int64)
    segmentation[:, 2:7, 2:7] = 1
    segmentation[:, 9:14, 9:14] = 2
    layer = viewer.add_labels(segmentation, name="segmentation")
    widget = ProofreadingWidget(viewer)
    qtbot.addWidget(widget)
    widget.labels_selector.setCurrentText("segmentation")
    viewer.dims.set_current_step(viewer.dims.ndim - 3, 4)
    return widget, layer


@pytest.mark.parametrize("operation", ["Merge", "Delete"])
def test_section_edits_change_the_shown_section(proofreading, operation):
    widget, layer = proofreading
    before = layer.data.copy()
    widget.group_selector.setCurrentText("Merge / Split")
    widget.edit_radios[operation].setChecked(True)
    widget.label_ids_input.setText("1, 2" if operation == "Merge" else "2")
    widget.apply_button.click()
    changed = np.nonzero((layer.data != before).any(axis=(1, 2)))[0]
    assert changed.tolist() == [4]


def test_fill_holes_around_an_enclosed_label_leaves_labels_unchanged(
    proofreading,
):
    widget, layer = proofreading
    layer.data[3:6, 3:6, 3:6] = 3
    before = layer.data.copy()
    widget.group_selector.setCurrentText("Morphology")
    widget.morph_radios["Fill holes"].setChecked(True)
    widget.apply3d_checkbox.setChecked(True)
    widget.apply_button.click()
    np.testing.assert_array_equal(layer.data, before)


def test_filter_turns_off_the_selection(proofreading):
    widget, _ = proofreading
    selection = (
        widget.selection_box,
        widget.label_ids_input,
        widget.apply3d_checkbox,
    )
    widget.group_selector.setCurrentText("Filter")
    assert not any(control.isEnabled() for control in selection)
    assert widget.labels_selector.isEnabled()
    widget.group_selector.setCurrentText("Merge / Split")
    assert all(control.isEnabled() for control in selection)


@pytest.mark.parametrize(
    "group,radio,text",
    [
        ("Morphology", "Dilate", "Dilate labels"),
        ("Morphology", "Fill holes", "Fill holes"),
        ("Merge / Split", "Merge", "Merge labels"),
        ("Merge / Split", "Split", "Split labels"),
        ("Merge / Split", "Delete", "Delete labels"),
        ("Filter", None, "Filter labels"),
    ],
)
def test_apply_button_names_the_operation(proofreading, group, radio, text):
    widget, _ = proofreading
    widget.group_selector.setCurrentText(group)
    if radio is not None:
        radios = (
            widget.morph_radios
            if group == "Morphology"
            else widget.edit_radios
        )
        radios[radio].click()
    assert widget.apply_button.text() == text


def test_unused_inputs_turn_off(proofreading):
    widget, _ = proofreading
    widget.group_selector.setCurrentText("Merge / Split")
    widget.edit_radios["Split"].click()
    widget.points_as_markers_cb.setChecked(True)
    assert not widget.min_distance_slider.isEnabled()
    widget.points_as_markers_cb.setChecked(False)
    assert widget.min_distance_slider.isEnabled()


def test_hidden_rows_do_not_widen_the_label_column(proofreading, qtbot):
    widget, _ = proofreading
    widget.group_selector.setCurrentText("Merge / Split")
    with qtbot.waitExposed(widget):
        widget.show()

    def field_x():
        return widget.labels_selector.mapTo(widget, QPoint(0, 0)).x()

    qtbot.wait(50)
    narrow = field_x()
    widget.group_selector.setCurrentText("Filter")
    qtbot.waitUntil(lambda: field_x() > narrow)
    widget.group_selector.setCurrentText("Merge / Split")
    qtbot.waitUntil(lambda: field_x() == narrow)


def test_filter_spinboxes_show_off_at_zero(proofreading):
    widget, _ = proofreading
    for spinbox in (
        widget.dust_spinbox,
        widget.max_size_spinbox,
        widget.largest_k_spinbox,
        widget.sigma_spinbox,
        widget.min_z_spinbox,
    ):
        assert spinbox.value() == 0
        assert spinbox.text() == "Off"


def test_proofreading_follows_the_viewer_theme(proofreading):
    widget, _ = proofreading
    for theme_id in ("light", "dark"):
        widget.viewer.theme = theme_id
        assert get_theme(theme_id).current.as_rgb() in widget.styleSheet()
    assert widget.apply_button.property("primary") is True
    assert widget.undo_button.property("primary") is not True
