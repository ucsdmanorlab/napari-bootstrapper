import numpy as np
import pytest

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
