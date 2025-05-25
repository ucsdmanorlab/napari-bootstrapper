### taken from https://github.com/volume-em/empanada-napari/blob/main/empanada_napari/_merge_split_widget.py
### and https://github.com/volume-em/empanada-napari/blob/main/empanada/array_utils.py

import dask.array as da
import napari
import numpy as np
from magicgui import magicgui
from napari_plugin_engine import napari_hook_implementation
from scipy import ndimage as ndi
from skimage import draw
from skimage import morphology as morph
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.segmentation import watershed


def take(array, indices, axis=0):
    r"""Take indices from array along an axis

    Args:
        array: np.ndarray
        indices: List of indices
        axis: Int. Axis to take from.

    Returns:
        output: np.ndarray

    """
    indices = tuple(
        [slice(None) if n != axis else indices for n in range(array.ndim)]
    )

    return array[indices]


def put(array, indices, value, axis=0):
    r"""Put values at indices, inplace, along an axis.

    Args:
        array: np.ndarray
        indices: List of indices
        axis: Int. Axis to put along.

    """
    indices = tuple(
        [slice(None) if n != axis else indices for n in range(array.ndim)]
    )

    # modify the array inplace
    array[indices] = value


def crop_and_binarize(mask, box, label):
    r"""Crop a mask from a bounding box and binarize the cropped mask
    where it's equal to the given label value.

    Args:
        mask: Array of (h, w) or (d, h, w) defining an image.
        box: Bounding box tuple of (y1, x1, y2, x2) or (z1, y1, x1, z2, y2, x2).
        label: Label value to binarize within cropped mask.

    Returns:
        binary_cropped_mask: Boolean array of (h', w') or (d', h', w').

    """
    ndim = len(box) // 2
    slices = tuple([slice(box[i], box[i + ndim]) for i in range(ndim)])

    return mask[slices] == label


def map_points(world_points, labels_layer):
    assert all(
        s == 1 for s in labels_layer.scale
    ), "Labels layer must have scale of all ones!"
    # assert all(t == 0 for t in labels_layer.translate), "Labels layer must have translation of (0, 0, 0)!"

    local_points = []
    for pt in world_points:
        local_points.append(
            tuple([int(c) for c in labels_layer.world_to_data(pt)])
        )

    return local_points


def get_local_points(labels_layer, label_ids):
    local_points = []
    for rp in regionprops(labels_layer.data):
        if rp.label in label_ids:
            world_points = rp.centroid
            local_points.append(tuple([int(c) for c in world_points]))

    return local_points


def _box_to_slice(shed_box):
    n = len(shed_box)
    n_dim = n // 2

    slices = []
    for i in range(n_dim):
        s = shed_box[i]
        e = shed_box[i + n_dim]
        slices.append(slice(s, e))

    return tuple(slices)


def morph_labels():

    ops = {
        "Dilate": morph.binary_dilation,
        "Erode": morph.binary_erosion,
        "Close": morph.binary_closing,
        "Open": morph.binary_opening,
        "Fill holes": morph.remove_small_holes,
    }

    def _pad_box(shed_box, shape, radius=0):
        n = len(shed_box)
        n_dim = n // 2

        padded = [0] * len(shed_box)
        for i in range(n_dim):
            s = max(0, shed_box[i] - radius)
            e = min(shape[i], shed_box[i + n_dim] + radius)
            padded[i] = s
            padded[i + n_dim] = e

        return tuple(padded)

    @magicgui(
        call_button="Apply",
        layout="vertical",
        operation={
            "widget_type": "ComboBox",
            "choices": list(ops.keys()),
            "value": list(ops.keys())[0],
            "label": "Operation",
            "tooltip": "Morphological operation to apply",
        },
        radius={
            "widget_type": "Slider",
            "label": "Radius",
            "min": 1,
            "max": 7,
            "value": 1,
            "tooltip": "Radius of selem for morphology op.",
        },
        hole_size={
            "widget_type": "LineEdit",
            "value": "64",
            "label": "Hole size",
            "tooltip": "Max hole size to fill if op is fill hole",
        },
        apply3d={
            "widget_type": "CheckBox",
            "text": "Apply in 3D",
            "value": False,
            "tooltip": "Check box to apply the operation in 3D.",
        },
    )
    def widget(
        viewer: napari.viewer.Viewer,
        labels_layer: napari.layers.Labels,
        points_layer: napari.layers.Points,
        operation: int,
        radius: bool,
        hole_size: str,
        apply3d: bool,
    ):
        hole_size = int(hole_size)
        labels = labels_layer.data

        if operation == "Fill holes":
            op_arg = hole_size
        elif labels.ndim == 3 and apply3d:
            op_arg = morph.ball(radius)
        else:
            op_arg = morph.disk(radius)

        if apply3d and labels.ndim != 3:
            print("Apply 3D checked, but labels are not 3D. Ignoring.")

        if points_layer is None:
            label_ids = np.unique(labels)[1:].tolist()
            local_points = get_local_points(labels_layer, label_ids)
        else:
            world_points = points_layer.data
            local_points = map_points(world_points, labels_layer)

            # get points as indices in local coordinates
            # local_points = map_points(world_points, labels_layer)

            if type(labels) is da.core.Array:
                raise Exception(
                    "Morph operations are not supported on Dask Array labels!"
                )
            else:
                label_ids = [labels[pt].item() for pt in local_points]

        # drop any label_ids equal to 0 in case point
        # was placed on the background
        label_ids = list(filter(lambda x: x > 0, label_ids))

        if len(label_ids) == 0:
            print("No labels selected!")
            return

        for label_id in label_ids:
            if labels.ndim == 2 or (labels.ndim == 3 and apply3d):
                shed_box = [
                    rp.bbox
                    for rp in regionprops(labels)
                    if rp.label == label_id
                ][0]
                shed_box = _pad_box(shed_box, labels.shape, radius)
                slices = _box_to_slice(shed_box)

                # apply op
                binary = crop_and_binarize(labels, shed_box, label_id)

                labels[slices][binary] = 0
                binary = ops[operation](binary, op_arg)
                labels[slices][binary] = label_id

            elif labels.ndim == 3:
                # # get the current viewer axis
                # axis = viewer.dims.order[0]
                if points_layer is None:
                    plane = viewer.dims.current_step[0]
                    labels2d = labels[plane]

                    shed_box = [
                        rp.bbox
                        for rp in regionprops(labels2d)
                        if rp.label == label_ids
                    ]
                    shed_box = _pad_box(shed_box, labels.shape, radius)
                    slices = _box_to_slice(shed_box)

                    binary = crop_and_binarize(labels2d, shed_box, label_id)
                    labels2d[slices][binary] = 0
                    binary = ops[operation](binary, op_arg)
                    labels2d[slices][binary] = label_id

                    put(labels, plane, labels2d)
                else:
                    # get the current viewer axis
                    axis = viewer.dims.order[0]
                    plane = local_points[0][axis]
                    labels2d = take(labels, plane, axis)
                    assert all(
                        local_pt[axis] == plane for local_pt in local_points
                    )

                    shed_box = [
                        rp.bbox
                        for rp in regionprops(labels2d)
                        if rp.label == label_id
                    ][0]
                    shed_box = _pad_box(shed_box, labels.shape, radius)
                    slices = _box_to_slice(shed_box)

                    binary = crop_and_binarize(labels2d, shed_box, label_id)
                    labels2d[slices][binary] = 0
                    binary = ops[operation](binary, op_arg)
                    labels2d[slices][binary] = label_id

                    put(labels, plane, labels2d, axis)

            elif labels.ndim == 4:
                # get the current viewer axes
                assert (
                    viewer.dims.order[0] == 0
                ), "Dims expected to be (0, 1, 2, 3) for 4D labels!"
                assert (
                    viewer.dims.order[1] == 1
                ), "Dims expected to be (0, 1, 2, 3) for 4D labels!"
                if points_layer is not None:
                    plane1 = viewer.dims.current_step[0]
                    plane2 = viewer.dims.current_step[1]
                    labels = labels_layer.data
                    labels2d = labels[plane1, plane2]

                    shed_box = [
                        rp.bbox
                        for rp in regionprops(labels2d)
                        if rp.label == label_ids
                    ]
                else:
                    # get the current viewer axes
                    plane1 = local_points[0][0]
                    plane2 = local_points[0][1]
                    assert all(
                        local_pt[0] == plane1 for local_pt in local_points
                    )
                    assert all(
                        local_pt[1] == plane2 for local_pt in local_points
                    )

                    labels2d = labels[plane1, plane2]

                    shed_box = [
                        rp.bbox
                        for rp in regionprops(labels2d)
                        if rp.label == label_id
                    ][0]

                shed_box = _pad_box(shed_box, labels.shape, radius)
                slices = _box_to_slice(shed_box)

                binary = crop_and_binarize(labels2d, shed_box, label_id)
                labels2d[slices][binary] = 0
                binary = ops[operation](binary, op_arg)
                labels2d[slices][binary] = label_id

                labels[plane1, plane2] = labels2d

        labels_layer.data = labels
        # if points_layer is not None:
        #     return
        # else:
        #     points_layer.data = []

    return widget


def delete_labels():
    @magicgui(
        call_button="Delete labels",
        layout="vertical",
        apply3d={
            "widget_type": "CheckBox",
            "text": "Apply in 3D",
            "value": False,
            "tooltip": "Check box to delete label in 3D.",
        },
    )
    def widget(
        viewer: napari.viewer.Viewer,
        labels_layer: napari.layers.Labels,
        points_layer: napari.layers.Points,
        apply3d,
    ):
        if points_layer is None:
            points_layer = viewer.add_points([])
            points_layer.mode = "ADD"
            print("Add points!")
            return

        labels = labels_layer.data
        world_points = points_layer.data

        if apply3d and labels.ndim != 3:
            print("Apply 3D checked, but labels are not 3D. Ignoring.")

        # get points as indices in local coordinates
        local_points = map_points(world_points, labels_layer)

        if type(labels) is da.core.Array:
            label_ids = [labels[pt].compute() for pt in local_points]
        else:
            label_ids = [labels[pt].item() for pt in local_points]

        # drop any label_ids equal to 0 in case point
        # was placed on the background
        label_ids = list(filter(lambda x: x > 0, label_ids))

        if labels.ndim == 2 or (labels.ndim == 3 and apply3d):
            for l_ in label_ids:
                labels[labels == l_] = 0
        elif labels.ndim == 3:
            # get the current viewer axis
            axis = viewer.dims.order[0]

            # take labels along axis
            for local_pt in local_points:
                labels2d = take(labels, local_pt[axis], axis)
                for l_ in label_ids:
                    labels2d[labels2d == l_] = 0

                put(labels, local_pt[axis], labels2d, axis)
        elif labels.ndim == 4:
            # get the current viewer axes
            assert (
                viewer.dims.order[0] == 0
            ), "Dims expected to be (0, 1, 2, 3) for 4D labels!"
            assert (
                viewer.dims.order[1] == 1
            ), "Dims expected to be (0, 1, 2, 3) for 4D labels!"

            # take labels along axis
            for local_pt in local_points:
                labels2d = labels[local_pt[0], local_pt[1]]
                for l_ in label_ids:
                    labels2d[labels2d == l_] = 0

                labels[local_pt[0], local_pt[1]] = labels2d

        labels_layer.data = labels
        points_layer.data = []

        print(f"Removed labels {label_ids}")

    return widget


def merge_labels():

    def _line_to_indices(line, axis):
        if len(line[0]) == 2:
            line = line.ravel().astype("int").tolist()
            indices = np.stack(draw.line(*line), axis=1)
        elif len(line[0]) == 3:
            plane = line[0][axis]
            keep_axes = [i for i in range(3) if i != axis]
            line = line[:, keep_axes]
            line = line.ravel().astype("int").tolist()
            y, x = draw.line(*line)
            # add plane to indices
            z = np.full_like(x, plane)
            indices = [y, x]
            indices.insert(axis, z)
            indices = np.stack(indices, axis=1)
        elif len(line[0]) == 4:
            assert axis == 0
            planes = line[0][:2]
            line = line[:, [2, 3]]
            line = line.ravel().astype("int").tolist()
            y, x = draw.line(*line)
            # add plane to indices
            t = np.full_like(x, planes[0])
            z = np.full_like(x, planes[1])
            indices = np.stack([t, z, y, x], axis=1)
        else:
            raise Exception("Only lines in 2d, 3d, and 4d are supported!")

        return indices

    @magicgui(
        call_button="Merge labels",
        layout="vertical",
        apply3d={
            "widget_type": "CheckBox",
            "text": "Apply in 3D",
            "value": False,
            "tooltip": "Check box to merge label in 3D.",
        },
    )
    def widget(
        viewer: napari.viewer.Viewer,
        labels_layer: napari.layers.Labels,
        points_layer: napari.layers.Points,
        shapes_layer: napari.layers.Shapes,
        apply3d,
    ):
        if points_layer is None and shapes_layer is None:
            points_layer = viewer.add_points([])
            points_layer.mode = "ADD"
            print("Add points!")
            return

        axis = viewer.dims.order[0]
        labels = labels_layer.data
        world_points = []
        if points_layer is not None:
            world_points.append(points_layer.data)

        if shapes_layer is not None:
            for stype, shape in zip(
                shapes_layer.shape_type, shapes_layer.data, strict=False
            ):
                if stype == "line":
                    world_points.append(_line_to_indices(shape, axis))
                elif stype == "path":
                    n = len(shape)  # number of vertices
                    for i in range(n):
                        world_points.append(
                            _line_to_indices(shape[i : i + 2], axis)
                        )
                        if i == n - 2:
                            break

        world_points = np.concatenate(world_points, axis=0)

        if apply3d and labels.ndim != 3:
            print("Apply 3D checked, but labels are not 3D. Ignoring.")

        # get points as indices in local coordinates
        local_points = map_points(world_points, labels_layer)

        # clip local points outside of labels shape
        for idx, pt in enumerate(local_points):
            clipped_point = ()
            for i, size in enumerate(labels.shape):
                clipped_point += (min(size - 1, max(0, pt[i])),)

            local_points[idx] = clipped_point

        if type(labels) is da.core.Array:
            label_ids = [labels[pt].compute() for pt in local_points]
        else:
            label_ids = [labels[pt].item() for pt in local_points]

        # drop any label_ids equal to 0 in case point
        # was placed on the background
        label_ids = list(filter(lambda x: x > 0, label_ids))
        label_ids = np.unique(label_ids)

        # get merged label value
        # prefer the currently selected label
        if labels_layer.selected_label in label_ids:
            new_label_id = labels_layer.selected_label
        else:
            new_label_id = min(label_ids)

        if labels.ndim == 2 or (labels.ndim == 3 and apply3d):
            # replace labels with minimum of the selected labels
            for l_ in label_ids:
                if l_ != new_label_id:
                    labels[labels == l_] = new_label_id
        elif labels.ndim == 3:
            # take labels along axis
            for local_pt in local_points:
                labels2d = take(labels, local_pt[axis], axis)
                # replace labels with minimum of the selected labels
                for l_ in label_ids:
                    if l_ != new_label_id:
                        labels2d[labels2d == l_] = new_label_id

                put(labels, local_pt[axis], labels2d, axis)
        elif labels.ndim == 4:
            # get the current viewer axes
            assert (
                viewer.dims.order[0] == 0
            ), "Dims expected to be (0, 1, 2, 3) for 4D labels!"
            assert (
                viewer.dims.order[1] == 1
            ), "Dims expected to be (0, 1, 2, 3) for 4D labels!"

            # take labels along axis
            for local_pt in local_points:
                labels2d = labels[local_pt[0], local_pt[1]]
                for l_ in label_ids:
                    if l_ != new_label_id:
                        labels2d[labels2d == l_] = new_label_id

                labels[local_pt[0], local_pt[1]] = labels2d

        labels_layer.data = labels
        if points_layer is not None:
            points_layer.data = []
        if shapes_layer is not None:
            shapes_layer.data = []

        print(f"Merged labels {label_ids} to {new_label_id}")

    return widget


def split_labels():

    def _translate_point_in_box(point, shed_box):
        n_dim = len(shed_box) // 2
        return tuple([int(point[i] - shed_box[i]) for i in range(n_dim)])

    def _distance_markers(binary, min_distance):
        distance = ndi.distance_transform_edt(binary)
        energy = -distance

        # handle irritating quirk of peak_local_max
        if 1 in distance.shape:
            coords = peak_local_max(
                np.squeeze(distance), min_distance=min_distance
            )
            markers = np.zeros(np.squeeze(distance).shape, dtype=bool)
            markers[tuple(coords.T)] = True

            expand_axis = [s == 1 for s in distance.shape].index(True)
            markers = np.expand_dims(markers, axis=expand_axis)
        else:
            coords = peak_local_max(distance, min_distance=min_distance)
            markers = np.zeros(distance.shape, dtype=bool)
            markers[tuple(coords.T)] = True

        markers, _ = ndi.label(markers)
        return energy, markers

    def _point_markers(binary, local_points, shed_box):
        markers = np.zeros(binary.shape, dtype=bool)
        for local_pt in local_points:
            markers[_translate_point_in_box(local_pt, shed_box)] = True

        markers, _ = ndi.label(markers)
        energy = binary
        return energy, markers

    @magicgui(
        call_button="Split labels",
        layout="vertical",
        min_distance={
            "widget_type": "Slider",
            "label": "Minimum Distance",
            "min": 1,
            "max": 100,
            "value": 10,
            "tooltip": "Min Distance between Markers",
        },
        points_as_markers={
            "widget_type": "CheckBox",
            "text": "Use points as markers",
            "value": False,
            "tooltip": "Whether to use the placed points as markers for watershed. If checked, Min. Distance is ignored.",
        },
        apply3d={
            "widget_type": "CheckBox",
            "text": "Apply in 3D",
            "value": False,
            "tooltip": "Check box to split label in 3D.",
        },
        # new_label_header={"widget_type": "Label", "label": "<h3 text-align=\"center\">Specify new label value (optional)</h3>"},
        new_label={
            "widget_type": "CheckBox",
            "text": "Specify new label IDs (optional)",
            "value": False,
            "tooltip": "Whether to slect the new label IDs for the split labels",
        },
        start_label={
            "widget_type": "LineEdit",
            "label": "Start new label IDs from:",
            "value": "",
            "tooltip": "The label ID to start the new label IDs from.",
        },
    )
    def widget(
        viewer: napari.viewer.Viewer,
        labels_layer: napari.layers.Labels,
        points_layer: napari.layers.Points,
        min_distance: int,
        points_as_markers: bool,
        apply3d,
        # new_label_header,
        new_label: bool,
        start_label: int,
    ):
        if points_layer is None:
            points_layer = viewer.add_points([])
            points_layer.mode = "ADD"
            return

        labels = labels_layer.data
        world_points = points_layer.data

        if apply3d and labels.ndim != 3:
            print("Apply 3D checked, but labels are not 3D. Ignoring.")

        # get points as indices in local coordinates
        local_points = map_points(world_points, labels_layer)

        if type(labels) is da.core.Array:
            raise Exception(
                "Split operation is not supported on Dask Array labels!"
            )

        label_ids = np.array([labels[pt].item() for pt in local_points])
        local_points = np.stack(local_points, axis=0)

        # drop any label_ids equal to 0; in case point
        # was placed on the background
        background_pts = label_ids == 0
        local_points = local_points[~background_pts]
        label_ids = label_ids[~background_pts]

        if len(label_ids) == 0:
            print("No labels selected!")
            return

        # group local_points by label_ids
        labels_points = {
            label_id: local_points[label_ids == label_id]
            for label_id in np.unique(label_ids)
        }

        for label_id, local_points in labels_points.items():
            if labels.ndim == 2 or (labels.ndim == 3 and apply3d):
                shed_box = [
                    rp.bbox
                    for rp in regionprops(labels)
                    if rp.label == label_id
                ][0]
                binary = crop_and_binarize(labels, shed_box, label_id)

                if points_as_markers:
                    energy, markers = _point_markers(
                        binary, local_points, shed_box
                    )
                else:
                    energy, markers = _distance_markers(binary, min_distance)

                marker_ids = np.unique(markers)[1:]

                if len(marker_ids) > 1:
                    new_labels = watershed(energy, markers, mask=binary)
                    slices = _box_to_slice(shed_box)

                    if new_label:
                        new_label_id = int(start_label) - 1
                        max_label = new_label_id
                    else:
                        max_label = labels.max()

                    # Check if any of the new label IDs are already in use
                    new_labels_exist = any(
                        labels.max() >= (marker_ids + max_label)
                    )
                    if new_labels_exist:
                        print(
                            f"Label ID {start_label} is already in use. Please specify new label IDs."
                        )
                    else:
                        labels[slices][binary] = new_labels[binary] + max_label
                        print(
                            f"Split label {label_id} to {marker_ids + max_label}"
                        )
                else:
                    print("Nothing to split.")

            elif labels.ndim == 3:
                # get the current viewer axis
                axis = viewer.dims.order[0]
                plane = local_points[0][axis]
                labels2d = take(labels, plane, axis)
                assert all(
                    local_pt[axis] == plane for local_pt in local_points
                )

                shed_box = [
                    rp.bbox
                    for rp in regionprops(labels2d)
                    if rp.label == label_id
                ][0]
                binary = crop_and_binarize(labels2d, shed_box, label_id)

                if points_as_markers:
                    local_points2d = []
                    for lp in local_points:
                        local_points2d.append(
                            [p for i, p in enumerate(lp) if i != axis]
                        )
                    energy, markers = _point_markers(
                        binary, local_points2d, shed_box
                    )
                else:
                    energy, markers = _distance_markers(binary, min_distance)

                marker_ids = np.unique(markers)[1:]

                if len(marker_ids) > 1:
                    new_labels = watershed(energy, markers, mask=binary)
                    slices = _box_to_slice(shed_box)

                    if new_label:
                        new_label_id = int(start_label) - 1
                        max_label = new_label_id
                    else:
                        max_label = labels2d.max()
                        # Check if any of the new label IDs are already in use
                    new_labels_exist = any(
                        labels2d.max() >= (marker_ids + max_label)
                    )
                    if new_labels_exist:
                        print(
                            f"Label ID {start_label} is already in use. Please specify new label IDs."
                        )
                    else:
                        labels2d[slices][binary] = (
                            new_labels[binary] + max_label
                        )
                        print(
                            f"Split label {label_id} to {marker_ids + max_label}"
                        )
                else:
                    print("Nothing to split.")

                put(labels, local_points[0][axis], labels2d, axis)

            elif labels.ndim == 4:
                # get the current viewer axes
                assert (
                    viewer.dims.order[0] == 0
                ), "Dims expected to be (0, 1, 2, 3) for 4D labels!"
                assert (
                    viewer.dims.order[1] == 1
                ), "Dims expected to be (0, 1, 2, 3) for 4D labels!"
                plane1 = local_points[0][0]
                plane2 = local_points[0][1]
                assert all(local_pt[0] == plane1 for local_pt in local_points)
                assert all(local_pt[1] == plane2 for local_pt in local_points)

                labels2d = labels[plane1, plane2]

                shed_box = [
                    rp.bbox
                    for rp in regionprops(labels2d)
                    if rp.label == label_id
                ][0]
                binary = crop_and_binarize(labels2d, shed_box, label_id)

                if points_as_markers:
                    energy, markers = _point_markers(
                        binary, local_points, shed_box
                    )
                else:
                    energy, markers = _distance_markers(binary, min_distance)

                marker_ids = np.unique(markers)[1:]

                if len(marker_ids) > 1:
                    new_labels = watershed(energy, markers, mask=binary)
                    slices = _box_to_slice(shed_box)

                    if new_label:
                        new_label_id = int(start_label) - 1
                        max_label = new_label_id
                    else:
                        max_label = labels2d.max()
                    # Check if any of the new label IDs are already in use
                    new_labels_exist = any(
                        labels2d.max() >= (marker_ids + max_label)
                    )
                    if new_labels_exist:
                        print(
                            f"Label ID {start_label} is already in use. Please specify new label IDs."
                        )
                    else:
                        labels2d[slices][binary] = (
                            new_labels[binary] + max_label
                        )
                        print(
                            f"Split label {label_id} to {marker_ids + max_label}"
                        )
                else:
                    print("Nothing to split.")

                labels[local_points[0][0], local_points[0][1]] = labels2d

        labels_layer.data = labels
        points_layer.data = []

    return widget


def filter_labels():
    @magicgui(
        call_button="Apply Filter",
        layout="vertical",
        min_size={
            "widget_type": "SpinBox",
            "label": "Minimum Size (voxels)",
            "min": 0,
            "value": 64,
            "tooltip": "Minimum size in voxels",
        },
        sigma={
            "widget_type": "FloatSpinBox",
            "label": "Sigma",
            "min": 0.0,
            "value": 0.0,
            "step": 0.1,
            "tooltip": "Outlier threshold in standard deviations",
        },
        min_z_slices={
            "widget_type": "SpinBox",
            "label": "Min Z Slices",
            "min": 0,
            "value": 4,
            "tooltip": "Minimum number of consecutive Z slices a label must exist in",
        },
        relabel={
            "widget_type": "CheckBox",
            "text": "Relabel after filtering",
            "value": True,
            "tooltip": "Relabel connected components after filtering",
        },
    )
    def widget(
        viewer: napari.viewer.Viewer,
        labels_layer: napari.layers.Labels,
        min_size: int,
        sigma: float,
        min_z_slices: int,
        relabel: bool,
    ):
        if labels_layer is None:
            print("No labels layer selected!")
            return

        labels_array = labels_layer.data.copy()

        if len(labels_array.shape) > 3 and 1 in labels_array.shape:
            channels_dim = labels_array.shape.index(1)
            labels_array = np.squeeze(labels_array, axis=channels_dim)
        elif len(labels_array.shape) == 3:
            channels_dim = None
        else:
            raise ValueError(
                "Labels array has more than 3 dimensions and num_channels > 1!"
            )

        all_ids, id_counts = np.unique(labels_array, return_counts=True)

        # Initialize filtered_ids with all non-zero IDs
        filtered_ids = all_ids[all_ids != 0]

        if len(all_ids) == 0:
            print("No labels to filter!")
            return

        if min_size > 0:
            # Filter by size
            filtered_ids = filtered_ids[id_counts[all_ids != 0] >= min_size]
            print(f"After size filter: {len(filtered_ids)} ids")

        if sigma > 0:
            # Get mean and std of counts for surviving IDs
            surviving_counts = id_counts[np.isin(all_ids, filtered_ids)]
            mean, std = np.mean(surviving_counts), np.std(surviving_counts)
            filtered_ids = filtered_ids[
                np.abs(surviving_counts - mean) <= sigma * std
            ]
            print(f"After outlier removal: {len(filtered_ids)} ids")

        if min_z_slices > 1:
            # Find unique IDs by z-slice
            unique_ids_by_slice = [
                np.unique(labels_array[z])
                for z in range(labels_array.shape[0])
            ]
            # Find IDs that exist in at least N z-slices
            z_id_counts = np.array(
                [
                    np.sum(
                        [uid in slice_ids for slice_ids in unique_ids_by_slice]
                    )
                    for uid in filtered_ids
                ]
            )
            filtered_ids = filtered_ids[z_id_counts >= min_z_slices]
            print(f"After z-fragment removal: {len(filtered_ids)} ids")

        # IDs to remove
        ids_to_remove = np.setdiff1d(all_ids, filtered_ids)

        # Remove filtered out labels
        if len(ids_to_remove) > 0:
            labels_array[np.isin(labels_array, ids_to_remove)] = 0

            # Relabel connected components if requested
            if relabel:
                labels_array = label(labels_array, connectivity=1).astype(
                    labels_array.dtype
                )
                print("Re-labeled connected components")

            labels_layer.data = (
                np.expand_dims(labels_array, channels_dim)
                if channels_dim is not None
                else labels_array
            )
            print(f"Removed {len(ids_to_remove)} label IDs")
        else:
            print("No labels were filtered out")

    return widget


@napari_hook_implementation(specname="napari_experimental_provide_dock_widget")
def morph_labels_widget():
    return morph_labels, {"name": "Morph Labels"}


@napari_hook_implementation(specname="napari_experimental_provide_dock_widget")
def delete_labels_widget():
    return delete_labels, {"name": "Delete Labels"}


@napari_hook_implementation(specname="napari_experimental_provide_dock_widget")
def merge_labels_widget():
    return merge_labels, {"name": "Merge Labels"}


@napari_hook_implementation(specname="napari_experimental_provide_dock_widget")
def split_labels_widget():
    return split_labels, {"name": "Split Labels"}


@napari_hook_implementation(specname="napari_experimental_provide_dock_widget")
def filter_labels_widget():
    return filter_labels, {"name": "Filter Labels"}
