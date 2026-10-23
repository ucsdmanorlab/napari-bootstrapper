import cc3d
import dask.array as da
import fastremap
import fastmorph
import napari
import numpy as np
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy import ndimage as ndi
from skimage import draw
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed

from .._ui import (
    GROUP_SPACING,
    SECTION_INSET,
    align_label_columns,
    apply_plugin_theme,
    current_theme,
    format_label_grid,
    make_action_grid,
    make_section,
)

APPLY_TEXTS = {
    "Dilate": "Dilate labels",
    "Erode": "Erode labels",
    "Open": "Open labels",
    "Close": "Close labels",
    "Fill holes": "Fill holes",
    "Merge": "Merge labels",
    "Split": "Split labels",
    "Delete": "Delete labels",
    "Filter": "Filter labels",
}


def take(array, indices, axis=0):
    return np.take(array, indices, axis=axis)


def put(array, indices, value, axis=0):
    np.put_along_axis(
        array,
        np.expand_dims(
            np.full(value.shape, indices, dtype=int), axis=axis
        ),
        np.expand_dims(value, axis=axis),
        axis=axis,
    )


def crop_and_binarize(mask, box, label):
    n = len(box)
    n_dim = n // 2
    slices = tuple(slice(box[i], box[i + n_dim]) for i in range(n_dim))
    return mask[slices] == label


def map_points(world_points, labels_layer):
    local_points = []
    for wp in world_points:
        lp = labels_layer.world_to_data(wp)
        local_points.append(tuple(int(round(x)) for x in lp))
    return local_points


def _box_to_slice(shed_box):
    n = len(shed_box)
    n_dim = n // 2
    return tuple(
        slice(shed_box[i], shed_box[i + n_dim]) for i in range(n_dim)
    )


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
        t = np.full_like(x, planes[0])
        z = np.full_like(x, planes[1])
        indices = np.stack([t, z, y, x], axis=1)
    else:
        raise Exception("Lines must be 2D, 3D, or 4D!")
    return indices


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


def _resolve_label_ids(
    labels, label_ids_str, selection_layer, labels_layer, viewer
):
    """Resolve label IDs from the text entry or from a Points layer."""
    if label_ids_str:
        ids = [
            int(x.strip())
            for x in label_ids_str.split(",")
            if x.strip()
        ]
        return [x for x in ids if x > 0] or None

    if selection_layer is not None:
        sel_data = selection_layer.data
        has_data = (
            len(sel_data) > 0
            if isinstance(sel_data, (list, np.ndarray))
            else sel_data is not None
        )
        if not has_data:
            return None

        if isinstance(selection_layer, napari.layers.Points):
            local_points = map_points(sel_data, labels_layer)
            if isinstance(labels, da.core.Array):
                ids = [labels[pt].compute() for pt in local_points]
            else:
                ids = [labels[pt].item() for pt in local_points]
            return [x for x in ids if x > 0] or None

    return None


def _reset_selection(selection_layer):
    if selection_layer is not None:
        try:
            if len(selection_layer.data) > 0:
                selection_layer.data = []
        except Exception:
            pass


def _section_axis(labels_layer, viewer):
    """Return the labels axis the viewer is slicing, in the layer's own axes.

    Image layers with more dimensions (such as the stacked 2D predictions)
    add leading viewer axes, so viewer axis numbers cannot index the labels.
    """
    offset = viewer.dims.ndim - labels_layer.ndim
    axes = [a - offset for a in viewer.dims.order if a >= offset]
    hidden = [a for a in axes if a + offset not in viewer.dims.displayed]
    return (hidden or axes)[0]


def _get_data_slice(labels, labels_layer, viewer, apply3d):
    if labels.ndim == 2 or (labels.ndim == 3 and apply3d):
        return labels, None
    point = [
        int(round(x))
        for x in labels_layer.world_to_data(viewer.dims.point)
    ]
    if labels.ndim == 3:
        axis = _section_axis(labels_layer, viewer)
        plane = point[axis]
        data_2d = take(labels, plane, axis)
        return data_2d, lambda result: put(labels, plane, result, axis)
    elif labels.ndim == 4:
        p0, p1 = point[0], point[1]
        data_2d = labels[p0, p1]
        return data_2d, lambda result: labels.__setitem__(
            (p0, p1), result
        )
    return labels, None


def _apply_morph(data, sub_op, op_style, radius, label_ids):
    morph_ops_stenciled = {
        "Dilate": fastmorph.dilate,
        "Erode": fastmorph.erode,
        "Open": fastmorph.opening,
        "Close": fastmorph.closing,
    }
    morph_ops_spherical = {
        "Dilate": fastmorph.spherical_dilate,
        "Erode": fastmorph.spherical_erode,
        "Open": fastmorph.spherical_open,
        "Close": fastmorph.spherical_close,
    }

    def _run_op(arr):
        if op_style == "Stenciled":
            return morph_ops_stenciled[sub_op](arr)
        else:
            return morph_ops_spherical[sub_op](arr, radius=radius)

    sel_str = f" on labels {label_ids}" if label_ids else " on all labels"
    style_str = f"spherical, radius {radius}" if op_style == "Spherical" else "stenciled"
    print(f"Applying {sub_op} ({style_str}){sel_str}")

    if label_ids is not None:
        selected = fastremap.mask_except(data.copy(), label_ids)
        result = _run_op(selected)
        changed = result != selected
        data[changed] = result[changed]
    else:
        result = _run_op(data.copy())
        np.copyto(data, result)

    return data


def _apply_fill_holes(data, version, label_ids, **kwargs):
    sel_str = f" on labels {label_ids}" if label_ids else " on all labels"
    print(f"Applying fill_holes_{version}{sel_str}")

    def _run_fill(arr, ver, kw):
        # fill_holes requires 3D input
        squeezed = False
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
            squeezed = True

        if ver == "v1":
            # fill_holes_v1 returns array, or tuple if return_fill_count
            result = fastmorph.fill_holes_v1(arr, **kw)
            filled = (
                result[0] if isinstance(result, tuple) else result
            )
        else:
            # fill_holes_v2 returns (filled, holes)
            filled, _ = fastmorph.fill_holes_v2(arr, **kw)

        if squeezed:
            filled = filled[0]
        return filled

    try:
        if label_ids is not None:
            selected = fastremap.mask_except(data.copy(), label_ids)
            filled = _run_fill(selected, version, kwargs)
            changed = filled != selected
            data[changed] = filled[changed]
        else:
            filled = _run_fill(data, version, kwargs)
            np.copyto(data, filled)
    except fastmorph.FillError as exc:
        print(
            f"Fill holes made no change: {exc} "
            "Turn on Remove enclosed to fill over enclosed labels."
        )

    return data


def _apply_merge(labels, label_ids, labels_layer, viewer, apply3d):
    if label_ids is None or len(label_ids) < 2:
        print("Merging needs at least 2 labels!")
        return

    if labels_layer.selected_label in label_ids:
        target = labels_layer.selected_label
    else:
        target = min(label_ids)

    mapping = {lid: target for lid in label_ids if lid != target}

    data, write_back = _get_data_slice(
        labels, labels_layer, viewer, apply3d
    )
    fastremap.remap(
        data, mapping, preserve_missing_labels=True, in_place=True
    )
    if write_back:
        write_back(data)

    print(f"Merged labels {label_ids} into label {target}")


def _apply_split(
    labels,
    selection_layer,
    labels_layer,
    viewer,
    apply3d,
    min_distance,
    points_as_markers,
    new_label,
    start_label,
):
    if not isinstance(selection_layer, napari.layers.Points):
        print("Split requires a Points layer!")
        return

    world_points = selection_layer.data
    local_points = map_points(world_points, labels_layer)
    lp_arr = np.stack(local_points, axis=0)
    lp_label_ids = np.array(
        [labels[tuple(pt)].item() for pt in lp_arr]
    )

    fg = lp_label_ids > 0
    lp_arr = lp_arr[fg]
    lp_label_ids = lp_label_ids[fg]

    if len(lp_label_ids) == 0:
        print("No labels are selected!")
        return

    labels_points = {
        lid: lp_arr[lp_label_ids == lid]
        for lid in np.unique(lp_label_ids)
    }

    def _translate_pt(point, shed_box):
        n_dim = len(shed_box) // 2
        return tuple(
            int(point[i] - shed_box[i]) for i in range(n_dim)
        )

    def _distance_markers(binary, min_dist):
        distance = ndi.distance_transform_edt(binary)
        energy = -distance
        if 1 in distance.shape:
            coords = peak_local_max(
                np.squeeze(distance), min_distance=min_dist
            )
            markers = np.zeros(
                np.squeeze(distance).shape, dtype=bool
            )
            markers[tuple(coords.T)] = True
            expand_axis = [s == 1 for s in distance.shape].index(
                True
            )
            markers = np.expand_dims(markers, axis=expand_axis)
        else:
            coords = peak_local_max(distance, min_distance=min_dist)
            markers = np.zeros(distance.shape, dtype=bool)
            markers[tuple(coords.T)] = True
        markers, _ = ndi.label(markers)
        return energy, markers

    def _point_markers(binary, pts, shed_box):
        markers = np.zeros(binary.shape, dtype=bool)
        for pt in pts:
            markers[_translate_pt(pt, shed_box)] = True
        markers, _ = ndi.label(markers)
        return binary, markers

    for label_id, cur_points in labels_points.items():
        if labels.ndim == 2 or (labels.ndim == 3 and apply3d):
            shed_box = [
                rp.bbox
                for rp in regionprops(labels)
                if rp.label == label_id
            ][0]
            binary = crop_and_binarize(labels, shed_box, label_id)

            if points_as_markers:
                energy, markers = _point_markers(
                    binary, cur_points, shed_box
                )
            else:
                energy, markers = _distance_markers(
                    binary, min_distance
                )

            marker_ids = np.unique(markers)[1:]
            if len(marker_ids) > 1:
                new_labels = watershed(energy, markers, mask=binary)
                slices = _box_to_slice(shed_box)
                if new_label and start_label:
                    max_label = int(start_label) - 1
                else:
                    max_label = labels.max()
                labels[slices][binary] = (
                    new_labels[binary] + max_label
                )
                print(
                    f"Split label {label_id} into labels "
                    f"{marker_ids + max_label}"
                )
            else:
                print("Nothing to split.")

        elif labels.ndim == 3:
            axis = _section_axis(labels_layer, viewer)
            plane = cur_points[0][axis]
            labels2d = take(labels, plane, axis)

            shed_box = [
                rp.bbox
                for rp in regionprops(labels2d)
                if rp.label == label_id
            ][0]
            binary = crop_and_binarize(labels2d, shed_box, label_id)

            if points_as_markers:
                pts_2d = [
                    [p for i, p in enumerate(lp) if i != axis]
                    for lp in cur_points
                ]
                energy, markers = _point_markers(
                    binary, pts_2d, shed_box
                )
            else:
                energy, markers = _distance_markers(
                    binary, min_distance
                )

            marker_ids = np.unique(markers)[1:]
            if len(marker_ids) > 1:
                new_labels = watershed(energy, markers, mask=binary)
                slices = _box_to_slice(shed_box)
                if new_label and start_label:
                    max_label = int(start_label) - 1
                else:
                    max_label = labels2d.max()
                labels2d[slices][binary] = (
                    new_labels[binary] + max_label
                )
                print(
                    f"Split label {label_id} into labels "
                    f"{marker_ids + max_label}"
                )
            else:
                print("Nothing to split.")

            put(labels, plane, labels2d, axis)


def _apply_delete(labels, label_ids, labels_layer, viewer, apply3d):
    if label_ids is None or len(label_ids) == 0:
        print("No labels are selected for deletion!")
        return

    data, write_back = _get_data_slice(
        labels, labels_layer, viewer, apply3d
    )
    result = fastremap.mask(data, label_ids)
    np.copyto(data, result)
    if write_back:
        write_back(data)

    print(f"Deleted labels {label_ids}")


def _apply_filter(
    labels,
    min_size,
    max_size,
    sigma,
    min_z_slices,
    largest_k_val,
    do_relabel,
):
    data = labels
    print(
        f"Applying filter (min_size={min_size}, max_size={max_size}, "
        f"sigma={sigma}, min_z={min_z_slices}, "
        f"largest_k={largest_k_val}, relabel={do_relabel})"
    )

    if min_size > 0:
        cc3d.dust(data, threshold=min_size, in_place=True)
        n_remaining = len(fastremap.unique(data)) - 1
        print(f"After min-size filter: {n_remaining} labels")

    if max_size > 0:
        uniq, counts = fastremap.unique(data, return_counts=True)
        to_remove = [
            int(uid)
            for uid, c in zip(uniq, counts)
            if uid > 0 and c > max_size
        ]
        if to_remove:
            result = fastremap.mask(data, to_remove)
            np.copyto(data, result)
        n_remaining = len(fastremap.unique(data)) - 1
        print(f"After max-size filter: {n_remaining} labels")

    if largest_k_val > 0:
        result = cc3d.largest_k(data, k=largest_k_val)
        np.copyto(data, result)
        n_remaining = len(fastremap.unique(data)) - 1
        print(f"After largest-k: {n_remaining} labels")

    if sigma > 0:
        uniq, counts = fastremap.unique(data, return_counts=True)
        fg_mask = uniq > 0
        fg_labels = uniq[fg_mask]
        fg_counts = counts[fg_mask]
        if len(fg_counts) > 0:
            mean_c = fg_counts.mean()
            std_c = fg_counts.std()
            outliers = [
                int(lid)
                for lid, c in zip(fg_labels, fg_counts)
                if abs(c - mean_c) > sigma * std_c
            ]
            if outliers:
                result = fastremap.mask(data, outliers)
                np.copyto(data, result)
            n_remaining = len(fastremap.unique(data)) - 1
            print(f"After sigma filter: {n_remaining} labels")

    if min_z_slices > 1 and data.ndim == 3:
        uniq = fastremap.unique(data)
        uniq = uniq[uniq > 0]
        to_remove = []
        for uid in uniq:
            z_count = sum(
                1
                for z in range(data.shape[0])
                if uid in data[z]
            )
            if z_count < min_z_slices:
                to_remove.append(int(uid))
        if to_remove:
            result = fastremap.mask(data, to_remove)
            np.copyto(data, result)
        n_remaining = len(fastremap.unique(data)) - 1
        print(f"After z-filter: {n_remaining} labels")

    if do_relabel:
        result = cc3d.connected_components(data)
        np.copyto(data, result.astype(data.dtype))
        print("Relabeled the connected components")

    return data


class ProofreadingWidget(QMainWindow):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._undo_state = None

        self.scroll = QScrollArea()
        main_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        target_grid = QGridLayout()
        format_label_grid(target_grid, inset=SECTION_INSET)
        self._target_label = QLabel("Labels layer")
        target_grid.addWidget(self._target_label, 0, 0)
        self.labels_selector = QComboBox()
        self.labels_selector.setToolTip("Labels layer to operate on")
        target_grid.addWidget(self.labels_selector, 0, 1)
        layout.addLayout(target_grid)

        sel_grid = QGridLayout()

        self._points_label = QLabel("Points layer")
        sel_grid.addWidget(self._points_label, 0, 0)
        self.selection_selector = QComboBox()
        self.selection_selector.setToolTip(
            "Points layer whose points select labels"
        )
        self.selection_selector.addItem("(none)")
        sel_grid.addWidget(self.selection_selector, 0, 1)

        self._label_ids_label = QLabel("Label IDs")
        sel_grid.addWidget(self._label_ids_label, 1, 0)
        self.label_ids_input = QLineEdit()
        self.label_ids_input.setPlaceholderText("1, 5, 12 (overrides points)")
        self.label_ids_input.setToolTip(
            "Comma-separated label IDs. They override the points layer."
        )
        sel_grid.addWidget(self.label_ids_input, 1, 1)

        self.selection_box = make_section(
            "Selection",
            sel_grid,
            "Choose the objects to edit with a points layer or label IDs.",
        )
        layout.addWidget(self.selection_box)

        op_layout = QVBoxLayout()
        op_layout.setSpacing(6)

        self.group_selector = QComboBox()
        self.group_selector.addItems(
            ["Morphology", "Merge / Split", "Filter"]
        )
        self.group_selector.setToolTip("Operation category")
        op_layout.addWidget(self.group_selector)

        self.morph_container = QWidget()
        morph_layout = QVBoxLayout()
        morph_layout.setContentsMargins(0, 0, 0, 0)
        morph_layout.setSpacing(6)

        morph_op_grid = QGridLayout()
        morph_op_grid.setSpacing(6)
        self.morph_op_group = QButtonGroup(self)
        self.morph_radios = {}
        morph_tooltips = {
            "Dilate": "Dilate foreground labels",
            "Erode": "Erode foreground labels",
            "Open": "Morphological opening (erode then dilate)",
            "Close": "Morphological closing (dilate then erode)",
            "Fill holes": "Fill holes in topologically closed objects"
            " (github.com/seung-lab/fastmorph)",
        }
        morph_positions = {
            "Dilate": (0, 0),
            "Erode": (1, 0),
            "Open": (0, 1),
            "Close": (1, 1),
            "Fill holes": (0, 2),
        }
        for op in ["Dilate", "Erode", "Open", "Close", "Fill holes"]:
            rb = QRadioButton(op)
            rb.setToolTip(morph_tooltips[op])
            self.morph_op_group.addButton(rb)
            self.morph_radios[op] = rb
            morph_op_grid.addWidget(rb, *morph_positions[op])
        for column in range(3):
            morph_op_grid.setColumnStretch(column, 1)
        self.morph_radios["Dilate"].setChecked(True)
        morph_layout.addLayout(morph_op_grid)

        style_grid = QGridLayout()
        format_label_grid(style_grid)
        self.style_group = QButtonGroup(self)
        self.style_spherical = QRadioButton("Spherical")
        self.style_stenciled = QRadioButton("Stenciled")
        self.style_spherical.setToolTip("Variable radius")
        self.style_stenciled.setToolTip("Fast 3x3x3 kernel")
        self.style_group.addButton(self.style_spherical)
        self.style_group.addButton(self.style_stenciled)
        self.style_spherical.setChecked(True)
        self._style_label = QLabel("Style")
        style_grid.addWidget(self._style_label, 0, 0)
        style_grid.addWidget(self.style_spherical, 0, 1)
        style_grid.addWidget(self.style_stenciled, 1, 1)

        self.radius_label = QLabel("Radius")
        self.radius_slider = QSlider(Qt.Horizontal)
        self.radius_slider.setMinimum(2)
        self.radius_slider.setMaximum(7)
        self.radius_slider.setValue(2)
        self.radius_slider.setToolTip("Radius for spherical operations")
        self.radius_value_label = QLabel("2")
        self.radius_slider.valueChanged.connect(
            lambda v: self.radius_value_label.setText(str(v))
        )
        radius_row = QHBoxLayout()
        radius_row.setSpacing(6)
        radius_row.addWidget(self.radius_slider, 1)
        radius_row.addWidget(self.radius_value_label)
        style_grid.addWidget(self.radius_label, 2, 0)
        style_grid.addLayout(radius_row, 2, 1)

        self.style_container = QWidget()
        self.style_container.setLayout(style_grid)
        morph_layout.addWidget(self.style_container)

        self.fill_container = QWidget()
        fill_layout = QVBoxLayout()
        fill_layout.setContentsMargins(0, 0, 0, 0)
        fill_layout.setSpacing(6)

        fill_ver_grid = QGridLayout()
        format_label_grid(fill_ver_grid)
        self.fill_ver_group = QButtonGroup(self)
        self.fill_v1_radio = QRadioButton("v1")
        self.fill_v1_radio.setToolTip(
            "Iterative hole filling"
            " (see github.com/seung-lab/fastmorph)"
        )
        self.fill_v2_radio = QRadioButton("v2")
        self.fill_v2_radio.setToolTip(
            "Threshold-based hole filling"
            " (see github.com/seung-lab/fastmorph)"
        )
        self.fill_ver_group.addButton(self.fill_v1_radio)
        self.fill_ver_group.addButton(self.fill_v2_radio)
        self.fill_v1_radio.setChecked(True)
        self._version_label = QLabel("Version")
        version_row = QHBoxLayout()
        version_row.setSpacing(6)
        version_row.addWidget(self.fill_v1_radio)
        version_row.addWidget(self.fill_v2_radio)
        version_row.addStretch(1)
        fill_ver_grid.addWidget(self._version_label, 0, 0)
        fill_ver_grid.addLayout(version_row, 0, 1)
        fill_layout.addLayout(fill_ver_grid)

        self.fill_v1_container = QWidget()
        v1_grid = QGridLayout()
        v1_grid.setContentsMargins(0, 0, 0, 0)
        v1_grid.setVerticalSpacing(6)
        self.morph_closing_cb = QCheckBox("Morphological closing")
        self.morph_closing_cb.setToolTip(
            "Dilate first, then erode at the end of hole filling"
        )
        self.remove_enclosed_cb = QCheckBox("Remove enclosed")
        self.remove_enclosed_cb.setToolTip(
            "If one label totally encloses another, remove the interior label"
        )
        self.fix_borders_v1_cb = QCheckBox("Fix borders")
        self.fix_borders_v1_cb.setToolTip(
            "Run a 2D fill along the image edges of each binary image"
        )
        v1_grid.addWidget(self.morph_closing_cb, 0, 0)
        v1_grid.addWidget(self.remove_enclosed_cb, 1, 0)
        v1_grid.addWidget(self.fix_borders_v1_cb, 2, 0)
        self.fill_v1_container.setLayout(v1_grid)
        fill_layout.addWidget(self.fill_v1_container)

        self.fill_v2_container = QWidget()
        v2_grid = QGridLayout()
        format_label_grid(v2_grid)
        self.fix_borders_v2_cb = QCheckBox("Fix borders")
        self.fix_borders_v2_cb.setToolTip(
            "Along each image edge, try filling each label as a binary image"
        )
        self._merge_threshold_name = QLabel("Merge threshold")
        self.merge_threshold_slider = QSlider(Qt.Horizontal)
        self.merge_threshold_slider.setMinimum(0)
        self.merge_threshold_slider.setMaximum(100)
        self.merge_threshold_slider.setValue(100)
        self.merge_threshold_slider.setToolTip(
            "Fraction of a hole's surface (0.0 to 1.0) that one label"
            " must enclose"
        )
        self.merge_threshold_label = QLabel("1.00")
        self.merge_threshold_slider.valueChanged.connect(
            lambda v: self.merge_threshold_label.setText(
                f"{v / 100:.2f}"
            )
        )
        threshold_row = QHBoxLayout()
        threshold_row.setSpacing(6)
        threshold_row.addWidget(self.merge_threshold_slider, 1)
        threshold_row.addWidget(self.merge_threshold_label)
        v2_grid.addWidget(self._merge_threshold_name, 0, 0)
        v2_grid.addLayout(threshold_row, 0, 1)
        v2_grid.addWidget(self.fix_borders_v2_cb, 1, 0, 1, 2)
        self.fill_v2_container.setLayout(v2_grid)
        fill_layout.addWidget(self.fill_v2_container)

        self.fill_container.setLayout(fill_layout)
        morph_layout.addWidget(self.fill_container)

        self.morph_container.setLayout(morph_layout)
        op_layout.addWidget(self.morph_container)

        self.edit_container = QWidget()
        edit_layout = QVBoxLayout()
        edit_layout.setContentsMargins(0, 0, 0, 0)
        edit_layout.setSpacing(6)

        edit_op_grid = QGridLayout()
        edit_op_grid.setSpacing(6)
        self.edit_op_group = QButtonGroup(self)
        self.edit_radios = {}
        edit_tooltips = {
            "Merge": "Combine the selected labels into one. The result keeps"
            " the layer's selected label if it is among them, else the"
            " smallest ID",
            "Split": "Split a label using watershed"
            " (place points as markers)",
            "Delete": "Remove the selected labels (set them to 0)",
        }
        for i, op in enumerate(["Merge", "Split", "Delete"]):
            rb = QRadioButton(op)
            rb.setToolTip(edit_tooltips[op])
            self.edit_op_group.addButton(rb)
            self.edit_radios[op] = rb
            edit_op_grid.addWidget(rb, 0, i)
            edit_op_grid.setColumnStretch(i, 1)
        self.edit_radios["Merge"].setChecked(True)
        edit_layout.addLayout(edit_op_grid)

        self.split_container = QWidget()
        split_grid = QGridLayout()
        format_label_grid(split_grid)

        self._min_distance_name = QLabel("Min distance")
        self.min_distance_slider = QSlider(Qt.Horizontal)
        self.min_distance_slider.setMinimum(1)
        self.min_distance_slider.setMaximum(100)
        self.min_distance_slider.setValue(10)
        self.min_distance_slider.setToolTip(
            "Minimum distance between markers"
        )
        self.min_distance_label = QLabel("10")
        self.min_distance_slider.valueChanged.connect(
            lambda v: self.min_distance_label.setText(str(v))
        )
        distance_row = QHBoxLayout()
        distance_row.setSpacing(6)
        distance_row.addWidget(self.min_distance_slider, 1)
        distance_row.addWidget(self.min_distance_label)
        split_grid.addWidget(self._min_distance_name, 0, 0)
        split_grid.addLayout(distance_row, 0, 1)

        self.points_as_markers_cb = QCheckBox("Use points as markers")
        self.points_as_markers_cb.setToolTip(
            "Use placed points as watershed markers"
        )
        self.points_as_markers_cb.toggled.connect(self._on_points_as_markers)
        split_grid.addWidget(self.points_as_markers_cb, 1, 0, 1, 2)

        self.new_label_cb = QCheckBox("Specify new label IDs")
        self.new_label_cb.setToolTip(
            "Assign specific label IDs to split fragments"
            " instead of auto-incrementing"
        )

        self.start_label_input = QLineEdit()
        self.start_label_input.setPlaceholderText("Start from...")
        self.start_label_input.setToolTip(
            "Starting label ID for split fragments (e.g. 100)"
        )
        self.start_label_input.setEnabled(False)
        self.new_label_cb.toggled.connect(
            self.start_label_input.setEnabled
        )
        new_label_row = QHBoxLayout()
        new_label_row.setSpacing(6)
        new_label_row.addWidget(self.new_label_cb)
        new_label_row.addWidget(self.start_label_input)
        split_grid.addLayout(new_label_row, 2, 0, 1, 2)

        self.split_container.setLayout(split_grid)
        edit_layout.addWidget(self.split_container)

        self.edit_container.setLayout(edit_layout)
        op_layout.addWidget(self.edit_container)

        self.filter_container = QWidget()
        filter_layout = QGridLayout()
        format_label_grid(filter_layout)

        dust_label = QLabel("Min size (voxels)")
        filter_layout.addWidget(dust_label, 0, 0)
        self.dust_spinbox = QSpinBox()
        self.dust_spinbox.setRange(0, 1000000)
        self.dust_spinbox.setValue(0)
        self.dust_spinbox.setSpecialValueText("Off")
        self.dust_spinbox.setToolTip(
            "Remove labels smaller than this. Off skips this filter."
        )
        filter_layout.addWidget(self.dust_spinbox, 0, 1)

        max_size_label = QLabel("Max size (voxels)")
        filter_layout.addWidget(max_size_label, 1, 0)
        self.max_size_spinbox = QSpinBox()
        self.max_size_spinbox.setRange(0, 10000000)
        self.max_size_spinbox.setValue(0)
        self.max_size_spinbox.setSpecialValueText("Off")
        self.max_size_spinbox.setToolTip(
            "Remove labels larger than this. Off skips this filter."
        )
        filter_layout.addWidget(self.max_size_spinbox, 1, 1)

        largest_k_label = QLabel("Largest K")
        filter_layout.addWidget(largest_k_label, 2, 0)
        self.largest_k_spinbox = QSpinBox()
        self.largest_k_spinbox.setRange(0, 1000000)
        self.largest_k_spinbox.setValue(0)
        self.largest_k_spinbox.setSpecialValueText("Off")
        self.largest_k_spinbox.setToolTip(
            "Keep only the K largest labels. Off skips this filter."
        )
        filter_layout.addWidget(self.largest_k_spinbox, 2, 1)

        sigma_label = QLabel("Sigma")
        filter_layout.addWidget(sigma_label, 3, 0)
        self.sigma_spinbox = QDoubleSpinBox()
        self.sigma_spinbox.setRange(0.0, 10.0)
        self.sigma_spinbox.setValue(0.0)
        self.sigma_spinbox.setSingleStep(0.1)
        self.sigma_spinbox.setDecimals(1)
        self.sigma_spinbox.setSpecialValueText("Off")
        self.sigma_spinbox.setToolTip(
            "Remove labels whose size lies more than this many standard"
            " deviations from the mean size. Off skips this filter."
        )
        filter_layout.addWidget(self.sigma_spinbox, 3, 1)

        min_z_label = QLabel("Min Z slices")
        filter_layout.addWidget(min_z_label, 4, 0)
        self.min_z_spinbox = QSpinBox()
        self.min_z_spinbox.setRange(0, 1000)
        self.min_z_spinbox.setValue(0)
        self.min_z_spinbox.setSpecialValueText("Off")
        self.min_z_spinbox.setToolTip(
            "Remove labels that appear in fewer Z slices than this."
            " Off skips this filter."
        )
        filter_layout.addWidget(self.min_z_spinbox, 4, 1)

        self.relabel_cb = QCheckBox("Relabel after filtering")
        self.relabel_cb.setChecked(True)
        self.relabel_cb.setToolTip(
            "Relabel the connected components after filtering"
        )
        filter_layout.addWidget(self.relabel_cb, 5, 0, 1, 2)

        self.filter_container.setLayout(filter_layout)
        op_layout.addWidget(self.filter_container)

        # Apply in 3D and the buttons set the scope and run the operation,
        # so a group gap separates them from the operation's options.
        op_layout.addSpacing(GROUP_SPACING - op_layout.spacing())
        self.apply3d_checkbox = QCheckBox("Apply in 3D")
        self.apply3d_checkbox.setToolTip(
            "Apply the operation to the full 3D volume "
            "instead of the current 2D slice"
        )
        op_layout.addWidget(self.apply3d_checkbox)

        self.apply_button = QPushButton("Apply")
        self.apply_button.setProperty("primary", True)
        self.apply_button.setToolTip("Apply the selected operation")
        self.apply_button.clicked.connect(self._on_apply)

        self.undo_button = QPushButton("Undo")
        self.undo_button.setToolTip("Undo the last operation")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self._on_undo)
        op_layout.addLayout(
            make_action_grid([[(self.apply_button, 2), (self.undo_button, 1)]])
        )

        self.operation_box = make_section(
            "Operation",
            op_layout,
            "Choose an operation, set its options, then apply it.",
        )
        layout.addWidget(self.operation_box)

        layout.addStretch()

        main_widget.setLayout(layout)

        self._common_labels = [
            self._target_label,
            self._points_label,
            self._label_ids_label,
        ]
        self._group_labels = {
            "Morphology": [
                self._style_label,
                self.radius_label,
                self._version_label,
                self._merge_threshold_name,
            ],
            "Merge / Split": [self._min_distance_name],
            "Filter": [
                dust_label,
                max_size_label,
                largest_k_label,
                sigma_label,
                min_z_label,
            ],
        }
        self._label_grids = [
            target_grid,
            sel_grid,
            style_grid,
            fill_ver_grid,
            v2_grid,
            split_grid,
            filter_layout,
        ]

        self.scroll.setWidget(main_widget)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )
        self.scroll.setWidgetResizable(True)
        self.setMinimumWidth(380)
        self.setCentralWidget(self.scroll)

        self.group_selector.currentTextChanged.connect(
            self._update_visibility
        )
        self.morph_op_group.buttonClicked.connect(
            self._update_visibility
        )
        self.style_group.buttonClicked.connect(
            self._update_visibility
        )
        self.fill_ver_group.buttonClicked.connect(
            self._update_visibility
        )
        self.edit_op_group.buttonClicked.connect(
            self._update_visibility
        )

        self.viewer.layers.events.inserted.connect(
            self._update_layer_selectors
        )
        self.viewer.layers.events.removed.connect(
            self._update_layer_selectors
        )

        self._populate_layer_selectors()
        self._update_visibility()
        self.viewer.events.theme.connect(self._apply_theme)
        self._apply_theme()

    def showEvent(self, event):
        super().showEvent(event)
        self._align_label_columns()

    def _apply_theme(self, event=None):
        apply_plugin_theme(
            self, current_theme(self.viewer), self._align_label_columns
        )

    def _align_label_columns(self):
        # The labels of the current operation group set the width, so the
        # fields stay in place while the user picks options in that group.
        group = self.group_selector.currentText()
        labels = self._common_labels + self._group_labels.get(group, [])
        align_label_columns(labels, self._label_grids)

    def _on_points_as_markers(self, checked):
        for widget in (
            self.min_distance_slider,
            self.min_distance_label,
            self._min_distance_name,
        ):
            widget.setEnabled(not checked)

    def _populate_layer_selectors(self):
        self.labels_selector.clear()
        self.selection_selector.clear()
        self.selection_selector.addItem("(none)")
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                self.labels_selector.addItem(layer.name)
            if isinstance(layer, napari.layers.Points):
                self.selection_selector.addItem(layer.name)

    def _update_layer_selectors(self, event=None):
        cur_labels = self.labels_selector.currentText()
        cur_sel = self.selection_selector.currentText()
        self._populate_layer_selectors()
        idx = self.labels_selector.findText(cur_labels)
        if idx >= 0:
            self.labels_selector.setCurrentIndex(idx)
        idx = self.selection_selector.findText(cur_sel)
        if idx >= 0:
            self.selection_selector.setCurrentIndex(idx)

    def _get_labels_layer(self):
        name = self.labels_selector.currentText()
        if name and name in self.viewer.layers:
            layer = self.viewer.layers[name]
            if isinstance(layer, napari.layers.Labels):
                return layer
        return None

    def _get_selection_layer(self):
        name = self.selection_selector.currentText()
        if name == "(none)" or not name:
            return None
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            if isinstance(layer, napari.layers.Points):
                return layer
        return None

    def _update_visibility(self, *_args):
        group = self.group_selector.currentText()
        is_morph = group == "Morphology"
        is_edit = group == "Merge / Split"
        is_filter = group == "Filter"

        self.morph_container.setVisible(is_morph)
        self.edit_container.setVisible(is_edit)
        self.filter_container.setVisible(is_filter)

        checked = self.morph_op_group.checkedButton()
        morph_op = checked.text() if checked else "Dilate"
        checked = self.edit_op_group.checkedButton()
        edit_op = checked.text() if checked else "Merge"

        if is_morph:
            is_fill = morph_op == "Fill holes"
            self.style_container.setVisible(not is_fill)
            self.fill_container.setVisible(is_fill)

            if not is_fill:
                is_spherical = self.style_spherical.isChecked()
                self.radius_slider.setVisible(is_spherical)
                self.radius_label.setVisible(is_spherical)
                self.radius_value_label.setVisible(is_spherical)
            else:
                is_v1 = self.fill_v1_radio.isChecked()
                self.fill_v1_container.setVisible(is_v1)
                self.fill_v2_container.setVisible(not is_v1)

        if is_edit:
            self.split_container.setVisible(edit_op == "Split")

        sel_relevant = not is_filter
        self.selection_box.setEnabled(sel_relevant)
        self.selection_selector.setEnabled(sel_relevant)
        self.label_ids_input.setEnabled(sel_relevant)
        self.apply3d_checkbox.setEnabled(not is_filter)

        if is_morph:
            self.apply_button.setText(APPLY_TEXTS[morph_op])
        elif is_edit:
            self.apply_button.setText(APPLY_TEXTS[edit_op])
        else:
            self.apply_button.setText(APPLY_TEXTS["Filter"])

        self._align_label_columns()

    def _on_undo(self):
        if self._undo_state is None:
            return
        layer_name, saved_data = self._undo_state
        if layer_name in self.viewer.layers:
            self.viewer.layers[layer_name].data = saved_data
            print(f"Undo: restored {layer_name}")
        self._undo_state = None
        self.undo_button.setEnabled(False)

    def _on_apply(self):
        labels_layer = self._get_labels_layer()
        if labels_layer is None:
            print("No labels layer is selected!")
            return

        selection_layer = self._get_selection_layer()
        labels = labels_layer.data

        self._undo_state = (
            labels_layer.name,
            labels_layer.data.copy(),
        )
        self.undo_button.setEnabled(True)

        if isinstance(labels, da.core.Array):
            print(
                "Warning: Dask arrays may not support all operations."
            )

        label_ids = _resolve_label_ids(
            labels,
            self.label_ids_input.text().strip(),
            selection_layer,
            labels_layer,
            self.viewer,
        )

        group = self.group_selector.currentText()
        apply3d = self.apply3d_checkbox.isChecked()

        if group == "Morphology":
            checked = self.morph_op_group.checkedButton()
            morph_op = checked.text() if checked else "Dilate"

            data, write_back = _get_data_slice(
                labels, labels_layer, self.viewer, apply3d
            )

            if morph_op == "Fill holes":
                version = (
                    "v1"
                    if self.fill_v1_radio.isChecked()
                    else "v2"
                )
                kwargs = {}
                if version == "v1":
                    kwargs["morphological_closing"] = (
                        self.morph_closing_cb.isChecked()
                    )
                    kwargs["remove_enclosed"] = (
                        self.remove_enclosed_cb.isChecked()
                    )
                    kwargs["fix_borders"] = (
                        self.fix_borders_v1_cb.isChecked()
                    )
                else:
                    kwargs["merge_threshold"] = (
                        self.merge_threshold_slider.value() / 100.0
                    )
                    kwargs["fix_borders"] = (
                        self.fix_borders_v2_cb.isChecked()
                    )
                _apply_fill_holes(
                    data, version, label_ids, **kwargs
                )
            else:
                op_style = (
                    "Spherical"
                    if self.style_spherical.isChecked()
                    else "Stenciled"
                )
                radius = self.radius_slider.value()
                _apply_morph(
                    data, morph_op, op_style, radius, label_ids
                )

            if write_back:
                write_back(data)
            labels_layer.data = labels

        elif group == "Merge / Split":
            checked = self.edit_op_group.checkedButton()
            edit_op = checked.text() if checked else "Merge"

            if edit_op == "Merge":
                if label_ids is None or len(label_ids) < 2:
                    print("Merging needs at least 2 labels!")
                    _reset_selection(selection_layer)
                    return
                _apply_merge(
                    labels,
                    label_ids,
                    labels_layer,
                    self.viewer,
                    apply3d,
                )
                labels_layer.data = labels

            elif edit_op == "Split":
                _apply_split(
                    labels,
                    selection_layer,
                    labels_layer,
                    self.viewer,
                    apply3d,
                    self.min_distance_slider.value(),
                    self.points_as_markers_cb.isChecked(),
                    self.new_label_cb.isChecked(),
                    self.start_label_input.text().strip(),
                )
                labels_layer.data = labels

            elif edit_op == "Delete":
                if label_ids is None or len(label_ids) == 0:
                    print("No labels are selected for deletion!")
                    _reset_selection(selection_layer)
                    return
                _apply_delete(
                    labels, label_ids, labels_layer, self.viewer, apply3d
                )
                labels_layer.data = labels

        elif group == "Filter":
            if labels.ndim > 3 and 1 in labels.shape:
                channels_dim = labels.shape.index(1)
                data = np.squeeze(labels, axis=channels_dim)
            elif labels.ndim <= 3:
                channels_dim = None
                data = labels
            else:
                print(
                    "The filter does not support multi-channel labels with"
                    " more than 3 dimensions!"
                )
                return

            sigma_val = self.sigma_spinbox.value()
            _apply_filter(
                data,
                self.dust_spinbox.value(),
                self.max_size_spinbox.value(),
                sigma_val,
                self.min_z_spinbox.value(),
                self.largest_k_spinbox.value(),
                self.relabel_cb.isChecked(),
            )

            if channels_dim is not None:
                labels_layer.data = np.expand_dims(
                    data, channels_dim
                )
            else:
                labels_layer.data = data

        _reset_selection(selection_layer)


def proofreading(napari_viewer):
    return ProofreadingWidget(napari_viewer)


@napari_hook_implementation(
    specname="napari_experimental_provide_dock_widget"
)
def proofreading_widget():
    return proofreading, {"name": "Proofreading"}
