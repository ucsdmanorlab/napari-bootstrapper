import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import contextlib
import hashlib
import sys
import zipfile
from copy import deepcopy
from pathlib import Path

import gunpowder as gp
import numpy as np
import pyqtgraph as pg
import requests
import torch
from napari.layers import Image, Labels
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from tqdm import tqdm

from ._ui import (
    GROUP_SPACING,
    SECTION_INSET,
    align_label_columns,
    apply_plugin_theme,
    current_theme,
    format_label_grid,
    logo_pixmap,
    make_action_grid,
    make_collapsible,
    make_section,
    style_loss_plot,
)
from .datasets import stack_samples
from .datasets.napari_2d_dataset import Napari2DDataset
from .datasets.napari_3d_dataset import Napari3DDataset
from .gp.napari_image_source import NapariImageSource
from .gp.np_source import NpArraySource
from .gp.torch_predict import torchPredict
from .models import (
    DEFAULT_2D_MODEL_CONFIG,
    DEFAULT_3D_MODEL_CONFIG,
    PRETRAINED_3D_MODEL_SHA256,
    PRETRAINED_3D_MODEL_URLS,
    get_2d_model,
    get_3d_model,
    get_loss,
)
from .post import DEFAULT_SEG_PARAMS, segment_affs


def autocast(device, enabled):
    """
    Returns a bfloat16 autocast context on MPS and on CUDA GPUs with native
    bfloat16 (compute capability 8.0 or newer), and a context that does
    nothing otherwise.
    """
    if enabled and (
        device.type == "mps"
        or (
            device.type == "cuda"
            and torch.cuda.is_bf16_supported()
            # is_bf16_supported() also accepts older GPUs that emulate
            # bfloat16 without bfloat16 hardware.
            and torch.cuda.get_device_capability(device)[0] >= 8
        )
    ):
        try:
            return torch.autocast(device.type, dtype=torch.bfloat16)
        except RuntimeError:
            pass
    return contextlib.nullcontext()


def train_iteration(
    batch,
    model,
    criterion,
    optimizer,
    device,
    dimension,
    mixed_precision=False,
):
    batch = [item.to(device) for item in batch]

    model.train()
    with autocast(device, mixed_precision):
        if dimension == "3d" and len(batch) > 3:
            outputs = model(batch[0], batch[1])
        else:
            outputs = model(batch[0])
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.float()
    else:
        outputs = tuple(output.float() for output in outputs)

    if dimension == "2d":
        if len(outputs) == 1:
            loss = criterion(outputs[0], batch[1], batch[2])
        else:
            loss = criterion(
                outputs[0],
                batch[1],
                batch[2],
                outputs[1],
                batch[3],
                batch[4],
            )
    else:
        if len(batch) == 3:
            loss = criterion(outputs, batch[1], batch[2])
        elif len(batch) == 4:
            loss = criterion(outputs, batch[2], batch[3])
        else:
            raise ValueError("Invalid batch size")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), outputs


class Widget(QMainWindow):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.scroll = QScrollArea()
        self.widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self.tmp_dir = Path.home() / ".napari_bootstrapper"
        self.tmp_dir.mkdir(exist_ok=True)

        self._row_labels = []
        self.grid_0 = QGridLayout()
        self.set_grid_0()
        self.grid_1 = QGridLayout()
        self.set_grid_1()
        self.grid_2 = QGridLayout()
        self.set_grid_2()
        self.grid_3 = QGridLayout()
        self.set_grid_3()
        self.grid_4 = QGridLayout()
        self.set_grid_4()
        self.grid_5 = QGridLayout()
        self.set_grid_5()
        self.grid_6 = QGridLayout()
        self.set_grid_6()

        self.data_box = make_section(
            "1. Data",
            self.grid_2,
            "Select the 3D image to train on and segment, its sparse labels,"
            " and a mask of the labelled voxels.",
        )
        self.model_2d_box = make_section(
            "2. 2D model",
            self.grid_3,
            "Train a 2D model on the labelled sections, or load a 2D"
            " checkpoint.",
            button=self.advanced_2d_config_button,
        )
        self.model_3d_box = make_section(
            "3. 3D model",
            self.grid_4,
            "Download the pretrained 3D model, or load or train your own.",
            button=self.advanced_3d_config_button,
        )
        self.seg_box = make_section(
            "4. Segmentation",
            self.grid_5,
            "Predict with both models, then segment the 3D affinities.",
            button=self.advanced_seg_config_button,
        )
        self._label_grids = [
            self.grid_1,
            self.grid_2,
            self.grid_3,
            self.grid_4,
            self.grid_5,
        ]

        layout.addLayout(self.grid_0)
        layout.addLayout(self.grid_1)
        layout.addWidget(self.data_box)
        layout.addWidget(self.model_2d_box)
        layout.addWidget(self.model_3d_box)
        layout.addWidget(self.seg_box)
        layout.addLayout(self.grid_6)
        layout.addStretch(1)
        self.widget.setLayout(layout)
        self.set_scroll_area()
        self.viewer.layers.events.inserted.connect(self.update_selectors)
        self.viewer.layers.events.removed.connect(self.update_selectors)
        self.viewer.events.theme.connect(self._apply_theme)
        self._apply_theme()

    def showEvent(self, event):
        super().showEvent(event)
        self._align_label_columns()
        # Redraws the wordmark at the device pixel ratio of the dock's screen.
        self._set_logo()

    def _apply_theme(self, event=None):
        self._theme = current_theme(self.viewer)
        apply_plugin_theme(self, self._theme, self._align_label_columns)
        self._loss_pen = pg.mkPen(self._theme.text.as_hex())
        style_loss_plot(self.losses_2d_widget, self._theme, self._loss_pen)
        style_loss_plot(self.losses_3d_widget, self._theme, self._loss_pen)
        self._set_link_labels(self._theme)
        self._set_logo()

    def _set_logo(self):
        self._logo_label.setPixmap(
            logo_pixmap(self._theme, self.devicePixelRatioF())
        )

    def _align_label_columns(self):
        align_label_columns(self._row_labels, self._label_grids)

    def _set_link_labels(self, theme):
        color = theme.text.as_hex()
        self._description_label.setText(
            "<small>A 2D-to-3D method for dense 3D segmentations from sparse"
            " 2D labels.<br>If you use this in your research, please"
            ' <a href="https://github.com/ucsdmanorlab/bootstrapper#citation"'
            f' style="color:{color};">cite us</a>.</small>'
        )
        self._feedback_label.setText(
            "<small>Please share any feedback <a"
            ' href="https://github.com/ucsdmanorlab/napari-bootstrapper'
            '/issues/new/choose"'
            f' style="color:{color};">here</a>.</small>'
        )

    def update_selectors(self, event):
        """
        Updates `raw_selector`, `labels_selector` and `mask_selector`
        when the user adds or removes a layer. Each picker lists only the
        layers of its own type.
        """
        name = f"{event.value}"
        for picker, kind in (
            (self.raw_selector, Image),
            (self.labels_selector, Labels),
            (self.mask_selector, Labels),
        ):
            index = picker.findText(name)
            if event.type == "removed":
                if index >= 0:
                    picker.removeItem(index)
            elif isinstance(event.value, kind) and index < 0:
                picker.addItem(name)

    def show_hide_channels_dim(self):
        if self.raw_selector.currentText() == "":
            self.channels_dim = None
            self.channels_dim_combo_box.setEnabled(False)
            self.channels_dim_combo_box.hide()
            self.channels_dim_label.hide()
            return

        raw_layer = self.viewer.layers[self.raw_selector.currentText()]
        raw_data = raw_layer.data

        if len(raw_data.shape) == 3:
            self.channels_dim = None
            self.channels_dim_combo_box.setEnabled(False)
            self.channels_dim_combo_box.hide()
            self.channels_dim_label.hide()

        elif len(raw_data.shape) == 4:
            self.channels_dim_combo_box.setEnabled(True)
            self.channels_dim_combo_box.show()
            self.channels_dim_label.show()

            if raw_layer.rgb:
                # RGB layers in napari always have channels last
                self.channels_dim = 3
                self.channels_dim_combo_box.setCurrentText("3")
                self.channels_dim_combo_box.setEnabled(False)
            elif 1 in raw_data.shape:
                self.channels_dim = raw_data.shape.index(1)
                self.channels_dim_combo_box.setCurrentText(
                    f"{self.channels_dim}"
                )
                self.channels_dim_combo_box.setEnabled(True)
            else:
                self.update_channels_dim()

        else:
            raise ValueError(
                f"Images must be 3D or 4D, but this image has shape {raw_data.shape}"
            )

    def update_channels_dim(self):
        self.channels_dim = int(self.channels_dim_combo_box.currentText())
        print(f"channels_dim: {self.channels_dim}")

    def update_3d_model_type_selector(self):
        self.model_3d_type_selector.clear()
        if self.model_2d_type_selector.currentText() in [
            "2d_affs",
            "2d_mtlsd",
        ]:
            self.model_3d_type_selector.addItems(["3d_affs_from_2d_affs"])
        if self.model_2d_type_selector.currentText() in ["2d_lsd", "2d_mtlsd"]:
            self.model_3d_type_selector.addItems(["3d_affs_from_2d_lsd"])
        if self.model_2d_type_selector.currentText() == "2d_mtlsd":
            self.model_3d_type_selector.addItems(["3d_affs_from_2d_mtlsd"])
            self.model_3d_type_selector.setCurrentText("3d_affs_from_2d_mtlsd")

    def set_grid_0(self):
        """
        Adds the Bootstrapper wordmark above the plugin description.
        `_set_logo` draws the wordmark and `_set_link_labels` sets the
        description text.
        """
        self._logo_label = QLabel(self)
        self._logo_label.setAccessibleName("Bootstrapper")
        self._logo_label.setToolTip("Bootstrapper")
        self._description_label = QLabel(self)
        self._description_label.setWordWrap(True)
        self._description_label.setOpenExternalLinks(True)
        self.grid_0.setContentsMargins(SECTION_INSET, 0, SECTION_INSET, 0)
        # The wordmark ends with an empty row of pixels below its descenders,
        # so it needs no extra space above the description.
        self.grid_0.setVerticalSpacing(0)
        self.grid_0.addWidget(self._logo_label, 0, 0, Qt.AlignLeft)
        self.grid_0.addWidget(self._description_label, 1, 0, 1, 1)

    def set_grid_1(self):
        """
        Adds the device selector for training and inference.
        """

        device_label = QLabel(self)
        device_label.setText("Device")
        device_label.setToolTip("Select the device for training and inference")
        self.device_combo_box = QComboBox(self)
        self.device_combo_box.setToolTip(
            "Select the device for training and inference"
        )
        self.device_combo_box.addItem("cpu")
        if torch.cuda.is_available():
            self.device_combo_box.addItem("cuda:0")
            self.device_combo_box.setCurrentText("cuda:0")
        if torch.backends.mps.is_available():
            self.device_combo_box.addItem("mps")
            self.device_combo_box.setCurrentText("mps")
        format_label_grid(self.grid_1, inset=SECTION_INSET)
        self.grid_1.addWidget(device_label, 0, 0, 1, 1)
        self.grid_1.addWidget(self.device_combo_box, 0, 1, 1, 1)
        self._row_labels.append(device_label)

    def set_grid_2(self):
        """
        Adds the image, labels and mask selectors and the Make mask button.
        """
        raw_label = QLabel(self)
        raw_label.setText("Image layer")
        raw_label.setToolTip("Select the image layer to train on")
        self.raw_selector = QComboBox(self)
        self.raw_selector.setToolTip(
            "Select the image layer to train on"
        )
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                self.raw_selector.addItem(f"{layer}")

        self.channels_dim = None
        self.channels_dim_label = QLabel(self)
        self.channels_dim_label.setText("Channel axis")
        self.channels_dim_label.setToolTip(
            "The image has 4 dimensions. Select its channel axis, for example"
            " 3 for DHWC or 0 for CDHW."
        )
        self.channels_dim_combo_box = QComboBox(self)
        self.channels_dim_combo_box.setToolTip(
            "Select the channel axis, for example 3 for DHWC or 0 for CDHW."
        )
        self.channels_dim_combo_box.addItem("0")
        self.channels_dim_combo_box.addItem("1")
        self.channels_dim_combo_box.addItem("2")
        self.channels_dim_combo_box.addItem("3")

        self.channels_dim_combo_box.setCurrentText(f"{self.channels_dim}")
        self.channels_dim_label.hide()
        self.channels_dim_combo_box.hide()
        self.show_hide_channels_dim()

        self.raw_selector.currentTextChanged.connect(
            self.show_hide_channels_dim
        )
        self.channels_dim_combo_box.currentTextChanged.connect(
            self.update_channels_dim
        )

        labels_label = QLabel(self)
        labels_label.setText("Labels layer")
        labels_label.setToolTip("Select the labels layer to use for training")
        self.labels_selector = QComboBox(self)
        self.labels_selector.setToolTip(
            "Select the labels layer to use for training"
        )
        for layer in self.viewer.layers:
            if isinstance(layer, Labels):
                self.labels_selector.addItem(f"{layer}")

        self.make_mask_button = QPushButton(self)
        self.make_mask_button.setText("Make mask")
        self.make_mask_button.setToolTip(
            "Make a binary mask from the labels data"
            " (1 where labeled, 0 elsewhere)"
        )
        self.make_mask_button.clicked.connect(self.make_mask)

        mask_label = QLabel(self)
        mask_label.setText("Mask layer")
        mask_label.setToolTip(
            "Select the mask layer to use for training. Training computes the"
            " loss only where the mask is 1"
        )
        self.mask_selector = QComboBox(self)
        self.mask_selector.setToolTip(
            "Select the mask layer. Training computes the loss only where the"
            " mask is 1"
        )
        for layer in self.viewer.layers:
            if isinstance(layer, Labels):
                self.mask_selector.addItem(f"{layer}")

        self.grid_2.addWidget(raw_label, 0, 0, 1, 1)
        self.grid_2.addWidget(self.raw_selector, 0, 1, 1, 1)

        self.grid_2.addWidget(self.channels_dim_label, 1, 0, 1, 1)
        self.grid_2.addWidget(self.channels_dim_combo_box, 1, 1, 1, 1)

        self.grid_2.addWidget(labels_label, 2, 0, 1, 1)
        self.grid_2.addWidget(self.labels_selector, 2, 1, 1, 1)

        self.grid_2.addWidget(self.make_mask_button, 3, 1, 1, 1)
        self.grid_2.addWidget(mask_label, 4, 0, 1, 1)
        self.grid_2.addWidget(self.mask_selector, 4, 1, 1, 1)

        self._row_labels += [
            raw_label,
            self.channels_dim_label,
            labels_label,
            mask_label,
        ]

    def set_grid_3(self):
        """
        Adds the 2D task selector, the Load model, Train, Stop and Save
        buttons, and the 2D loss plot.
        """
        model_2d_type_label = QLabel(self)
        model_2d_type_label.setText("Task")
        model_2d_type_label.setToolTip(
            "Select the task (output) for the 2D model: 2D affs (affinity"
            " maps), 2D lsd (local shape descriptors), or 2D mtlsd"
            " (multi-task, both lsd and affs)"
        )
        self.model_2d_type_selector = QComboBox(self)
        self.model_2d_type_selector.setToolTip(
            "2D model task: affs (affinities),"
            " lsd (local shape descriptors),"
            " or mtlsd (multi-task, both)"
        )
        self.model_2d_type_selector.addItems(["2d_affs", "2d_lsd", "2d_mtlsd"])
        self.model_2d_type_selector.currentTextChanged.connect(
            self.update_3d_model_type_selector
        )

        self.advanced_2d_config_button = QPushButton("Advanced Config")
        self.advanced_2d_config_button.setToolTip(
            "Configure the training, network, and task parameters for the 2D"
            " model"
        )
        self.advanced_2d_config_button.clicked.connect(
            lambda: self.open_model_options("2d")
        )

        self.model_2d_config = deepcopy(DEFAULT_2D_MODEL_CONFIG)

        self.train_2d_model_from_scratch_checkbox = QCheckBox(self)
        self.train_2d_model_from_scratch_checkbox.setText("Train from scratch")
        self.train_2d_model_from_scratch_checkbox.setToolTip(
            "Check to train the 2D model from scratch, which resets the"
            " current model state and weights"
        )

        self.load_2d_model_button = QPushButton("Load model")
        self.load_2d_model_button.setToolTip(
            "Select a checkpoint file to load the 2D model from"
        )

        self.losses_2d_widget = pg.PlotWidget()
        self.losses_2d_widget.setFixedHeight(220)
        self.losses_2d_widget.hide()

        self.start_2d_training_button = QPushButton("Train")
        self.start_2d_training_button.setProperty("primary", True)
        self.start_2d_training_button.setToolTip("Start training the 2D model")
        self.stop_2d_training_button = QPushButton("Stop")
        self.stop_2d_training_button.setToolTip("Stop training the 2D model")
        self.save_2d_weights_button = QPushButton("Save")
        self.save_2d_weights_button.setToolTip(
            "Save the 2D model state to a checkpoint file"
        )
        self.stop_2d_training_button.setEnabled(False)
        self.save_2d_weights_button.setEnabled(False)

        train_2d_actions = make_action_grid(
            [
                [
                    (self.train_2d_model_from_scratch_checkbox, 2),
                    (self.load_2d_model_button, 1),
                ],
                [
                    (self.start_2d_training_button, 1),
                    (self.stop_2d_training_button, 1),
                    (self.save_2d_weights_button, 1),
                ],
            ]
        )

        self.train_2d_model_from_scratch_checkbox.stateChanged.connect(
            lambda: self.affect_load_weights("2d")
        )
        self.train_2d_model_from_scratch_checkbox.setChecked(True)

        self.load_2d_model_button.clicked.connect(
            lambda: self.load_weights("2d")
        )

        self.start_2d_training_button.clicked.connect(
            lambda: self.prepare_for_start_training("2d")
        )
        self.stop_2d_training_button.clicked.connect(
            lambda: self.prepare_for_stop_training("2d")
        )
        self.save_2d_weights_button.clicked.connect(
            lambda: self.save_weights("2d")
        )

        self.grid_3.addWidget(model_2d_type_label, 0, 0, 1, 1)
        self.grid_3.addWidget(self.model_2d_type_selector, 0, 1, 1, 1)
        self.grid_3.addLayout(train_2d_actions, 1, 0, 1, 2)
        self.grid_3.addWidget(self.losses_2d_widget, 2, 0, 1, 2)
        self._row_labels.append(model_2d_type_label)

    def set_grid_4(self):
        """
        Adds the 3D task selector, the Download and Load model buttons, and
        the Train the 3D model panel with its Train, Stop and Save buttons
        and the 3D loss plot.
        """

        model_3d_type_label = QLabel(self)
        model_3d_type_label.setText("Task")
        model_3d_type_label.setToolTip(
            "Select the 2D input for the 3D affinities model. The widget sets"
            " the input from the 2D model task unless that task is 2D mtlsd"
        )

        self.model_3d_type_selector = QComboBox(self)
        self.model_3d_type_selector.setToolTip(
            "The input type for the 3D model. The widget sets it"
            " from the 2D task unless the task is mtlsd"
        )
        self.model_3d_type_selector.addItems(
            [
                "3d_affs_from_2d_affs",
                "3d_affs_from_2d_lsd",
                "3d_affs_from_2d_mtlsd",
            ]
        )

        self.advanced_3d_config_button = QPushButton("Advanced Config")
        self.advanced_3d_config_button.setToolTip(
            "Configure the training, network, and task parameters for the 3D"
            " model"
        )
        self.advanced_3d_config_button.clicked.connect(
            lambda: self.open_model_options("3d")
        )

        self.model_3d_config = deepcopy(DEFAULT_3D_MODEL_CONFIG)

        self.train_3d_model_from_scratch_checkbox = QCheckBox(
            "Train from scratch"
        )
        self.train_3d_model_from_scratch_checkbox.setToolTip(
            "Check to train the 3D model from scratch, which resets the"
            " current model state and weights"
        )

        self.download_3d_model_button = QPushButton("Download")
        self.download_3d_model_button.setProperty("primary", True)
        self.download_3d_model_button.setToolTip(
            "Download the pretrained 3D model for the current 3D task"
        )
        self.load_3d_model_button = QPushButton("Load model")
        self.load_3d_model_button.setToolTip(
            "Select a checkpoint file to load the 3D model from"
        )

        model_3d_actions = make_action_grid(
            [
                [
                    (self.download_3d_model_button, 2),
                    (self.load_3d_model_button, 1),
                ]
            ]
        )

        self.losses_3d_widget = pg.PlotWidget()
        self.losses_3d_widget.setFixedHeight(220)
        self.losses_3d_widget.hide()

        self.start_3d_training_button = QPushButton("Train")
        self.start_3d_training_button.setToolTip("Start training the 3D model")
        self.stop_3d_training_button = QPushButton("Stop")
        self.stop_3d_training_button.setToolTip("Stop training the 3D model")
        self.save_3d_weights_button = QPushButton("Save")
        self.save_3d_weights_button.setToolTip(
            "Save the 3D model state to a checkpoint file"
        )
        self.start_3d_training_button.setEnabled(False)
        self.stop_3d_training_button.setEnabled(False)
        self.save_3d_weights_button.setEnabled(False)

        train_3d_actions = make_action_grid(
            [
                [
                    (self.start_3d_training_button, 1),
                    (self.stop_3d_training_button, 1),
                    (self.save_3d_weights_button, 1),
                ]
            ]
        )

        self.train_3d_toggle = QToolButton(self)
        self.train_3d_toggle.setText("Train the 3D model")
        self.train_3d_toggle.setCheckable(True)
        self.train_3d_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.train_3d_toggle.setArrowType(Qt.RightArrow)
        self.train_3d_toggle.setProperty("disclosure", True)
        self.train_3d_toggle.setToolTip(
            "Show the controls to train a 3D model from scratch or from a"
            " loaded checkpoint."
        )
        self.train_3d_toggle.toggled.connect(self._show_3d_training)

        self.train_3d_panel = QWidget(self)
        train_3d_layout = QVBoxLayout(self.train_3d_panel)
        train_3d_layout.setContentsMargins(0, 0, 0, 0)
        train_3d_layout.setSpacing(6)
        train_3d_layout.addWidget(self.train_3d_model_from_scratch_checkbox)
        train_3d_layout.addLayout(train_3d_actions)
        train_3d_layout.addWidget(self.losses_3d_widget)
        self.train_3d_panel.setVisible(False)

        self.grid_4.addWidget(model_3d_type_label, 0, 0, 1, 1)
        self.grid_4.addWidget(self.model_3d_type_selector, 0, 1, 1, 1)
        self.grid_4.addLayout(model_3d_actions, 1, 0, 1, 2)
        self.grid_4.addWidget(self.train_3d_toggle, 2, 0, 1, 2, Qt.AlignLeft)
        self.grid_4.addWidget(self.train_3d_panel, 3, 0, 1, 2)
        self._row_labels.append(model_3d_type_label)

        self.train_3d_model_from_scratch_checkbox.stateChanged.connect(
            lambda: self.affect_load_weights("3d")
        )
        self.train_3d_model_from_scratch_checkbox.setChecked(False)

        self.download_3d_model_button.clicked.connect(
            lambda: self.download_pretrained_model()
        )

        self.load_3d_model_button.clicked.connect(
            lambda: self.load_weights("3d")
        )

        self.start_3d_training_button.clicked.connect(
            lambda: self.prepare_for_start_training("3d")
        )
        self.stop_3d_training_button.clicked.connect(
            lambda: self.prepare_for_stop_training("3d")
        )
        self.save_3d_weights_button.clicked.connect(
            lambda: self.save_weights("3d")
        )

    def _show_3d_training(self, shown):
        """
        Shows or hides the Train the 3D model panel. The panel stays open
        while 3D training runs, so that Stop stays in reach.
        """
        if not shown and self.stop_3d_training_button.isEnabled():
            self.train_3d_toggle.setChecked(True)
            return
        self.train_3d_panel.setVisible(shown)
        self.train_3d_toggle.setArrowType(
            Qt.DownArrow if shown else Qt.RightArrow
        )

    def set_grid_5(self):
        """
        Adds the segmentation method selector and the Start and Stop buttons.
        """
        self.seg_method_label = QLabel("Method")
        self.seg_method_label.setToolTip(
            "Select the segmentation algorithm to run on the output 3D"
            " affinities"
        )
        self.seg_method_selector = QComboBox()
        self.seg_method_selector.setToolTip(
            "The segmentation algorithm for the output 3D affinities"
        )
        self.seg_method_selector.addItems(
            [
                "mutex watershed",
                "connected components",
            ]
        )
        self.seg_method_selector.setCurrentText("mutex watershed")

        self.advanced_seg_config_button = QPushButton("Advanced Config")
        self.advanced_seg_config_button.setToolTip(
            "Configure the parameters for the selected segmentation"
            " algorithm. Changes apply on the next run"
        )
        self.advanced_seg_config_button.clicked.connect(self.open_seg_options)

        self.start_inference_button = QPushButton("Start")
        self.start_inference_button.setProperty("primary", True)
        self.start_inference_button.setToolTip(
            "Run 2D model inference, then 3D model inference, then"
            " segmentation"
        )
        self.stop_inference_button = QPushButton("Stop")
        self.stop_inference_button.setToolTip(
            "Stop the current inference pipeline"
        )
        self.stop_inference_button.setEnabled(False)

        seg_actions = make_action_grid(
            [
                [
                    (self.start_inference_button, 2),
                    (self.stop_inference_button, 1),
                ]
            ]
        )

        self.grid_5.addWidget(self.seg_method_label, 0, 0, 1, 1)
        self.grid_5.addWidget(self.seg_method_selector, 0, 1, 1, 1)
        self.grid_5.addLayout(seg_actions, 1, 0, 1, 2)
        self._row_labels.append(self.seg_method_label)
        self.start_inference_button.clicked.connect(
            self.prepare_for_start_inference
        )
        self.stop_inference_button.clicked.connect(
            self.prepare_for_stop_inference
        )

        self.seg_params = deepcopy(DEFAULT_SEG_PARAMS)
        self.sync_seg_neighborhood()

    def set_grid_6(self):
        """
        Adds the feedback link. `_set_link_labels` sets its text.
        """
        self._feedback_label = QLabel(self)
        self._feedback_label.setOpenExternalLinks(True)
        self.grid_6.setContentsMargins(SECTION_INSET, 0, SECTION_INSET, 0)
        self.grid_6.addWidget(self._feedback_label, 0, 0, 1, 1)

    def prepare_for_start_training(self, dimension):
        self.start_2d_training_button.setEnabled(False)
        self.start_3d_training_button.setEnabled(False)
        self.save_2d_weights_button.setEnabled(False)
        self.save_3d_weights_button.setEnabled(False)

        self.advanced_2d_config_button.setEnabled(False)
        self.advanced_3d_config_button.setEnabled(False)

        self.device_combo_box.setEnabled(False)
        self.raw_selector.setEnabled(False)
        self.channels_dim_combo_box.setEnabled(False)
        self.labels_selector.setEnabled(False)
        self.make_mask_button.setEnabled(False)
        self.mask_selector.setEnabled(False)

        self.train_2d_model_from_scratch_checkbox.setEnabled(False)
        self.train_3d_model_from_scratch_checkbox.setEnabled(False)
        self.load_2d_model_button.setEnabled(False)
        self.load_3d_model_button.setEnabled(False)
        self.download_3d_model_button.setEnabled(False)

        self.model_2d_type_selector.setEnabled(False)
        self.model_3d_type_selector.setEnabled(False)

        self.stop_2d_training_button.setEnabled(dimension == "2d")
        self.stop_3d_training_button.setEnabled(dimension == "3d")

        self.seg_method_selector.setEnabled(False)
        self.advanced_seg_config_button.setEnabled(False)
        self.start_inference_button.setEnabled(False)
        self.stop_inference_button.setEnabled(False)

        getattr(self, f"losses_{dimension}_widget").show()

        worker = self.train(dimension)
        setattr(self, f"train_{dimension}_worker", worker)

        worker.yielded.connect(
            lambda data: self.on_yield_training(
                dimension=dimension, plot_data=data
            )
        )
        worker.returned.connect(
            lambda: self.prepare_for_stop_training(dimension)
        )

        worker.start()

    def remove_inference_attributes(self, dimension):
        """
        Removes existing predictions when training starts or the user loads a
        new model.
        """
        for attr in [f"affs_{dimension}", f"lsds_{dimension}"]:
            if hasattr(self, attr):
                delattr(self, attr)

        for attr in ["affs_3d", "segmentation"]:
            if hasattr(self, attr):
                delattr(self, attr)

    @thread_worker
    def train(self, dimension):
        """
        Trains the 2D or 3D model and yields the losses every 10 iterations.
        """

        self.remove_inference_attributes(dimension)

        if dimension == "2d":
            for layer in self.viewer.layers:
                if f"{layer}" == self.raw_selector.currentText():
                    raw_layer = layer
                    break
            for layer in self.viewer.layers:
                if f"{layer}" == self.labels_selector.currentText():
                    labels_layer = layer
                    break
            for layer in self.viewer.layers:
                if f"{layer}" == self.mask_selector.currentText():
                    mask_layer = layer
                    break

        model_type = getattr(
            self, f"model_{dimension}_type_selector"
        ).currentText()

        model_config = getattr(self, f"model_{dimension}_config")

        if "output_shape" not in model_config["net"]:
            self.compute_output_shape(dimension)

        if dimension == "2d":
            self.napari_dataset = Napari2DDataset(
                raw_layer,
                labels_layer,
                mask_layer,
                model_type,
                channels_dim=self.channels_dim,
                adj_slices=model_config["net"]["adj_slices"],
                input_shape=model_config["net"]["input_shape"],
                output_shape=model_config["net"]["output_shape"],
                **model_config["task"],
            )
        elif dimension == "3d":
            self.napari_dataset = Napari3DDataset(
                model_type,
                input_shape=model_config["net"]["input_shape"],
                output_shape=model_config["net"]["output_shape"],
                **model_config["task"],
            )

        train_dataloader = torch.utils.data.DataLoader(
            dataset=self.napari_dataset,
            batch_size=model_config["batch_size"],
            drop_last=True,
            num_workers=model_config["num_workers"],
            pin_memory=self.device_combo_box.currentText().startswith("cuda"),
            persistent_workers=model_config["num_workers"] > 0,
            collate_fn=stack_samples if sys.platform == "darwin" else None,
        )

        if dimension == "2d":
            model = get_2d_model(
                num_channels=self.napari_dataset.num_channels,
                model_type=model_type,
                **model_config["net"],
                **model_config["task"],
            )
        elif dimension == "3d":
            model = get_3d_model(
                num_channels=self.napari_dataset.num_channels,
                model_type=model_type,
                **model_config["net"],
                **model_config["task"],
            )
        self.device = torch.device(self.device_combo_box.currentText())
        setattr(self, f"model_{dimension}", model.to(self.device))
        criterion = get_loss(model_type).to(self.device)

        setattr(
            self,
            f"optimizer_{dimension}",
            torch.optim.Adam(
                getattr(self, f"model_{dimension}").parameters(),
                lr=model_config["learning_rate"],
            ),
        )

        if getattr(
            self, f"train_{dimension}_model_from_scratch_checkbox"
        ).isChecked():
            setattr(self, f"losses_{dimension}", [])
            setattr(self, f"iterations_{dimension}", [])
            setattr(self, f"start_iteration_{dimension}", 0)
            getattr(self, f"losses_{dimension}_widget").clear()
        else:
            if not hasattr(self, f"pretrained_{dimension}_model_checkpoint"):
                pass
            else:
                print(
                    f"Resuming training from {getattr(self, f'pretrained_{dimension}_model_checkpoint')}"
                )
                self._load_weights(
                    dimension,
                    getattr(self, f"pretrained_{dimension}_model_checkpoint"),
                    training=True,
                )

        for iteration, batch in tqdm(
            zip(
                range(
                    getattr(self, f"start_iteration_{dimension}"),
                    model_config["max_iterations"],
                ),
                train_dataloader,
                strict=False,
            ),
            initial=getattr(self, f"start_iteration_{dimension}"),
        ):
            batch = [torch.as_tensor(item) for item in batch]
            loss, outputs = train_iteration(
                batch,
                model=getattr(self, f"model_{dimension}"),
                criterion=criterion,
                optimizer=getattr(self, f"optimizer_{dimension}"),
                device=self.device,
                dimension=dimension,
                mixed_precision=model_config.get("mixed_precision", True),
            )
            if (
                model_config["save_snapshots_every"] > 0
                and iteration % model_config["save_snapshots_every"] == 0
            ):
                self.save_snapshot(
                    batch,
                    outputs,
                    self.tmp_dir / f"snapshots_{model_type}.zarr",
                    iteration,
                    dimension,
                )

            getattr(self, f"losses_{dimension}").append(loss)
            getattr(self, f"iterations_{dimension}").append(iteration)

            if iteration % 10 == 0:
                yield getattr(self, f"losses_{dimension}"), getattr(
                    self, f"iterations_{dimension}"
                )
        print(f"{dimension.upper()} training finished.")
        return

    def on_yield_training(self, dimension, plot_data):
        """
        Updates the loss plot every 10 training iterations.
        """
        getattr(self, f"losses_{dimension}_widget").plot(
            plot_data[1], plot_data[0], clear=True, pen=self._loss_pen
        )

    def prepare_for_stop_training(self, dimension):
        self.start_2d_training_button.setEnabled(True)
        self.stop_2d_training_button.setEnabled(False)
        self.save_2d_weights_button.setEnabled(True)
        self.start_3d_training_button.setEnabled(True)
        self.stop_3d_training_button.setEnabled(False)
        self.save_3d_weights_button.setEnabled(True)

        self.load_2d_model_button.setEnabled(True)
        self.load_3d_model_button.setEnabled(True)
        self.download_3d_model_button.setEnabled(True)

        self.train_2d_model_from_scratch_checkbox.setEnabled(True)
        self.train_3d_model_from_scratch_checkbox.setEnabled(True)

        self.device_combo_box.setEnabled(True)
        self.raw_selector.setEnabled(True)
        self.labels_selector.setEnabled(True)
        self.make_mask_button.setEnabled(True)
        self.mask_selector.setEnabled(True)
        self.show_hide_channels_dim()

        self.model_2d_type_selector.setEnabled(True)
        self.model_3d_type_selector.setEnabled(True)

        self.advanced_2d_config_button.setEnabled(True)
        self.advanced_3d_config_button.setEnabled(True)

        self.seg_method_selector.setEnabled(True)
        self.advanced_seg_config_button.setEnabled(True)
        self.start_inference_button.setEnabled(True)
        self.stop_inference_button.setEnabled(False)

        worker_attr = f"train_{dimension}_worker"
        if (
            hasattr(self, worker_attr)
            and getattr(self, worker_attr) is not None
        ):
            state = {
                "model_state_dict": getattr(
                    self, f"model_{dimension}"
                ).state_dict(),
                "optim_state_dict": getattr(
                    self, f"optimizer_{dimension}"
                ).state_dict(),
                "iterations": getattr(self, f"iterations_{dimension}"),
                "losses": getattr(self, f"losses_{dimension}"),
                "model_type": getattr(
                    self, f"model_{dimension}_type_selector"
                ).currentText(),
                "model_config": getattr(
                    self, f"model_{dimension}_config"
                ),
            }
            checkpoint_file_name = (
                self.tmp_dir
                / f"last_{getattr(self, f'model_{dimension}_type_selector').currentText()}.pth"
            )
            torch.save(state, checkpoint_file_name)

            setattr(
                self,
                f"pretrained_{dimension}_model_checkpoint",
                checkpoint_file_name,
            )
            setattr(
                self,
                f"last_loaded_{dimension}_model_checkpoint",
                checkpoint_file_name,
            )
            getattr(self, worker_attr).quit()
            setattr(self, worker_attr, None)

    def prepare_for_start_inference(self):
        self.advanced_2d_config_button.setEnabled(False)
        self.advanced_3d_config_button.setEnabled(False)

        self.device_combo_box.setEnabled(False)
        self.raw_selector.setEnabled(False)
        self.channels_dim_combo_box.setEnabled(False)
        self.labels_selector.setEnabled(False)
        self.make_mask_button.setEnabled(False)
        self.mask_selector.setEnabled(False)

        self.train_2d_model_from_scratch_checkbox.setEnabled(False)
        self.train_3d_model_from_scratch_checkbox.setEnabled(False)

        self.load_2d_model_button.setEnabled(False)
        self.load_3d_model_button.setEnabled(False)
        self.download_3d_model_button.setEnabled(False)

        self.model_2d_type_selector.setEnabled(False)
        self.model_3d_type_selector.setEnabled(False)

        self.seg_method_selector.setEnabled(False)
        self.advanced_seg_config_button.setEnabled(False)
        self.start_inference_button.setEnabled(False)
        self.stop_inference_button.setEnabled(True)

        self._inference_cancelled = False

        self.start_2d_training_button.setEnabled(False)
        self.stop_2d_training_button.setEnabled(False)
        self.save_2d_weights_button.setEnabled(False)
        self.start_3d_training_button.setEnabled(False)
        self.stop_3d_training_button.setEnabled(False)
        self.save_3d_weights_button.setEnabled(False)

        self.inference_worker = self.infer()
        self.inference_worker.returned.connect(self.on_return_infer)
        self.inference_worker.start()

    def prepare_for_stop_inference(self):
        self._inference_cancelled = True

        self.start_2d_training_button.setEnabled(True)
        self.stop_2d_training_button.setEnabled(False)
        self.save_2d_weights_button.setEnabled(True)
        self.start_3d_training_button.setEnabled(True)
        self.stop_3d_training_button.setEnabled(False)
        self.save_3d_weights_button.setEnabled(True)

        self.advanced_2d_config_button.setEnabled(True)
        self.advanced_3d_config_button.setEnabled(True)

        self.device_combo_box.setEnabled(True)
        self.raw_selector.setEnabled(True)
        self.labels_selector.setEnabled(True)
        self.make_mask_button.setEnabled(True)
        self.mask_selector.setEnabled(True)
        self.show_hide_channels_dim()

        self.train_2d_model_from_scratch_checkbox.setEnabled(True)
        self.train_3d_model_from_scratch_checkbox.setEnabled(True)

        self.load_2d_model_button.setEnabled(True)
        self.load_3d_model_button.setEnabled(True)
        self.download_3d_model_button.setEnabled(True)

        self.model_2d_type_selector.setEnabled(True)
        self.model_3d_type_selector.setEnabled(True)

        self.seg_method_selector.setEnabled(True)
        self.advanced_seg_config_button.setEnabled(True)
        self.start_inference_button.setEnabled(True)
        self.stop_inference_button.setEnabled(False)

        if self.inference_worker is not None:
            self.inference_worker.quit()
            self.inference_worker = None

    def _check_inference_cancelled(self):
        if self._inference_cancelled:
            raise InterruptedError("The user cancelled inference")

    def reload_model_if_needed(self, dimension, model_type, num_channels):
        """Loads the model when its type or checkpoint changes."""
        model_attr = f"model_{dimension}"
        model_config_attr = f"model_{dimension}_config"
        checkpoint_attr = f"pretrained_{dimension}_model_checkpoint"
        last_checkpoint_attr = f"last_loaded_{dimension}_model_checkpoint"
        model_needs_reload = False

        if hasattr(self, model_attr):
            model = getattr(self, model_attr)
            if (
                not hasattr(model, "model_type")
                or model.model_type != model_type
            ):
                model_needs_reload = True
                print(
                    f"{dimension.upper()} model type changed from {getattr(model, 'model_type', 'unknown')} to {model_type}"
                )
        else:
            model_needs_reload = True

        if hasattr(self, checkpoint_attr):
            current_checkpoint = getattr(self, checkpoint_attr)
            last_checkpoint = getattr(self, last_checkpoint_attr, None)
            if current_checkpoint != last_checkpoint:
                model_needs_reload = True
                print(
                    f"{dimension.upper()} model checkpoint changed from {last_checkpoint} to {current_checkpoint}"
                )

        if model_needs_reload and hasattr(self, checkpoint_attr):
            checkpoint_path = getattr(self, checkpoint_attr)

            checkpoint_state = torch.load(
                checkpoint_path, map_location=self.device
            )
            if "model_config" in checkpoint_state:
                model_config = checkpoint_state["model_config"]
                setattr(self, model_config_attr, model_config)
                print(
                    f"Using the {dimension.upper()} model config "
                    f"from the checkpoint."
                )
            else:
                model_config = getattr(self, model_config_attr)

            model_func = get_2d_model if dimension == "2d" else get_3d_model

            new_model = model_func(
                model_type=model_type,
                num_channels=num_channels,
                **model_config["net"],
                **model_config["task"],
            )

            setattr(self, model_attr, new_model.to(self.device))

            print(
                f"Loading the {dimension.upper()} model from {checkpoint_path}"
            )
            self._load_weights(dimension, checkpoint_path)

            setattr(self, last_checkpoint_attr, checkpoint_path)
        elif not hasattr(self, checkpoint_attr):
            raise ValueError(
                f"No checkpoint is loaded for the {dimension.upper()} model"
            )

    @thread_worker
    def infer(self):
        try:
            return self._infer_impl()
        except InterruptedError:
            print("Inference cancelled.")
            return None

    def _infer_impl(self):

        model_2d_type = self.model_2d_type_selector.currentText()
        model_3d_type = self.model_3d_type_selector.currentText()

        raw = gp.ArrayKey("RAW")
        int_lsds = gp.ArrayKey("LSDS_2D")
        int_affs = gp.ArrayKey("AFFS_2D")
        pred_affs = gp.ArrayKey("AFFS_3D")

        if "output_shape" not in self.model_2d_config["net"]:
            self.compute_output_shape("2d")
        if "output_shape" not in self.model_3d_config["net"]:
            self.compute_output_shape("3d")

        voxel_size = 1, 1, 1
        offset = 0, 0, 0
        input_shape_2d = [self.model_2d_config["net"]["adj_slices"], *[
            sum(x)
            for x in zip(
                self.model_2d_config["net"]["input_shape"],
                self.model_2d_config["net"]["shape_increase"],
                strict=False,
            )
        ]]
        output_shape_2d = [1, *[
            sum(x)
            for x in zip(
                self.model_2d_config["net"]["output_shape"],
                self.model_2d_config["net"]["shape_increase"],
                strict=False,
            )
        ]]
        input_shape_3d = [
            sum(x)
            for x in zip(
                self.model_3d_config["net"]["input_shape"],
                self.model_3d_config["net"]["shape_increase"],
                strict=False,
            )
        ]
        output_shape_3d = [
            sum(x)
            for x in zip(
                self.model_3d_config["net"]["output_shape"],
                self.model_3d_config["net"]["shape_increase"],
                strict=False,
            )
        ]

        outs_2d = []
        ins_3d = []

        if model_2d_type == "2d_lsd":
            outs_2d.append(int_lsds)
        elif model_2d_type == "2d_affs":
            outs_2d.append(int_affs)
        elif model_2d_type == "2d_mtlsd":
            outs_2d.append(int_lsds)
            outs_2d.append(int_affs)
        else:
            raise ValueError(f"Invalid 2D model type: {model_2d_type}")

        for layer in self.viewer.layers:
            if f"{layer}" == self.raw_selector.currentText():
                raw_image_layer = layer
                if len(raw_image_layer.data.shape) > 3:
                    num_channels_2d = (
                        raw_image_layer.data.shape[self.channels_dim]
                        * input_shape_2d[0]
                    )
                elif len(raw_image_layer.data.shape) == 3:
                    num_channels_2d = input_shape_2d[0]
                else:
                    raise ValueError(
                        f"The image layer must be 3D, but it has {len(raw_image_layer.data.shape)} dimensions"
                    )
                break

        if model_3d_type == "3d_affs_from_2d_lsd":
            ins_3d.append(int_lsds)
            num_channels_3d = 6
        elif model_3d_type == "3d_affs_from_2d_affs":
            ins_3d.append(int_affs)
            num_channels_3d = len(
                self.model_3d_config["task"]["in_aff_neighborhood"]
            )
        elif model_3d_type == "3d_affs_from_2d_mtlsd":
            ins_3d.append(int_lsds)
            ins_3d.append(int_affs)
            num_channels_3d = 6 + len(
                self.model_3d_config["task"]["in_aff_neighborhood"]
            )
        else:
            raise ValueError(f"Invalid 3D model type: {model_3d_type}")

        if not hasattr(self, "device"):
            self.device = torch.device(self.device_combo_box.currentText())

        self.reload_model_if_needed(
            dimension="2d",
            model_type=model_2d_type,
            num_channels=num_channels_2d,
        )

        self.reload_model_if_needed(
            dimension="3d",
            model_type=model_3d_type,
            num_channels=num_channels_3d,
        )

        self.model_2d.to(self.device)
        self.model_2d.eval()

        self.model_3d.to(self.device)
        self.model_3d.eval()

        if len(raw_image_layer.data.shape) == 4:
            total_shape = list(raw_image_layer.data.shape)
            total_shape.pop(self.channels_dim)
            roi = gp.Roi(offset, total_shape)
        else:
            roi = gp.Roi(offset, raw_image_layer.data.shape)

        scan_1 = gp.BatchRequest()
        scan_1.add(raw, input_shape_2d)
        for out in outs_2d:
            scan_1.add(out, output_shape_2d)

        scan_2 = gp.BatchRequest()
        for inp in ins_3d:
            scan_2.add(inp, input_shape_3d)
        scan_2.add(pred_affs, output_shape_3d)

        predict_2d = torchPredict(
            self.model_2d,
            inputs={0: raw},
            outputs=dict(enumerate(outs_2d)),
            array_specs={
                k: gp.ArraySpec(roi=roi, voxel_size=voxel_size)
                for k in outs_2d
            },
            device=self.device_combo_box.currentText(),
        )

        predict_3d = torchPredict(
            self.model_3d,
            inputs=dict(enumerate(ins_3d)),
            outputs={
                0: pred_affs,
            },
            array_specs={
                pred_affs: gp.ArraySpec(roi=roi, voxel_size=voxel_size)
            },
            device=self.device_combo_box.currentText(),
        )

        pipeline_1 = (
            NapariImageSource(
                image=raw_image_layer,
                key=raw,
                spec=gp.ArraySpec(
                    roi=roi,
                    voxel_size=voxel_size,
                    dtype=raw_image_layer.data.dtype,
                    interpolatable=True,
                ),
                channels_dim=self.channels_dim,
            )
            + gp.Normalize(raw)
            + gp.Pad(raw, None, "reflect")
            + gp.IntensityScaleShift(raw, 2, -1)
            + gp.Unsqueeze([raw])
            + predict_2d
            + gp.Squeeze(outs_2d)
            + gp.Scan(scan_1)
        )
        request_1 = gp.BatchRequest()
        for out in outs_2d:
            request_1.add(out, roi.shape)

        self._check_inference_cancelled()

        if hasattr(self, "affs_3d"):
            pass
        else:
            if all(hasattr(self, str(attr).lower()) for attr in ins_3d):
                pass
            else:
                assert all(
                    inp in outs_2d for inp in ins_3d
                ), f"All 3D inputs {ins_3d} must be in 2D outputs {outs_2d}"

                print(
                    f"Running 2D model inference on {raw_image_layer.name} to get {outs_2d}..."
                )
                with gp.build(pipeline_1):
                    batch = pipeline_1.request_batch(request_1)

                for out in outs_2d:
                    setattr(self, str(out).lower(), batch[out].data)

            self._check_inference_cancelled()

            pipeline_2 = (
                tuple(
                    NpArraySource(
                        getattr(self, str(inp).lower()),
                        spec=gp.ArraySpec(roi=roi, voxel_size=voxel_size),
                        key=inp,
                    )
                    for inp in ins_3d
                )
                + gp.MergeProvider()
            )

            for inp in ins_3d:
                pipeline_2 += gp.Normalize(inp, factor=1.0)
                pipeline_2 += gp.Pad(inp, None, "reflect")

            pipeline_2 += gp.Unsqueeze(ins_3d)
            pipeline_2 += predict_3d
            pipeline_2 += gp.Squeeze([pred_affs])
            pipeline_2 += gp.Scan(scan_2)

            request_2 = gp.BatchRequest()
            request_2.add(pred_affs, roi.shape)

            print(
                f"Running 3D model inference on {ins_3d} to get {pred_affs}..."
            )
            with gp.build(pipeline_2):
                batch = pipeline_2.request_batch(request_2)

            self.affs_3d = batch[pred_affs].data

        self._check_inference_cancelled()

        pred_layers = []

        for out in outs_2d:
            out_data = getattr(self, str(out).lower())
            raw_name = str(out).lower()
            parts = raw_name.split("_")
            out_name = (
                f"{parts[1]} {parts[0]}"
                if len(parts) == 2
                else raw_name.replace("_", " ")
            )

            pred_layers.append(
                (
                    out_data,
                    {"name": out_name},
                    "image",
                )
            )

        num_affs_3d = self.affs_3d.shape[0]
        group_size = 3
        for start in range(0, num_affs_3d, group_size):
            end = min(start + group_size, num_affs_3d)
            group_data = self.affs_3d[start:end]

            # Pad to 3 channels for RGB if needed
            pad_size = 3 - group_data.shape[0]
            if pad_size > 0:
                pad_shape = (pad_size, *group_data.shape[1:])
                group_data = np.concatenate(
                    [group_data, np.zeros(pad_shape)], axis=0
                )

            # Move channel dim to last for RGB: (3, D, H, W) -> (D, H, W, 3)
            rgb_data = np.ascontiguousarray(
                np.moveaxis(group_data, 0, -1)
            )

            group_name = f"3d affs [{start}:{end}]"
            pred_layers.append(
                (
                    rgb_data,
                    {
                        "name": group_name,
                        "rgb": True,
                    },
                    "image",
                )
            )

        self._check_inference_cancelled()

        method = self.seg_method_selector.currentText()
        seg_params = self.seg_params[method]
        selected = seg_params.get("selected_channels")

        if method == "mutex watershed":
            nbhd = seg_params["neighborhood"]

            bias = seg_params.get("bias", (-0.4, -0.7))
            if isinstance(bias, (list, tuple)) and len(bias) == 2:
                n = len(nbhd)
                per_ch_bias = [bias[0]] * min(3, n) + [bias[1]] * max(
                    0, n - 3
                )
            else:
                per_ch_bias = list(bias)

            use_affs = self.affs_3d
            use_nbhd = nbhd
            use_bias = per_ch_bias
            use_strides = seg_params.get("strides")

            if (
                selected is not None
                and len(selected) < self.affs_3d.shape[0]
            ):
                use_affs = self.affs_3d[selected]
                use_nbhd = [nbhd[i] for i in selected]
                use_bias = [per_ch_bias[i] for i in selected]
                if use_strides:
                    use_strides = [use_strides[i] for i in selected]

            clean_mws = {
                k: v
                for k, v in seg_params.items()
                if k not in {"selected_channels", "bias",
                             "neighborhood", "strides"}
            }
            clean_mws["neighborhood"] = use_nbhd
            clean_mws["bias"] = use_bias
            if use_strides:
                clean_mws["strides"] = use_strides
            clean_params = dict(self.seg_params)
            clean_params["mutex watershed"] = clean_mws

            self.segmentation = segment_affs(
                use_affs,
                method=method,
                params=clean_params,
            )
        elif method == "connected components":
            if selected is not None:
                use_affs = self.affs_3d[selected[:3]]
            else:
                use_affs = self.affs_3d[:3]

            clean_cc = {
                k: v
                for k, v in seg_params.items()
                if k not in {"selected_channels", "neighborhood"}
            }
            bias = clean_cc.get("bias", 0.0)
            if isinstance(bias, (int, float)):
                clean_cc["bias"] = [bias] * use_affs.shape[0]
            clean_params = dict(self.seg_params)
            clean_params["connected components"] = clean_cc

            self.segmentation = segment_affs(
                use_affs,
                method=method,
                params=clean_params,
            )
        else:
            self.segmentation = segment_affs(
                self.affs_3d,
                method=method,
                params=self.seg_params,
            )

        print("Inference complete.")

        return pred_layers + [
            (
                (
                    self.segmentation
                    if self.channels_dim is None
                    else np.expand_dims(self.segmentation, self.channels_dim)
                ),
                {"name": "segmentation"},
                "labels",
            )
        ]

    def on_return_infer(self, layers):
        if layers is None:
            self.prepare_for_stop_inference()
            return

        with self.viewer.layers.events.blocker():
            for data, metadata, layer_type in layers:
                if metadata["name"] in self.viewer.layers:
                    del self.viewer.layers[metadata["name"]]
                if layer_type == "image":
                    layer = self.viewer.add_image(data, **metadata)
                    layer.visible = False
                elif layer_type == "labels":
                    self.viewer.add_labels(
                        data.astype(int), **metadata
                    )

        self.inference_worker.quit()
        self.prepare_for_stop_inference()

    def make_mask(self):
        for layer in self.viewer.layers:
            if f"{layer}" == self.labels_selector.currentText():
                labels_layer = layer
                labels_name = f"{layer}"
                break

        mask = labels_layer.data > 0
        if f"mask_{labels_name}" in self.viewer.layers:
            del self.viewer.layers[f"mask_{labels_name}"]
        self.viewer.add_labels(
            mask.astype("uint8"), name=f"mask_{labels_name}"
        )
        print(
            f"Made a mask from {labels_name} in the new layer mask_{labels_name}."
        )

    def open_parameter_dialog(
        self, title, params, param_filter=None, readonly_params=None
    ):
        """
        Opens a dialog to edit parameters, such as a model config or the
        segmentation options. The parameters can be nested dictionaries.

        Parameters:
        -----------
        title : str
            The dialog window title.
        params : dict
            The parameters to edit. They can contain nested dictionaries.
        param_filter : callable, optional
            A function that takes a parameter name and returns True to
            include the parameter.
        readonly_params : set, optional
            Parameter names in dot notation that the dialog shows as
            read-only.

        Returns:
        --------
        bool:
            True if the user clicked OK, False if the user canceled.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle(title)

        layout = QVBoxLayout()
        layout.setSpacing(GROUP_SPACING)
        layout.setContentsMargins(8, 8, 8, 8)

        widgets = {}

        def flatten_dict(d, parent_key="", sep="."):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        flat_params = flatten_dict(params)
        param_items = list(flat_params.items())

        filtered_items = [
            (name, val)
            for name, val in param_items
            if not param_filter or param_filter(name)
        ]

        from collections import OrderedDict

        groups = OrderedDict()
        for param_name, param_value in filtered_items:
            if "." in param_name:
                prefix = param_name.split(".")[0]
            else:
                prefix = ""
            groups.setdefault(prefix, []).append(
                (param_name, param_value)
            )

        readonly_tooltips = {
            "net.output_shape": "Computed from the network settings.",
            "task.in_aff_neighborhood": (
                "Set from the 2D model's affinity neighborhood."
            ),
        }

        param_tooltips = {
            "mixed_precision": (
                "Runs the forward pass in bfloat16 on MPS and on CUDA GPUs"
                " with native bfloat16 support to speed up training. Weights"
                " stay in float32, and the CPU always uses float32."
            ),
        }

        def create_widget(param_name, param_value):
            is_readonly = (
                readonly_params and param_name in readonly_params
            )
            if isinstance(param_value, bool):
                widget = QCheckBox()
                widget.setChecked(param_value)
                if is_readonly:
                    widget.setEnabled(False)
                if param_name in param_tooltips:
                    widget.setToolTip(param_tooltips[param_name])
            else:
                widget = QLineEdit()
                widget.setMinimumWidth(240)
                widget.setText(str(param_value))
                widget.setCursorPosition(0)
                if is_readonly:
                    widget.setReadOnly(True)
                if isinstance(param_value, list):
                    widget.setToolTip(
                        f"Enter a {type(param_value).__name__}, for example"
                        f" {param_value}"
                    )
                if is_readonly and param_name in readonly_tooltips:
                    widget.setToolTip(readonly_tooltips[param_name])
            return widget

        field_labels = []
        group_grids = []
        training = None
        for prefix, items in groups.items():
            collapsible = make_collapsible(
                prefix.replace("_", " ").title() if prefix else "Training",
                self._theme,
            )
            group_widget = QWidget()
            group_grid = QGridLayout()
            group_grid.setContentsMargins(0, 2, 0, 0)
            group_grid.setSpacing(6)
            group_grid.setColumnStretch(1, 1)
            for row, (param_name, param_value) in enumerate(items):
                short_name = param_name.split(".", 1)[-1]
                label = QLabel(short_name.replace("_", " ").title())
                if param_name in param_tooltips:
                    label.setToolTip(param_tooltips[param_name])
                group_grid.addWidget(label, row, 0)
                field_labels.append(label)
                widget = create_widget(param_name, param_value)
                group_grid.addWidget(widget, row, 1)
                widgets[param_name] = widget
            group_widget.setLayout(group_grid)
            group_grids.append(group_grid)
            collapsible.addWidget(group_widget)
            layout.addWidget(collapsible)
            if prefix == "":
                training = collapsible

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )

        layout.addWidget(button_box)
        layout.setSizeConstraint(QVBoxLayout.SetMinAndMaxSize)
        dialog.setLayout(layout)
        if field_labels:
            align_label_columns(field_labels, group_grids)
        if training is not None:
            training.expand(animate=False)

        result = [False]

        def on_ok():
            updated_flat_params = {}
            for param_name, widget in widgets.items():
                if isinstance(widget, QCheckBox):
                    updated_flat_params[param_name] = widget.isChecked()
                else:
                    value = widget.text()
                    if value.lower() == "none":
                        updated_flat_params[param_name] = None
                    else:
                        try:
                            import ast

                            updated_flat_params[param_name] = ast.literal_eval(
                                value
                            )
                        except (SyntaxError, ValueError):
                            updated_flat_params[param_name] = value

            def update_nested_dict(nested_dict, flat_dict, sep="."):
                for key, value in flat_dict.items():
                    parts = key.split(sep)
                    d = nested_dict
                    for part in parts[:-1]:
                        if part not in d:
                            d[part] = {}
                        d = d[part]
                    d[parts[-1]] = value

            update_nested_dict(params, updated_flat_params)

            result[0] = True
            dialog.accept()

        button_box.accepted.connect(on_ok)
        button_box.rejected.connect(dialog.reject)

        dialog.exec_()
        return result[0]

    def sync_seg_neighborhood(self):
        """
        Sets the segmentation neighborhood from `aff_neighborhood` in the 3D
        model config.
        """
        aff_nbhd = self.model_3d_config["task"]["aff_neighborhood"]
        mws_params = self.seg_params["mutex watershed"]
        mws_params["neighborhood"] = [list(n) for n in aff_nbhd]

        n = len(aff_nbhd)

        if "bias" not in mws_params:
            mws_params["bias"] = (-0.4, -0.7)

        dims = len(aff_nbhd[0]) if aff_nbhd else 3
        strides = []
        for g in range(0, n, dims):
            group = aff_nbhd[g : g + dims]
            group_stride = [
                max(max(1, abs(off[d])) for off in group)
                for d in range(dims)
            ]
            for _ in group:
                strides.append(list(group_stride))
        mws_params["strides"] = strides

        mws_params["selected_channels"] = list(range(n))

        cc_params = self.seg_params["connected components"]
        cc_params["neighborhood"] = [list(nb) for nb in aff_nbhd]
        cc_params["selected_channels"] = list(range(min(3, n)))

    def compute_output_shape(self, dimension):
        """Computes `output_shape` by passing a dummy tensor through a UNet."""
        from napari_bootstrapper.models.unet import UNet

        config = getattr(self, f"model_{dimension}_config")
        net = config["net"]
        input_shape = net["input_shape"]

        unet = UNet(
            in_channels=1,
            num_fmaps=net["num_fmaps"],
            fmap_inc_factor=net["fmap_inc_factor"],
            downsample_factors=net["downsample_factors"],
            kernel_size_down=net.get("kernel_size_down"),
            kernel_size_up=net.get("kernel_size_up"),
            num_fmaps_out=net.get("num_fmaps_out"),
        )

        dummy = torch.zeros(1, 1, *input_shape)
        with torch.no_grad():
            out = unet(dummy)
        output_shape = list(out.shape[2:])

        config["net"]["output_shape"] = output_shape
        print(f"The {dimension.upper()} output_shape is {output_shape}")
        return output_shape

    def open_model_options(self, dimension):
        """
        Opens a dialog with the model configuration options.

        Parameters:
        -----------
        dimension : str
            Either "2d" or "3d".
        """
        model_type = getattr(
            self, f"model_{dimension}_type_selector"
        ).currentText()
        config = getattr(self, f"model_{dimension}_config")
        config.setdefault("mixed_precision", True)

        def param_filter(param_name):
            if "." not in param_name:
                return True

            if param_name.startswith("net."):
                return True

            if param_name.startswith("task."):
                task_param = param_name.split(".")[1]

                if "lsd" in task_param and "lsd" in model_type:
                    return True

                if "aff" in task_param and (
                    "affs" in model_type or "mtlsd" in model_type
                ):
                    return True

            return False

        net_before = {
            k: str(v)
            for k, v in config["net"].items()
            if k != "output_shape"
        }

        readonly = {"net.output_shape"}

        if dimension == "3d" and (
            "from_2d_affs" in model_type
            or "from_2d_mtlsd" in model_type
        ):
            config["task"]["in_aff_neighborhood"] = list(
                self.model_2d_config["task"]["aff_neighborhood"]
            )
            readonly.add("task.in_aff_neighborhood")

        success = self.open_parameter_dialog(
            title=f"Advanced {dimension.upper()} model configuration",
            params=config,
            param_filter=param_filter,
            readonly_params=readonly,
        )

        if success:
            net_after = {
                k: str(v)
                for k, v in config["net"].items()
                if k != "output_shape"
            }
            if net_before != net_after:
                self.compute_output_shape(dimension)
            if dimension == "3d":
                self.sync_seg_neighborhood()
            if dimension == "2d":
                m3type = self.model_3d_type_selector.currentText()
                if "from_2d_affs" in m3type or "from_2d_mtlsd" in m3type:
                    self.model_3d_config["task"][
                        "in_aff_neighborhood"
                    ] = list(config["task"]["aff_neighborhood"])
            print(f"Updated the {dimension} model configuration.")

    def open_seg_options(self):
        """
        Opens a dialog with advanced options for the selected segmentation
        method. The dialog includes a table for selecting affinity channels.
        """
        method = self.seg_method_selector.currentText()
        params = self.seg_params[method]
        is_mws = method == "mutex watershed"
        is_cc = method == "connected components"

        dialog = QDialog(self)
        dialog.setWindowTitle(f"{method.title()} Parameters")

        layout = QVBoxLayout()
        layout.setSpacing(GROUP_SPACING)
        layout.setContentsMargins(8, 8, 8, 8)

        channel_collapsible = make_collapsible(
            "Affinity channels", self._theme
        )
        channel_collapsible.setToolTip(
            "Select which affinity channels to use for segmentation"
        )
        channel_widget = QWidget()
        channel_grid = QGridLayout()
        channel_grid.setContentsMargins(0, 2, 0, 0)
        channel_grid.setSpacing(6)

        neighborhood = params.get("neighborhood", [])
        n = len(neighborhood)
        selected = params.get(
            "selected_channels", list(range(min(3, n) if is_cc else n))
        )
        strides = params.get("strides", [])
        bias = params.get("bias", (-0.4, -0.7) if is_mws else 0.0)

        if is_mws:
            if isinstance(bias, (list, tuple)) and len(bias) == 2:
                per_ch_bias = [bias[0]] * min(3, n) + [bias[1]] * max(
                    0, n - 3
                )
            else:
                per_ch_bias = (
                    list(bias)
                    if isinstance(bias, (list, tuple))
                    else [bias] * n
                )
        else:
            bias_val = float(bias) if not isinstance(bias, (list, tuple)) else 0.0
            per_ch_bias = [bias_val] * n

        headers = ["Offset", "Bias"]
        if is_mws:
            headers.append("Stride")
        headers.append("Include")
        # A spare column takes the extra width, so a wide note does not
        # push the Offset column away from Bias.
        channel_grid.setColumnStretch(len(headers), 1)
        header_row = 0
        if is_cc:
            channel_grid.addWidget(
                QLabel("Connected components uses at most 3 channels."),
                0,
                0,
                1,
                len(headers) + 1,
            )
            header_row = 1
        for col, header in enumerate(headers):
            lbl = QLabel(f"<b>{header}</b>")
            channel_grid.addWidget(lbl, header_row, col)

        include_col = len(headers) - 1

        channel_checkboxes = []
        bias_widgets = []
        stride_widgets = []
        row_labels = []
        for i, offset in enumerate(neighborhood):
            row = header_row + i + 1
            col = 0

            offset_label = QLabel(str(offset))
            channel_grid.addWidget(offset_label, row, col)
            col += 1

            bias_edit = QLineEdit()
            bv = per_ch_bias[i] if i < len(per_ch_bias) else 0.0
            bias_edit.setText(str(bv))
            bias_edit.setCursorPosition(0)
            bias_edit.setMaximumWidth(60)
            channel_grid.addWidget(bias_edit, row, col)
            bias_widgets.append(bias_edit)
            col += 1

            if is_mws:
                stride_edit = QLineEdit()
                stride_val = (
                    strides[i]
                    if i < len(strides)
                    else [1] * len(offset)
                )
                stride_edit.setText(str(stride_val))
                stride_edit.setCursorPosition(0)
                stride_edit.setMaximumWidth(120)
                channel_grid.addWidget(stride_edit, row, col)
                stride_widgets.append(stride_edit)

            cb = QCheckBox()
            cb.setChecked(i in selected)
            channel_grid.addWidget(cb, row, include_col)
            channel_checkboxes.append(cb)

            row_labels.append(offset_label)

        def update_row_style(idx, checked):
            row_labels[idx].setEnabled(checked)
            bias_widgets[idx].setEnabled(checked)
            if is_mws and idx < len(stride_widgets):
                stride_widgets[idx].setEnabled(checked)

        def on_cc_toggle(checked, idx):
            """Allows at most 3 selected channels for connected components."""
            update_row_style(idx, checked)
            if checked:
                num_checked = sum(
                    cb.isChecked() for cb in channel_checkboxes
                )
                if num_checked > 3:
                    channel_checkboxes[idx].setChecked(False)

        for i, cb in enumerate(channel_checkboxes):
            update_row_style(i, cb.isChecked())
            if is_cc:
                cb.toggled.connect(
                    lambda checked, idx=i: on_cc_toggle(checked, idx)
                )
            else:
                cb.toggled.connect(
                    lambda checked, idx=i: update_row_style(
                        idx, checked
                    )
                )

        channel_widget.setLayout(channel_grid)
        channel_collapsible.addWidget(channel_widget)
        channel_collapsible.expand()
        layout.addWidget(channel_collapsible)

        seg_collapsible = make_collapsible(
            "Segmentation parameters", self._theme
        )
        seg_collapsible.setToolTip(
            f"Configure the {method} algorithm parameters"
        )
        seg_widget = QWidget()
        seg_grid = QGridLayout()
        seg_grid.setContentsMargins(0, 2, 0, 0)
        seg_grid.setSpacing(6)

        exclude_keys = {
            "neighborhood",
            "selected_channels",
            "strides",
            "bias",
        }

        param_widgets = {}
        row = 0
        for key, value in params.items():
            if key in exclude_keys:
                continue
            label_text = key.replace("_", " ").title()
            seg_grid.addWidget(QLabel(label_text), row, 0)
            if isinstance(value, bool):
                w = QCheckBox()
                w.setChecked(value)
            else:
                w = QLineEdit()
                w.setText(str(value))
                w.setCursorPosition(0)
            seg_grid.addWidget(w, row, 1)
            param_widgets[key] = w
            row += 1

        seg_widget.setLayout(seg_grid)
        seg_collapsible.addWidget(seg_widget)
        layout.addWidget(seg_collapsible)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        layout.addWidget(button_box)
        layout.setSizeConstraint(QVBoxLayout.SetMinAndMaxSize)
        dialog.setLayout(layout)

        result = [False]

        def on_ok():
            import ast

            params["selected_channels"] = [
                i
                for i, cb in enumerate(channel_checkboxes)
                if cb.isChecked()
            ]

            if is_mws:
                new_bias = []
                for bw in bias_widgets:
                    try:
                        new_bias.append(float(bw.text()))
                    except ValueError:
                        new_bias.append(-0.5)

                if len(new_bias) >= 3:
                    short = set(new_bias[:3])
                    long = (
                        set(new_bias[3:])
                        if len(new_bias) > 3
                        else set()
                    )
                    if len(short) == 1 and len(long) <= 1:
                        long_val = (
                            long.pop() if long else short.pop()
                        )
                        params["bias"] = (new_bias[0], long_val)
                    else:
                        params["bias"] = new_bias
                else:
                    params["bias"] = new_bias

                new_strides = []
                for sw in stride_widgets:
                    try:
                        new_strides.append(
                            ast.literal_eval(sw.text())
                        )
                    except (SyntaxError, ValueError):
                        new_strides.append(sw.text())
                params["strides"] = new_strides
            else:
                sel = params["selected_channels"]
                if sel:
                    try:
                        params["bias"] = float(
                            bias_widgets[sel[0]].text()
                        )
                    except ValueError:
                        params["bias"] = 0.0

            for key, widget in param_widgets.items():
                if isinstance(widget, QCheckBox):
                    params[key] = widget.isChecked()
                else:
                    value = widget.text()
                    if value.lower() == "none":
                        params[key] = None
                    else:
                        try:
                            params[key] = ast.literal_eval(value)
                        except (SyntaxError, ValueError):
                            params[key] = value

            result[0] = True
            dialog.accept()

        button_box.accepted.connect(on_ok)
        button_box.rejected.connect(dialog.reject)
        dialog.exec_()

        if result[0]:
            print(f"Updated the segmentation parameters for {method}.")

    def load_weights(self, dimension):
        """
        Handles a click on the 2D or 3D Load model button.
        """
        file_name, _ = QFileDialog.getOpenFileName(
            caption=f"Load {dimension.upper()} model weights"
        )

        if not file_name:
            return

        setattr(self, f"pretrained_{dimension}_model_checkpoint", file_name)
        self.remove_inference_attributes(dimension)

        state = torch.load(file_name, map_location="cpu")
        if isinstance(state, dict):
            if "model_config" in state:
                setattr(
                    self,
                    f"model_{dimension}_config",
                    state["model_config"],
                )
                print(
                    f"Loaded the {dimension.upper()} model config "
                    f"from the checkpoint."
                )
            if "model_type" in state:
                selector = getattr(
                    self, f"model_{dimension}_type_selector"
                )
                current_type = selector.currentText()
                checkpoint_type = state["model_type"]
                if checkpoint_type != current_type:
                    idx = selector.findText(checkpoint_type)
                    if idx >= 0:
                        selector.setCurrentIndex(idx)
                        print(
                            f"Set the {dimension.upper()} model type "
                            f"to '{checkpoint_type}' from the checkpoint."
                        )
                    else:
                        print(
                            f"WARNING: The checkpoint model type "
                            f"'{checkpoint_type}' is not a "
                            f"selector option."
                        )

        if dimension == "3d":
            self.sync_seg_neighborhood()

        getattr(self, f"start_{dimension}_training_button").setEnabled(True)
        getattr(self, f"save_{dimension}_weights_button").setEnabled(True)

        print(
            f"The widget will load the {dimension.upper()} model weights from {getattr(self, f'pretrained_{dimension}_model_checkpoint')}"
        )

    def download_pretrained_model(self):
        """
        Downloads the pretrained 3D model weights from the Bootstrapper GitHub
        releases.
        """
        model_type = self.model_3d_type_selector.currentText()

        if model_type not in PRETRAINED_3D_MODEL_URLS:
            raise ValueError(f"Unknown model: {model_type}")

        url = PRETRAINED_3D_MODEL_URLS[model_type]
        file_path = self.tmp_dir / f"checkpoints_{model_type}.zip"
        out_dir = self.tmp_dir / f"checkpoints_{model_type}"
        member = f"{model_type}/model_checkpoint_25000.ckpt"
        model_checkpoint = out_dir / member
        expected = PRETRAINED_3D_MODEL_SHA256[model_type]
        verified = False
        if model_checkpoint.is_file():
            with model_checkpoint.open("rb") as file:
                verified = (
                    hashlib.file_digest(file, "sha256").hexdigest()
                    == expected["checkpoint"]
                )

        if not verified:
            print(
                f"Downloading the {model_type} checkpoint archive to {file_path}..."
            )
            digest = hashlib.sha256()
            try:
                with requests.get(url, stream=True, timeout=60) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get("content-length", 0))
                    with (
                        open(file_path, "wb") as file,
                        tqdm(
                            desc=f"{model_type} checkpoints",
                            total=total_size,
                            unit="iB",
                            unit_scale=True,
                            unit_divisor=1024,
                        ) as progress_bar,
                    ):
                        for data in response.iter_content(chunk_size=1 << 20):
                            digest.update(data)
                            progress_bar.update(file.write(data))

                if digest.hexdigest() != expected["archive"]:
                    raise ValueError(
                        f"SHA-256 mismatch for {model_type} archive"
                    )
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extract(member, out_dir)
                with model_checkpoint.open("rb") as file:
                    checksum = hashlib.file_digest(file, "sha256").hexdigest()
                if checksum != expected["checkpoint"]:
                    model_checkpoint.unlink()
                    raise ValueError(
                        f"SHA-256 mismatch for {model_type} checkpoint"
                    )
            finally:
                file_path.unlink(missing_ok=True)

        self.remove_inference_attributes("3d")
        self.pretrained_3d_model_checkpoint = model_checkpoint
        print(
            f"The widget will load the 3D model weights from {self.pretrained_3d_model_checkpoint}"
        )

    def _load_weights(self, dimension, model_checkpoint, training=False):
        """
        Reads model weights from a file and loads them into the model.
        """

        if not hasattr(self, "device"):
            self.device = torch.device(self.device_combo_box.currentText())

        state = torch.load(model_checkpoint, map_location=self.device)

        if "model_state_dict" in state:
            weights = state["model_state_dict"]
        elif "state_dict" in state:
            # External checkpoint format
            weights = state["state_dict"]
            weights = {
                k.removeprefix("model."): v
                for k, v in weights.items()
            }
        else:
            weights = {
                k.removeprefix("model."): v
                for k, v in state.items()
            }

        if "model_config" in state:
            current_config = getattr(
                self, f"model_{dimension}_config"
            )
            checkpoint_config = state["model_config"]

            if (
                checkpoint_config["net"]
                != current_config["net"]
            ):
                print(
                    f"WARNING: The {dimension.upper()} model config "
                    f"in the checkpoint differs from the current config. "
                    f"Using the checkpoint config to match the weights."
                )
                setattr(
                    self,
                    f"model_{dimension}_config",
                    checkpoint_config,
                )

                model_type = state.get(
                    "model_type",
                    getattr(
                        self,
                        f"model_{dimension}_type_selector",
                    ).currentText(),
                )
                model_func = (
                    get_2d_model
                    if dimension == "2d"
                    else get_3d_model
                )
                first_weight = next(
                    v
                    for k, v in weights.items()
                    if "weight" in k and v.ndim >= 3
                )
                num_channels = first_weight.shape[1]

                model = model_func(
                    model_type=model_type,
                    num_channels=num_channels,
                    **checkpoint_config["net"],
                    **checkpoint_config["task"],
                )
                setattr(
                    self,
                    f"model_{dimension}",
                    model.to(self.device),
                )

                if training and hasattr(
                    self, f"optimizer_{dimension}"
                ):
                    lr = checkpoint_config.get(
                        "learning_rate",
                        current_config.get(
                            "learning_rate", 1e-4
                        ),
                    )
                    setattr(
                        self,
                        f"optimizer_{dimension}",
                        torch.optim.Adam(
                            model.parameters(), lr=lr
                        ),
                    )
            else:
                print(
                    f"The {dimension.upper()} model config in the "
                    f"checkpoint matches the current config."
                )

        if "model_type" in state:
            current_type = getattr(
                self, f"model_{dimension}_type_selector"
            ).currentText()
            if state["model_type"] != current_type:
                print(
                    f"WARNING: The checkpoint model type "
                    f"'{state['model_type']}' differs from the "
                    f"current selection '{current_type}'."
                )

        model = getattr(self, f"model_{dimension}")
        model.load_state_dict(weights, strict=True)

        if training:
            if "optim_state_dict" in state:
                getattr(self, f"optimizer_{dimension}").load_state_dict(
                    state["optim_state_dict"]
                )

            if all(k in state for k in ["iterations", "losses"]):
                setattr(
                    self, f"iterations_{dimension}", state["iterations"]
                )
                setattr(self, f"losses_{dimension}", state["losses"])
                setattr(
                    self,
                    f"start_iteration_{dimension}",
                    state["iterations"][-1] + 1,
                )
                getattr(self, f"losses_{dimension}_widget").clear()
                getattr(self, f"losses_{dimension}_widget").plot(
                    state["iterations"], state["losses"], pen=self._loss_pen
                )
                getattr(self, f"losses_{dimension}_widget").show()
            else:
                setattr(self, f"losses_{dimension}", [])
                setattr(self, f"iterations_{dimension}", [])
                setattr(self, f"start_iteration_{dimension}", 0)
                getattr(self, f"losses_{dimension}_widget").clear()

        setattr(self, f"model_{dimension}", model.to(self.device))
        print(
            f"Loaded the {dimension.upper()} model weights from {model_checkpoint}"
        )

    def affect_load_weights(self, dimension):
        checkbox = getattr(
            self, f"train_{dimension}_model_from_scratch_checkbox"
        )
        load_button = getattr(self, f"load_{dimension}_model_button")
        download_button = getattr(
            self, f"download_{dimension}_model_button", None
        )

        start_button = getattr(self, f"start_{dimension}_training_button")
        save_button = getattr(self, f"save_{dimension}_weights_button")

        if checkbox.isChecked():
            load_button.setEnabled(False)
            if download_button is not None:
                download_button.setEnabled(False)

            start_button.setEnabled(True)
            save_button.setEnabled(False)
        else:
            load_button.setEnabled(True)
            if download_button is not None:
                download_button.setEnabled(True)

            has_checkpoint = hasattr(
                self, f"pretrained_{dimension}_model_checkpoint"
            )
            start_button.setEnabled(has_checkpoint)
            save_button.setEnabled(has_checkpoint)

    def save_weights(self, dimension):
        """
        Handles a click on the Save button.
        """
        checkpoint_file_name, _ = QFileDialog.getSaveFileName(
            caption=f"Save {dimension.upper()} model weights"
        )
        if not checkpoint_file_name:
            return
        if (
            hasattr(self, f"model_{dimension}")
            and hasattr(self, f"optimizer_{dimension}")
            and hasattr(self, f"iterations_{dimension}")
            and hasattr(self, f"losses_{dimension}")
        ):
            state = {
                "model_state_dict": getattr(
                    self, f"model_{dimension}"
                ).state_dict(),
                "optim_state_dict": getattr(
                    self, f"optimizer_{dimension}"
                ).state_dict(),
                "iterations": getattr(self, f"iterations_{dimension}"),
                "losses": getattr(self, f"losses_{dimension}"),
                "model_type": getattr(
                    self, f"model_{dimension}_type_selector"
                ).currentText(),
                "model_config": getattr(
                    self, f"model_{dimension}_config"
                ),
            }
            torch.save(state, checkpoint_file_name)
            print(
                f"Saved the {dimension.upper()} model weights to {checkpoint_file_name}"
            )

    def set_scroll_area(self):
        """
        Creates a scroll area that appears when the user resizes the main
        napari window.
        """
        self.scroll.setWidget(self.widget)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)

        self.setMinimumWidth(420)
        self.setCentralWidget(self.scroll)

    def save_snapshot(self, batch, outputs, path, iteration, dimension):
        import zarr

        print(f"Saving a snapshot to {path}/{iteration}")
        f = zarr.open(path, mode="a")

        is_2d = dimension == "2d"
        offset = (8, 46, 46) if not is_2d else (0, 46, 46)

        model_type = getattr(self, f"model_{dimension}").model_type
        num_inputs = 2 if model_type == "3d_affs_from_2d_mtlsd" else 1

        def process_and_save_array(arr, iteration, name, idx, is_input=False):
            array = arr.detach().cpu().numpy()
            shape = array.shape
            arr_name = f"{iteration}/{name}_{idx}"
            print(f"{arr_name}: {shape}, is_2d: {is_2d}")

            if is_2d:
                if len(shape) == 4:
                    array = array.swapaxes(0, 1)
                elif len(shape) == 5:
                    array = array.swapaxes(0, 2)
                    array = array[0]
            else:
                if len(shape) == 5:
                    array = array.swapaxes(0, 2)
                    array = array[0]

            f[arr_name] = array
            f[arr_name].attrs["offset"] = (
                (0, 0, 0) if (is_input and idx == 0) else offset
            )
            f[arr_name].attrs["voxel_size"] = (1, 1, 1)
            f[arr_name].attrs["axis_names"] = ["c^", "z", "y", "x"]

        for i, arr in enumerate(batch):
            process_and_save_array(
                arr, iteration, "arr", i, is_input=i < num_inputs
            )

        for i, arr in enumerate(outputs):
            process_and_save_array(
                arr, iteration, "outputs", i, is_input=False
            )
