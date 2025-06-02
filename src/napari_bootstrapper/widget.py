import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import re
import zipfile
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
    QVBoxLayout,
    QWidget,
)
from tqdm import tqdm

from .datasets.napari_2d_dataset import Napari2DDataset
from .datasets.napari_3d_dataset import Napari3DDataset
from .gp.napari_image_source import NapariImageSource
from .gp.np_source import NpArraySource
from .gp.torch_predict import torchPredict
from .models import (
    DEFAULT_2D_MODEL_CONFIG,
    DEFAULT_3D_MODEL_CONFIG,
    PRETRAINED_3D_MODEL_URLS,
    get_2d_model,
    get_3d_model,
    get_loss,
)
from .post import DEFAULT_SEG_PARAMS, segment_affs


def train_iteration(batch, model, criterion, optimizer, device, dimension):
    # if model is 2d_lsd
    batch = [item.to(device) for item in batch]

    model.train()
    if dimension == "3d" and len(batch) > 3:
        outputs = model(batch[0], batch[1])
    else:
        outputs = model(batch[0])

    if dimension == "2d":
        if len(outputs) == 1:
            # For 2d_lsd: batch = [raw, gt_lsds, lsds_weights]
            # For 2d_affs: batch = [raw, gt_affs, affs_weights]
            loss = criterion(outputs[0], batch[1], batch[2])
        else:  # Multiple outputs (2d_mtlsd)
            # For 2d_mtlsd: batch = [raw, gt_lsds, lsds_weights, gt_affs, affs_weights]
            loss = criterion(
                outputs[0],
                batch[1],
                batch[2],  # lsds part
                outputs[1],
                batch[3],
                batch[4],  # affs part
            )
    else:
        if len(batch) == 3:
            loss = criterion(outputs, batch[1], batch[2])
        elif len(batch) == 4:  # multiple inputs
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
        # initialize outer layout
        layout = QVBoxLayout()

        self.tmp_dir = Path.home() / ".cache" / "napari_bootstrapper"
        self.tmp_dir.mkdir(exist_ok=True)

        # initialize individual grid layouts from top to bottom
        self.grid_0 = QGridLayout()  # title
        self.set_grid_0()
        self.grid_1 = QGridLayout()  # device
        self.set_grid_1()
        self.grid_2 = QGridLayout()  # 2d model config
        self.set_grid_2()
        self.grid_3 = QGridLayout()  # 2d model training
        self.set_grid_3()
        self.grid_4 = QGridLayout()  # 3d model config and training
        self.set_grid_4()
        self.grid_5 = QGridLayout()  # inference
        self.set_grid_5()
        self.grid_6 = QGridLayout()  # feedback
        self.set_grid_6()

        layout.addLayout(self.grid_0)
        layout.addLayout(self.grid_1)
        layout.addLayout(self.grid_2)
        layout.addLayout(self.grid_3)
        layout.addLayout(self.grid_4)
        layout.addLayout(self.grid_5)
        layout.addLayout(self.grid_6)
        self.widget.setLayout(layout)
        self.set_scroll_area()
        self.viewer.layers.events.inserted.connect(self.update_selectors)
        self.viewer.layers.events.removed.connect(self.update_selectors)

    def update_selectors(self, event):
        """
        Whenever a new image is added or removed by the user,
        this function is called.
        It updates the `raw_selector` and `labels_selector` attributes.

        """
        count = 0
        for i in range(self.raw_selector.count() - 1, -1, -1):
            if self.raw_selector.itemText(i) == f"{event.value}":
                # remove item
                self.raw_selector.removeItem(i)
                count = 1
        if count == 0:
            self.raw_selector.addItems([f"{event.value}"])

        count = 0
        for i in range(self.labels_selector.count() - 1, -1, -1):
            if self.labels_selector.itemText(i) == f"{event.value}":
                # remove item
                self.labels_selector.removeItem(i)
                count = 1
        if count == 0:
            self.labels_selector.addItems([f"{event.value}"])

        count = 0
        for i in range(self.mask_selector.count() - 1, -1, -1):
            if self.mask_selector.itemText(i) == f"{event.value}":
                # remove item
                self.mask_selector.removeItem(i)
                count = 1
        if count == 0:
            self.mask_selector.addItems([f"{event.value}"])

    def show_hide_channels_dim(self):
        if self.raw_selector.currentText() == "":
            return

        raw_layer = self.viewer.layers[self.raw_selector.currentText()]
        raw_data = raw_layer.data

        if len(raw_data.shape) == 3:
            # No channel dimension for 3D data (d,h,w)
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
                f"Images must be 3D or 4D. Current image shape: {raw_data.shape}"
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
        Specifies the title of the plugin.
        """
        text_label = QLabel("<h3>Bootstrapper</h3>")
        method_description_label = QLabel(
            '<small>2D->3D method for dense 3D segmentations from sparse 2D labels.<br>If you are using this in your research, please <a href="https://github.com/ucsdmanorlab/bootstrapper#citation" style="color:gray;">cite us</a>.</small><br><small><tt><a href="https://github.com/ucsdmanorlab/bootstrapper" style="color:gray;">https://github.com/ucsdmanorlab/bootstrapper</a></tt></small>'
        )
        self.grid_0.addWidget(text_label, 0, 0, 1, 1)
        self.grid_0.addWidget(method_description_label, 1, 0, 2, 1)

    def set_grid_1(self):
        """
        Specifies the device used for training and inference.
        """

        device_label = QLabel(self)
        device_label.setText("Device")
        device_label.setToolTip("Select the device for training and inference")
        self.device_combo_box = QComboBox(self)
        self.device_combo_box.addItem("cpu")
        if torch.cuda.is_available():
            self.device_combo_box.addItem("cuda:0")
            self.device_combo_box.setCurrentText("cuda:0")
        if torch.backends.mps.is_available():
            self.device_combo_box.addItem("mps")
            self.device_combo_box.setCurrentText("mps")
        self.grid_1.addWidget(device_label, 0, 0, 1, 1)
        self.grid_1.addWidget(self.device_combo_box, 0, 1, 1, 1)

    def set_grid_2(self):
        """
        2D model and training config.
        """
        data_heading = self.set_section_heading("Data")
        data_heading.setToolTip(
            "Step 1: Specify the 3D image to train on and segment, its training labels and training mask"
        )

        model_2d_heading = self.set_section_heading("2D Model")
        model_2d_heading.setToolTip(
            "Step 2: Configure and train the 2D model. Optionally, load a pretrained model."
        )

        raw_label = QLabel(self)
        raw_label.setText("Image layer")
        raw_label.setToolTip("Select the image layer to train on")
        self.raw_selector = QComboBox(self)
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                self.raw_selector.addItem(f"{layer}")

        self.channels_dim = None
        self.channels_dim_label = QLabel(self)
        self.channels_dim_label.setText("Channels dimension")
        self.channels_dim_label.setToolTip(
            "Specify the channels dimension of the image data since the image layer has 4 dimensions (for example, DHWC -> 3, CDHW -> 0)"
        )
        self.channels_dim_combo_box = QComboBox(self)
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
        for layer in self.viewer.layers:
            if isinstance(layer, Labels):
                self.labels_selector.addItem(f"{layer}")

        self.make_mask_button = QPushButton(self)
        self.make_mask_button.setText("Make mask")
        self.make_mask_button.setToolTip(
            "Make a binary mask from the labels data (1 where labeled, 0 elsewhere)"
        )
        self.make_mask_button.clicked.connect(self.make_mask)

        mask_label = QLabel(self)
        mask_label.setText("Mask layer")
        mask_label.setToolTip(
            "Select the mask layer to use for training (loss is computed only where mask is 1)"
        )
        self.mask_selector = QComboBox(self)
        for layer in self.viewer.layers:
            if isinstance(layer, Labels):
                self.mask_selector.addItem(f"{layer}")

        model_2d_type_label = QLabel(self)
        model_2d_type_label.setText("Task")
        model_2d_type_label.setToolTip(
            "Select the task (output) for 2D model: 2D affs (affinity maps), 2D lsd (local shape descriptors), or 2D mtlsd (multi-task: lsd and affs)"
        )
        self.model_2d_type_selector = QComboBox(self)
        self.model_2d_type_selector.addItems(["2d_affs", "2d_lsd", "2d_mtlsd"])
        self.model_2d_type_selector.currentTextChanged.connect(
            self.update_3d_model_type_selector
        )

        self.advanced_2d_config_button = QPushButton("Advanced Config")
        self.advanced_2d_config_button.setToolTip(
            "Configure training, network, and task parameters for 2D model"
        )
        self.advanced_2d_config_button.clicked.connect(
            lambda: self.open_model_options("2d")
        )

        self.model_2d_config = DEFAULT_2D_MODEL_CONFIG

        self.grid_2.addWidget(data_heading, 0, 0, 1, 2)

        self.grid_2.addWidget(raw_label, 1, 0, 1, 1)
        self.grid_2.addWidget(self.raw_selector, 1, 1, 1, 1)

        self.grid_2.addWidget(self.channels_dim_label, 2, 0, 1, 1)
        self.grid_2.addWidget(self.channels_dim_combo_box, 2, 1, 1, 1)

        self.grid_2.addWidget(labels_label, 3, 0, 1, 1)
        self.grid_2.addWidget(self.labels_selector, 3, 1, 1, 1)

        self.grid_2.addWidget(self.make_mask_button, 4, 0, 1, 2)
        self.grid_2.addWidget(mask_label, 5, 0, 1, 1)
        self.grid_2.addWidget(self.mask_selector, 5, 1, 1, 1)

        self.grid_2.addWidget(model_2d_heading, 6, 0, 1, 2)

        self.grid_2.addWidget(model_2d_type_label, 7, 0, 1, 1)
        self.grid_2.addWidget(self.model_2d_type_selector, 7, 1, 1, 1)
        self.grid_2.addWidget(self.advanced_2d_config_button, 8, 0, 1, 2)

    def set_grid_3(self):
        """
        2D model load, plot, and train/stop/save buttons.
        """

        self.train_2d_model_from_scratch_checkbox = QCheckBox(self)
        self.train_2d_model_from_scratch_checkbox.setText("Train from scratch")
        self.train_2d_model_from_scratch_checkbox.setToolTip(
            "Check to train 2D model from scratch (will reset current model state and weights)"
        )

        self.load_2d_model_button = QPushButton("Load model")
        self.load_2d_model_button.setToolTip(
            "Select checkpoint file to load 2D model state or weights from"
        )

        self.losses_2d_widget = pg.PlotWidget()
        self.losses_2d_widget.setBackground((37, 41, 49))
        self.losses_2d_widget.setMinimumHeight(200)
        self.losses_2d_widget.hide()
        styles = {"color": "white", "font-size": "12px"}
        self.losses_2d_widget.setLabel("left", "Loss 2D", **styles)
        self.losses_2d_widget.setLabel("bottom", "Iterations", **styles)

        button_layout = QVBoxLayout()
        button_row = QGridLayout()

        self.start_2d_training_button = QPushButton("Train")
        self.start_2d_training_button.setToolTip("Start training 2D model")
        self.stop_2d_training_button = QPushButton("Stop")
        self.stop_2d_training_button.setToolTip("Stop training 2D model")
        self.save_2d_weights_button = QPushButton("Save")
        self.save_2d_weights_button.setToolTip(
            "Save 2D model state to checkpoint file"
        )
        self.stop_2d_training_button.setEnabled(False)
        self.save_2d_weights_button.setEnabled(False)

        button_row.addWidget(self.start_2d_training_button, 0, 0, 1, 1)
        button_row.addWidget(self.stop_2d_training_button, 0, 1, 1, 1)
        button_row.addWidget(self.save_2d_weights_button, 0, 2, 1, 1)
        button_row.setColumnStretch(0, 1)
        button_row.setColumnStretch(1, 1)
        button_row.setColumnStretch(2, 1)
        button_layout.addLayout(button_row)

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

        self.grid_3.addWidget(
            self.train_2d_model_from_scratch_checkbox, 0, 0, 1, 2
        )
        self.grid_3.addWidget(self.load_2d_model_button, 0, 2, 1, 2)
        self.grid_3.addWidget(self.losses_2d_widget, 1, 0, 4, 4)
        self.grid_3.addLayout(button_layout, 5, 0, 1, 4)

        self.grid_3.setColumnStretch(0, 1)
        self.grid_3.setColumnStretch(1, 1)
        self.grid_3.setColumnStretch(2, 1)
        self.grid_3.setColumnStretch(3, 1)

    def set_grid_4(self):
        """
        Specifies the 3D model configuration, loss plot and train/stop button.
        """

        model_3d_heading = self.set_section_heading("3D Model")
        model_3d_heading.setToolTip(
            "Step 3: Configure and load the 3D affinities model. Optionally, train from scratch."
        )

        model_3d_type_label = QLabel(self)
        model_3d_type_label.setText("Task")
        model_3d_type_label.setToolTip(
            "Select the 2D input for 3D affinities model (only relevant when 2D model task is 2D mtlsd, otherwise automatically inferred)"
        )

        self.model_3d_type_selector = QComboBox(self)
        self.model_3d_type_selector.addItems(
            [
                "3d_affs_from_2d_affs",
                "3d_affs_from_2d_lsd",
                "3d_affs_from_2d_mtlsd",
            ]
        )

        self.advanced_3d_config_button = QPushButton("Advanced Config")
        self.advanced_3d_config_button.setToolTip(
            "Configure training, network, and task parameters for 3D model"
        )
        self.advanced_3d_config_button.clicked.connect(
            lambda: self.open_model_options("3d")
        )

        self.model_3d_config = DEFAULT_3D_MODEL_CONFIG

        self.train_3d_model_from_scratch_checkbox = QCheckBox(
            "Train from scratch"
        )
        self.train_3d_model_from_scratch_checkbox.setToolTip(
            "Check to train 3D model from scratch (will reset current model state and weights)"
        )

        model_buttons_layout = QGridLayout()
        self.download_3d_model_button = QPushButton("Download")
        self.download_3d_model_button.setToolTip(
            "Download and load the latest pretrained weights for current 3D model type"
        )
        self.load_3d_model_button = QPushButton("Load")
        self.load_3d_model_button.setToolTip(
            "Select checkpoint file to load 3D model weights or state from"
        )

        model_buttons_layout.addWidget(
            self.download_3d_model_button, 0, 0, 1, 1
        )
        model_buttons_layout.addWidget(self.load_3d_model_button, 0, 1, 1, 1)
        model_buttons_layout.setColumnStretch(0, 1)
        model_buttons_layout.setColumnStretch(1, 1)

        self.losses_3d_widget = pg.PlotWidget()
        self.losses_3d_widget.setBackground((37, 41, 49))
        self.losses_3d_widget.setMinimumHeight(200)
        self.losses_3d_widget.hide()
        styles = {"color": "white", "font-size": "12px"}
        self.losses_3d_widget.setLabel("left", "Loss 3D", **styles)
        self.losses_3d_widget.setLabel("bottom", "Iterations", **styles)

        button_layout = QVBoxLayout()
        button_row = QGridLayout()

        self.start_3d_training_button = QPushButton("Train")
        self.start_3d_training_button.setToolTip("Start training 3D model")
        self.stop_3d_training_button = QPushButton("Stop")
        self.stop_3d_training_button.setToolTip("Stop training 3D model")
        self.save_3d_weights_button = QPushButton("Save")
        self.save_3d_weights_button.setToolTip(
            "Save 3D state to checkpoint file"
        )
        self.start_3d_training_button.setEnabled(False)
        self.stop_3d_training_button.setEnabled(False)
        self.save_3d_weights_button.setEnabled(False)

        button_row.addWidget(self.start_3d_training_button, 0, 0, 1, 1)
        button_row.addWidget(self.stop_3d_training_button, 0, 1, 1, 1)
        button_row.addWidget(self.save_3d_weights_button, 0, 2, 1, 1)
        button_row.setColumnStretch(0, 1)
        button_row.setColumnStretch(1, 1)
        button_row.setColumnStretch(2, 1)
        button_layout.addLayout(button_row)

        self.grid_4.addWidget(model_3d_heading, 0, 0, 1, 4)

        self.grid_4.addWidget(model_3d_type_label, 1, 0, 1, 2)
        self.grid_4.addWidget(self.model_3d_type_selector, 1, 2, 1, 2)
        self.grid_4.addWidget(self.advanced_3d_config_button, 2, 0, 1, 4)
        self.grid_4.addWidget(
            self.train_3d_model_from_scratch_checkbox, 3, 0, 1, 1
        )
        self.grid_4.addLayout(model_buttons_layout, 3, 1, 1, 3)
        self.grid_4.addWidget(self.losses_3d_widget, 4, 0, 4, 4)
        self.grid_4.addLayout(button_layout, 9, 0, 1, 4)

        self.grid_4.setColumnStretch(0, 1)
        self.grid_4.setColumnStretch(1, 1)
        self.grid_4.setColumnStretch(2, 1)
        self.grid_4.setColumnStretch(3, 1)

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

    def set_grid_5(self):

        seg_heading = self.set_section_heading("Segmentation")
        seg_heading.setToolTip("Step 4: Run the 2D → 3D pipeline.")

        self.seg_method_label = QLabel("Segmentation method:")
        self.seg_method_label.setToolTip(
            "Select the segmentation algorithm to use on output 3D affinities"
        )
        self.seg_method_selector = QComboBox()
        self.seg_method_selector.addItems(
            [
                # "watershed",
                "mutex watershed",
                "connected components",
            ]
        )
        self.seg_method_selector.setCurrentText("mutex watershed")

        self.advanced_seg_config_button = QPushButton(
            "Advanced Segmentation Parameters"
        )
        self.advanced_seg_config_button.setToolTip(
            "Configure parameters for the selected segmentation algorithm (changes apply on next run)"
        )
        self.advanced_seg_config_button.clicked.connect(self.open_seg_options)

        self.start_inference_button = QPushButton("Start")
        self.start_inference_button.setToolTip(
            "Run the complete pipeline: 2D model inference → 3D model inference → segmentation"
        )
        self.stop_inference_button = QPushButton("Stop")
        self.stop_inference_button.setEnabled(False)

        self.grid_5.addWidget(seg_heading, 0, 0, 1, 2)

        self.grid_5.addWidget(self.seg_method_label, 1, 0, 1, 1)
        self.grid_5.addWidget(self.seg_method_selector, 1, 1, 1, 1)
        self.grid_5.addWidget(self.advanced_seg_config_button, 2, 0, 1, 2)
        self.grid_5.addWidget(self.start_inference_button, 3, 0, 1, 1)
        self.grid_5.addWidget(self.stop_inference_button, 3, 1, 1, 1)
        self.start_inference_button.clicked.connect(
            self.prepare_for_start_inference
        )
        self.stop_inference_button.clicked.connect(
            self.prepare_for_stop_inference
        )

        self.seg_params = DEFAULT_SEG_PARAMS

    def set_grid_6(self):
        """
        Specifies the feedback URL.
        """

        feedback_label = QLabel(
            '<small>Please share any feedback <a href="https://github.com/ucsdmanorlab/napari-bootstrapper/issues/new/choose" style="color:gray;">here</a>.</small>'
        )
        self.grid_6.addWidget(feedback_label, 0, 0, 2, 1)

    def prepare_for_start_training(self, dimension):
        self.start_2d_training_button.setEnabled(False)
        self.start_3d_training_button.setEnabled(False)
        self.save_2d_weights_button.setEnabled(False)
        self.save_3d_weights_button.setEnabled(False)

        self.advanced_2d_config_button.setEnabled(False)
        self.advanced_3d_config_button.setEnabled(False)

        self.device_combo_box.setEnabled(False)
        self.raw_selector.setEnabled(False)
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
        When training is initiated or new model is loaded, then existing predictions are removed.
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
        Main train.
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

        # Create torch dataset
        if dimension == "2d":
            self.napari_dataset = Napari2DDataset(
                raw_layer,
                labels_layer,
                mask_layer,
                model_type,
                channels_dim=self.channels_dim,
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

        # Create dataloader
        train_dataloader = torch.utils.data.DataLoader(
            dataset=self.napari_dataset,
            batch_size=model_config["batch_size"],
            drop_last=True,
            num_workers=model_config["num_workers"],
            pin_memory=True,
            persistent_workers=True,
        )

        # Load model and loss
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

        # Set optimizer
        setattr(
            self,
            f"optimizer_{dimension}",
            torch.optim.Adam(
                getattr(self, f"model_{dimension}").parameters(),
                lr=model_config["learning_rate"],
                # weight_decay=0.01,
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
                    f"Resuming model from {getattr(self, f'pretrained_{dimension}_model_checkpoint')}"
                )
                self._load_weights(
                    dimension,
                    getattr(self, f"pretrained_{dimension}_model_checkpoint"),
                    training=True,
                )

        # Call Train Iteration
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
            loss, outputs = train_iteration(
                batch,
                model=getattr(self, f"model_{dimension}"),
                criterion=criterion,
                optimizer=getattr(self, f"optimizer_{dimension}"),
                device=self.device,
                dimension=dimension,
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
        print(f"{dimension.upper()} Training finished!")
        return

    def on_yield_training(self, dimension, plot_data):
        """
        The loss plot is updated every training N iterations.
        """
        getattr(self, f"losses_{dimension}_widget").plot(
            plot_data[1], plot_data[0]
        )

    def prepare_for_stop_training(self, dimension):
        # Re-enable all training buttons and save buttons
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

        self.model_2d_type_selector.setEnabled(True)
        self.model_3d_type_selector.setEnabled(True)

        self.advanced_2d_config_button.setEnabled(True)
        self.advanced_3d_config_button.setEnabled(True)

        # Re-enable all other UI elements
        self.seg_method_selector.setEnabled(True)
        self.advanced_seg_config_button.setEnabled(True)
        self.start_inference_button.setEnabled(True)
        self.stop_inference_button.setEnabled(False)

        # Handle the worker cleanup
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
            getattr(self, worker_attr).quit()
            setattr(self, worker_attr, None)

    def prepare_for_start_inference(self):
        self.advanced_2d_config_button.setEnabled(False)
        self.advanced_3d_config_button.setEnabled(False)

        self.device_combo_box.setEnabled(False)
        self.raw_selector.setEnabled(False)
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

    def reload_model_if_needed(self, dimension, model_type, num_channels):
        """Helper function to load a model if needed (new type or new checkpoint)"""
        model_attr = f"model_{dimension}"
        model_config_attr = f"model_{dimension}_config"
        checkpoint_attr = f"pretrained_{dimension}_model_checkpoint"
        last_checkpoint_attr = f"last_loaded_{dimension}_model_checkpoint"
        model_needs_reload = False

        # Check if model exists but model type has changed
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

        # Check if checkpoint has changed
        if hasattr(self, checkpoint_attr) and hasattr(
            self, last_checkpoint_attr
        ):
            current_checkpoint = getattr(self, checkpoint_attr)
            last_checkpoint = getattr(self, last_checkpoint_attr)
            if current_checkpoint != last_checkpoint:
                model_needs_reload = True
                print(
                    f"{dimension.upper()} model checkpoint changed from {last_checkpoint} to {current_checkpoint}"
                )

        # Load model if needed
        if model_needs_reload and hasattr(self, checkpoint_attr):
            checkpoint_path = getattr(self, checkpoint_attr)
            model_config = getattr(self, model_config_attr)

            # Get the right model constructor function
            model_func = get_2d_model if dimension == "2d" else get_3d_model

            # Create the model
            new_model = model_func(
                model_type=model_type,
                num_channels=num_channels,
                **model_config["net"],
                **model_config["task"],
            )

            # Store the model
            setattr(self, model_attr, new_model.to(self.device))

            print(f"Loading {dimension.upper()} model from {checkpoint_path}")
            self._load_weights(dimension, checkpoint_path)

            # Remember this checkpoint
            setattr(self, last_checkpoint_attr, checkpoint_path)
        elif not hasattr(self, checkpoint_attr):
            raise ValueError(
                f"No checkpoint loaded for {dimension.upper()} model"
            )

    @thread_worker
    def infer(self):

        model_2d_type = self.model_2d_type_selector.currentText()
        model_3d_type = self.model_3d_type_selector.currentText()

        raw = gp.ArrayKey("RAW")
        int_lsds = gp.ArrayKey("LSDS_2D")
        int_affs = gp.ArrayKey("AFFS_2D")
        pred_affs = gp.ArrayKey("AFFS_3D")

        voxel_size = 1, 1, 1
        offset = 0, 0, 0
        input_shape_2d = [
            sum(x)
            for x in zip(
                self.model_2d_config["net"]["input_shape"],
                self.model_2d_config["net"]["shape_increase"],
                strict=False,
            )
        ]
        output_shape_2d = [
            sum(x)
            for x in zip(
                self.model_2d_config["net"]["output_shape"],
                self.model_2d_config["net"]["shape_increase"],
                strict=False,
            )
        ]
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
                        f"Image layer must be 3D: {len(raw_image_layer.data.shape)}"
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

        # load model, set in eval mode
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

        # set up pipeline
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

        # do not predict if latest already available
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

        # create napari layers for returning
        pred_layers = []
        colormaps = ["red", "green", "blue"]

        # Add individual layers for each 2D output
        for out in outs_2d:
            out_data = getattr(self, str(out).lower())
            out_name = str(out).lower().replace("_", " ")
            num_channels = out_data.shape[0]

            if "affs" in out_name:
                channel_names = [
                    f"{out_name} {nb}" for nb in self.model_2d.aff_nbhd
                ]
            elif "lsds" in out_name:
                # For LSDs: offset(y), offset(x), covar(yy), covar(xx), corr(yx), size
                channel_names = [
                    f"{out_name} offset (y)",
                    f"{out_name} offset (x)",
                    f"{out_name} covar (yy)",
                    f"{out_name} covar (xx)",
                    f"{out_name} corr (yx)",
                    f"{out_name} size",
                ]
            else:
                channel_names = [
                    f"{out_name} ({i})" for i in range(out_data.shape[0])
                ]

            for i in range(num_channels):
                component_name = channel_names[i]
                component_color = colormaps[i % 3]

                pred_layers.append(
                    (
                        (
                            out_data[i : i + 1].copy()
                            if self.channels_dim is None
                            else np.moveaxis(
                                out_data[i : i + 1].copy(),
                                0,
                                self.channels_dim,
                            )
                        ),
                        {
                            "name": component_name,
                            "colormap": component_color,
                            "blending": "additive",
                        },
                        "image",
                    )
                )

        # For 3D affs
        affs_3d_names = [f"3d affs {nb}" for nb in self.model_3d.aff_nbhd]

        num_affs_3d = 6
        for i in range(num_affs_3d):
            component_name = affs_3d_names[i]
            component_color = colormaps[i % 3]

            pred_layers.append(
                (
                    (
                        self.affs_3d[i : i + 1].copy()
                        if self.channels_dim is None
                        else np.moveaxis(
                            self.affs_3d[i : i + 1].copy(),
                            0,
                            self.channels_dim,
                        )
                    ),
                    {
                        "name": component_name,
                        "colormap": component_color,
                        "blending": "additive",
                    },
                    "image",
                )
            )

        # segment affs
        self.segmentation = segment_affs(
            self.affs_3d,
            method=self.seg_method_selector.currentText(),
            params=self.seg_params,
        )

        print("Inference complete!")

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
        for data, metadata, layer_type in layers:
            if metadata["name"] in self.viewer.layers:
                del self.viewer.layers[metadata["name"]]
            if layer_type == "image":
                self.viewer.add_image(data, **metadata)
            elif layer_type == "labels":
                self.viewer.add_labels(data.astype(int), **metadata)

            if metadata["name"] != "segmentation":
                self.viewer.layers[metadata["name"]].visible = False

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
            f"Made mask for {labels_name} and made new layer: mask_{labels_name}!"
        )

    def open_parameter_dialog(self, title, params, param_filter=None):
        """
        Generic parameter dialog function that can be used for model configs or segmentation options.
        Supports nested dictionaries.

        Parameters:
        -----------
        title : str
            Dialog window title
        params : dict
            Dictionary of parameters to edit (can contain nested dictionaries)
        param_filter : callable, optional
            Function that takes a parameter name and returns True if it should be included

        Returns:
        --------
        bool:
            True if user clicked OK, False if canceled
        """
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setMinimumWidth(400)

        # Create layout
        layout = QVBoxLayout()
        form_layout = QGridLayout()

        # Create form fields for each parameter
        widgets = {}
        row = 0

        # Function to flatten nested dictionaries with path notation (e.g., "net.num_fmaps")
        def flatten_dict(d, parent_key="", sep="."):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        # Flatten the nested dictionary for display
        flat_params = flatten_dict(params)
        param_items = list(flat_params.items())

        for param_name, param_value in param_items:
            # Skip if filtered out
            if param_filter and not param_filter(param_name):
                continue

            # Create label with proper formatting (replace dots with spaces)
            label_text = (
                param_name.replace(".", " → ").replace("_", " ").title()
            )
            label = QLabel(label_text)
            form_layout.addWidget(label, row, 0)

            # Create appropriate widget based on parameter type
            if isinstance(param_value, bool):
                widget = QCheckBox()
                widget.setChecked(param_value)
            else:
                widget = QLineEdit()
                widget.setText(str(param_value))

                # Add tooltips for complex types
                if isinstance(param_value, list):
                    widget.setToolTip(
                        f"Format: {type(param_value).__name__}, e.g. {param_value}"
                    )

            form_layout.addWidget(widget, row, 1)
            widgets[param_name] = widget
            row += 1

        # Add buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )

        layout.addLayout(form_layout)
        layout.addWidget(button_box)
        dialog.setLayout(layout)

        # Track result
        result = [False]

        # Connect button actions
        def on_ok():
            # Update flattened parameters
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
                            # Try to parse as Python literal (list, float, etc.)
                            import ast

                            updated_flat_params[param_name] = ast.literal_eval(
                                value
                            )
                        except (SyntaxError, ValueError):
                            # If parsing fails, keep as string
                            updated_flat_params[param_name] = value

            # Function to update nested dict from flattened representation
            def update_nested_dict(nested_dict, flat_dict, sep="."):
                for key, value in flat_dict.items():
                    parts = key.split(sep)
                    d = nested_dict
                    for part in parts[:-1]:
                        if part not in d:
                            d[part] = {}
                        d = d[part]
                    d[parts[-1]] = value

            # Update the original nested dictionary
            update_nested_dict(params, updated_flat_params)

            result[0] = True
            dialog.accept()

        button_box.accepted.connect(on_ok)
        button_box.rejected.connect(dialog.reject)

        # Show dialog modally
        dialog.exec_()
        return result[0]

    def open_model_options(self, dimension):
        """
        Opens a dialog with model configuration options.

        Parameters:
        -----------
        dimension : str
            Either "2d" or "3d"
        """
        model_type = getattr(
            self, f"model_{dimension}_type_selector"
        ).currentText()
        config = getattr(self, f"model_{dimension}_config")

        # Define filter for nested parameters - now paths like "net.num_fmaps"
        def param_filter(param_name):
            # Always include top-level parameters
            if "." not in param_name:
                return True

            # For nested parameters, check based on model type
            # Network parameters are always included
            if param_name.startswith("net."):
                task_param = param_name.split(".")[1]
                if "shape" not in task_param:
                    return True

            # Task-specific parameters
            if param_name.startswith("task."):
                task_param = param_name.split(".")[1]

                # LSD parameters
                if "lsd" in task_param and "lsd" in model_type:
                    return True

                # Affinity parameters
                if "aff" in task_param and (
                    "affs" in model_type or "mtlsd" in model_type
                ):
                    return True

            return False

        success = self.open_parameter_dialog(
            title=f"Advanced {dimension.upper()} Model Configuration",
            params=config,
            param_filter=param_filter,
        )

        if success:
            print(f"Updated {dimension} model configuration:")

    def open_seg_options(self):
        """
        Opens a dialog with advanced options for the selected segmentation method.
        """
        method = self.seg_method_selector.currentText()

        # Get the parameters dictionary for the selected method
        params = self.seg_params[method]

        success = self.open_parameter_dialog(
            title=f"{method.title()} Parameters", params=params
        )

        if success:
            print(f"Updated segmentation parameters for {method}")

    def load_weights(self, dimension):
        """
        Describes sequence of actions after `Load Weights` button is pressed
        """
        file_name, _ = QFileDialog.getOpenFileName(
            caption=f"Load {dimension.upper()} Model Weights"
        )

        setattr(self, f"pretrained_{dimension}_model_checkpoint", file_name)
        self.remove_inference_attributes(dimension)

        getattr(self, f"start_{dimension}_training_button").setEnabled(True)
        getattr(self, f"save_{dimension}_weights_button").setEnabled(True)

        print(
            f" {dimension.upper()} Model weights will be loaded from {getattr(self, f'pretrained_{dimension}_model_checkpoint')}"
        )

    def download_pretrained_model(self):
        """
        Downloads pretrained model weights from Bootstrapper github.
        """
        model_type = self.model_3d_type_selector.currentText()

        if model_type not in PRETRAINED_3D_MODEL_URLS:
            raise ValueError(f"Unknown model: {model_type}")

        url = PRETRAINED_3D_MODEL_URLS[model_type]
        file_path = self.tmp_dir / f"checkpoints_{model_type}.zip"
        out_dir = self.tmp_dir / f"checkpoints_{model_type}"
        os.makedirs(out_dir, exist_ok=True)

        # if out_dir exists and contains checkpoints, skip download
        if not (out_dir.exists() and any(out_dir.iterdir())):
            print(
                f"Downloading {model_type} checkpoints zip to {file_path}..."
            )
            response = requests.get(url, stream=True)
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
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    progress_bar.update(size)

            # unzip checkpoints
            print(f"Unzipping {model_type} checkpoints in {self.tmp_dir}...")
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(out_dir)

            # clean up
            os.remove(file_path)

        # get largest checkpoint
        checkpoints = sorted(
            out_dir.glob("model_checkpoint_*"),
            key=lambda x: [
                int(c) if c.isdigit() else c.lower()
                for c in re.split("([0-9]+)", str(x))
            ],
        )
        if len(checkpoints) == 0:
            raise FileNotFoundError(f"No checkpoints found in {out_dir}")
        model_checkpoint = checkpoints[-1]

        # set pretrained checkpoint path
        self.remove_inference_attributes("3d")
        self.pretrained_3d_model_checkpoint = model_checkpoint
        print(
            f"3D Model weights will be loaded from {self.pretrained_3d_model_checkpoint}"
        )

    def _load_weights(self, dimension, model_checkpoint, training=False):
        """
        Gets model weights from a file and loads them into the model.
        """

        # Load the model architecture
        if not hasattr(self, "device"):
            self.device = torch.device(self.device_combo_box.currentText())

        # Load the state dict
        state = torch.load(model_checkpoint, map_location=self.device)

        model = getattr(self, f"model_{dimension}")

        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=True)

            if training:
                if "optim_state_dict" in state:
                    getattr(self, f"optimizer_{dimension}").load_state_dict(
                        state["optim_state_dict"]
                    )

                # If training state is available, restore it
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
                        state["iterations"], state["losses"]
                    )
                    getattr(self, f"losses_{dimension}_widget").show()
                else:
                    setattr(self, f"losses_{dimension}", [])
                    setattr(self, f"iterations_{dimension}", [])
                    setattr(self, f"start_iteration_{dimension}", 0)
                    getattr(self, f"losses_{dimension}_widget").clear()

        else:
            model.load_state_dict(state, strict=True)

        # Set the model attribute
        setattr(self, f"model_{dimension}", model.to(self.device))
        print(
            f"Successfully loaded {dimension.upper()} model weights from {model_checkpoint}"
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

            # Enable training from scratch
            start_button.setEnabled(True)
            save_button.setEnabled(False)  # Can only save after training
        else:
            load_button.setEnabled(True)
            if download_button is not None:
                download_button.setEnabled(True)

            # Only enable start/save if a checkpoint is loaded
            has_checkpoint = hasattr(
                self, f"pretrained_{dimension}_model_checkpoint"
            )
            start_button.setEnabled(has_checkpoint)
            save_button.setEnabled(has_checkpoint)

    def save_weights(self, dimension):
        """
        Describes sequence of actions which ensue, after `Save weights` button is pressed

        """
        checkpoint_file_name, _ = QFileDialog.getSaveFileName(
            caption=f"Save {dimension.upper()} Model Weights"
        )
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
            }
            torch.save(state, checkpoint_file_name)
            print(
                f" {dimension.upper()} Model weights will be saved at {checkpoint_file_name}"
            )

    def set_scroll_area(self):
        """
        Creates a scroll area.
        In case the main napari window is resized, the scroll area
        would appear.
        """
        self.scroll.setWidget(self.widget)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)

        self.setMinimumWidth(420)
        self.setCentralWidget(self.scroll)

    def save_snapshot(self, batch, outputs, path, iteration, dimension):
        import zarr

        # Save the batch and outputs to a Zarr array
        print(f"Saving snapshot to {path}/{iteration}")
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

        # Process batch arrays
        for i, arr in enumerate(batch):
            process_and_save_array(
                arr, iteration, "arr", i, is_input=i < num_inputs
            )

        # Process output arrays
        for i, arr in enumerate(outputs):
            process_and_save_array(
                arr, iteration, "outputs", i, is_input=False
            )

    def set_section_heading(self, text):
        heading = QLabel(text)

        # Set rich text formatting with lines on both sides
        heading.setStyleSheet(
            """
            QLabel {
                color: #555;
                font-size: 13px;
                padding: 5px;
                border-top: 0.5px solid #333;
                border-bottom: 0.5px solid #333;
                margin-top: 10px;
                margin-bottom: 10px;
                text-align: center;
                background: transparent;
            }
        """
        )

        heading.setAlignment(Qt.AlignCenter)
        return heading
