import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pathlib import Path

import gunpowder as gp
import pyqtgraph as pg
import torch
from napari.layers import Image, Labels
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QRadioButton,
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
from .models import get_2d_model, get_3d_model, get_loss
from .ws import watershed_from_affinities


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
        elif len(batch) == 4:
            loss = criterion(outputs, batch[2], batch[3])
        else:
            raise ValueError("Invalid batch size")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), outputs


def segment_affs(affs, method="watershed"):
    seg = watershed_from_affinities(affs[:3], fragments_in_xy=True)
    return seg[0]


class Widget(QMainWindow):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.scroll = QScrollArea()
        self.widget = QWidget()
        # initialize outer layout
        layout = QVBoxLayout()

        self.tmp_dir = Path("/tmp/models")
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
        self.device_combo_box = QComboBox(self)
        self.device_combo_box.addItem("cpu")
        self.device_combo_box.addItem("cuda:0")
        self.device_combo_box.addItem("mps")
        self.device_combo_box.setCurrentText("mps")
        self.grid_1.addWidget(device_label, 0, 0, 1, 1)
        self.grid_1.addWidget(self.device_combo_box, 0, 1, 1, 1)

    def set_grid_2(self):
        """
        2D model and training config.
        """
        raw_label = QLabel(self)
        raw_label.setText("Image layer")
        self.raw_selector = QComboBox(self)
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                self.raw_selector.addItem(f"{layer}")

        labels_label = QLabel(self)
        labels_label.setText("Labels layer")
        self.labels_selector = QComboBox(self)
        for layer in self.viewer.layers:
            if isinstance(layer, Labels):
                self.labels_selector.addItem(f"{layer}")

        # TODO: add mask layer, and option to quickly generate a mask from labels

        model_2d_type_label = QLabel(self)
        model_2d_type_label.setText("2D model type")
        self.model_2d_type_selector = QComboBox(self)
        self.model_2d_type_selector.addItems(["2d_lsd", "2d_affs", "2d_mtlsd"])

        num_fmaps_2d_label = QLabel(self)
        num_fmaps_2d_label.setText("Num fmaps 2D")
        self.num_fmaps_2d_line = QLineEdit(self)
        self.num_fmaps_2d_line.setAlignment(Qt.AlignCenter)
        self.num_fmaps_2d_line.setText("12")

        fmap_inc_factor_2d_label = QLabel(self)
        fmap_inc_factor_2d_label.setText("Fmap inc factor 2D")
        self.fmap_inc_factor_2d_line = QLineEdit(self)
        self.fmap_inc_factor_2d_line.setAlignment(Qt.AlignCenter)
        self.fmap_inc_factor_2d_line.setText("3")

        # TODO: add advanced model config dropdown. net config params, aff neighborhood, sigma

        max_iterations_2d_label = QLabel(self)
        max_iterations_2d_label.setText("Max iterations 2D")
        self.max_iterations_2d_line = QLineEdit(self)
        self.max_iterations_2d_line.setAlignment(Qt.AlignCenter)
        self.max_iterations_2d_line.setText("10000")

        batch_size_2d_label = QLabel(self)
        batch_size_2d_label.setText("Batch size 2D")
        self.batch_size_2d_line = QLineEdit(self)
        self.batch_size_2d_line.setAlignment(Qt.AlignCenter)
        self.batch_size_2d_line.setText("8")

        self.grid_2.addWidget(raw_label, 0, 0, 1, 1)
        self.grid_2.addWidget(self.raw_selector, 0, 1, 1, 1)
        self.grid_2.addWidget(labels_label, 1, 0, 1, 1)
        self.grid_2.addWidget(self.labels_selector, 1, 1, 1, 1)
        self.grid_2.addWidget(model_2d_type_label, 2, 0, 1, 1)
        self.grid_2.addWidget(self.model_2d_type_selector, 2, 1, 1, 1)
        self.grid_2.addWidget(num_fmaps_2d_label, 3, 0, 1, 1)
        self.grid_2.addWidget(self.num_fmaps_2d_line, 3, 1, 1, 1)
        self.grid_2.addWidget(fmap_inc_factor_2d_label, 4, 0, 1, 1)
        self.grid_2.addWidget(self.fmap_inc_factor_2d_line, 4, 1, 1, 1)
        self.grid_2.addWidget(max_iterations_2d_label, 5, 0, 1, 1)
        self.grid_2.addWidget(self.max_iterations_2d_line, 5, 1, 1, 1)
        self.grid_2.addWidget(batch_size_2d_label, 6, 0, 1, 1)
        self.grid_2.addWidget(self.batch_size_2d_line, 6, 1, 1, 1)

    def set_grid_3(self):
        """
        2D model load, plot, and train/stop/save buttons.
        """

        self.train_2d_model_from_scratch_checkbox = QCheckBox(self)
        self.train_2d_model_from_scratch_checkbox.setText(
            "Train 2D model from scratch"
        )

        self.load_2d_model_button = QPushButton("Load 2D model weights")

        self.losses_2d_widget = pg.PlotWidget()
        self.losses_2d_widget.setBackground((37, 41, 49))
        styles = {"color": "white", "font-size": "16px"}
        self.losses_2d_widget.setLabel("left", "Loss 2D", **styles)
        self.losses_2d_widget.setLabel("bottom", "Iterations", **styles)
        self.start_2d_training_button = QPushButton("Start training 2D")
        self.start_2d_training_button.setFixedSize(88, 30)
        self.stop_2d_training_button = QPushButton("Stop training 2D")
        self.stop_2d_training_button.setFixedSize(88, 30)
        self.save_2d_weights_button = QPushButton("Save 2D model weights")
        self.save_2d_weights_button.setFixedSize(88, 30)

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
            self.train_2d_model_from_scratch_checkbox, 0, 0, 1, 1
        )
        self.grid_3.addWidget(self.load_2d_model_button, 0, 1, 1, 1)
        self.grid_3.addWidget(self.losses_2d_widget, 1, 0, 4, 4)
        self.grid_3.addWidget(self.start_2d_training_button, 4, 0, 1, 1)
        self.grid_3.addWidget(self.stop_2d_training_button, 4, 1, 1, 1)
        self.grid_3.addWidget(self.save_2d_weights_button, 4, 2, 1, 1)

    def set_grid_4(self):
        """
        Specifies the 3D model configuration, loss plot and train/stop button.
        """
        model_3d_type_label = QLabel(self)
        model_3d_type_label.setText("3D model type")
        self.model_3d_type_selector = QComboBox(self)
        self.model_3d_type_selector.addItems(
            [
                "3d_affs_from_2d_affs",
                "3d_affs_from_2d_lsd",
                "3d_affs_from_2d_mtlsd",
            ]
        )  # TODO: update 3d model options based on selected 2d model type

        num_fmaps_3d_label = QLabel(self)
        num_fmaps_3d_label.setText("Num fmaps 3D")
        self.num_fmaps_3d_line = QLineEdit(self)
        self.num_fmaps_3d_line.setAlignment(Qt.AlignCenter)
        self.num_fmaps_3d_line.setText("8")

        fmap_inc_factor_3d_label = QLabel(self)
        fmap_inc_factor_3d_label.setText("Fmap inc factor 3D")
        self.fmap_inc_factor_3d_line = QLineEdit(self)
        self.fmap_inc_factor_3d_line.setAlignment(Qt.AlignCenter)
        self.fmap_inc_factor_3d_line.setText("3")

        max_iterations_3d_label = QLabel(self)
        max_iterations_3d_label.setText("Max iterations 3D")
        self.max_iterations_3d_line = QLineEdit(self)
        self.max_iterations_3d_line.setAlignment(Qt.AlignCenter)
        self.max_iterations_3d_line.setText("10000")

        batch_size_3d_label = QLabel(self)
        batch_size_3d_label.setText("Batch size 3D")
        self.batch_size_3d_line = QLineEdit(self)
        self.batch_size_3d_line.setAlignment(Qt.AlignCenter)
        self.batch_size_3d_line.setText("1")

        self.train_3d_model_from_scratch_checkbox = QCheckBox(
            "Train 3D model from scratch"
        )

        self.load_3d_model_button = QPushButton("Load 3D model weights")

        self.losses_3d_widget = pg.PlotWidget()
        self.losses_3d_widget.setBackground((37, 41, 49))
        styles = {"color": "white", "font-size": "16px"}
        self.losses_3d_widget.setLabel("left", "Loss 3D", **styles)
        self.losses_3d_widget.setLabel("bottom", "Iterations", **styles)
        self.start_3d_training_button = QPushButton("Start training 3d")
        self.start_3d_training_button.setFixedSize(88, 30)
        self.stop_3d_training_button = QPushButton("Stop training 3d")
        self.stop_3d_training_button.setFixedSize(88, 30)
        self.save_3d_weights_button = QPushButton("Save 3d model weights")
        self.save_3d_weights_button.setFixedSize(88, 30)

        self.grid_4.addWidget(model_3d_type_label, 0, 0, 1, 1)
        self.grid_4.addWidget(self.model_3d_type_selector, 0, 1, 1, 1)

        self.grid_4.addWidget(num_fmaps_3d_label, 1, 0, 1, 1)
        self.grid_4.addWidget(self.num_fmaps_3d_line, 1, 1, 1, 1)
        self.grid_4.addWidget(fmap_inc_factor_3d_label, 2, 0, 1, 1)
        self.grid_4.addWidget(self.fmap_inc_factor_3d_line, 2, 1, 1, 1)
        self.grid_4.addWidget(max_iterations_3d_label, 3, 0, 1, 1)
        self.grid_4.addWidget(self.max_iterations_3d_line, 3, 1, 1, 1)
        self.grid_4.addWidget(batch_size_3d_label, 4, 0, 1, 1)
        self.grid_4.addWidget(self.batch_size_3d_line, 4, 1, 1, 1)
        self.grid_4.addWidget(
            self.train_3d_model_from_scratch_checkbox, 5, 0, 1, 1
        )
        self.grid_4.addWidget(self.load_3d_model_button, 5, 1, 1, 1)

        self.grid_4.addWidget(self.losses_3d_widget, 6, 0, 2, 2)
        self.grid_4.addWidget(self.start_3d_training_button, 9, 0, 1, 1)
        self.grid_4.addWidget(self.stop_3d_training_button, 9, 1, 1, 1)
        self.grid_4.addWidget(self.save_3d_weights_button, 9, 2, 1, 1)

        # self.start_3d_training_button.setEnabled(False)
        # self.stop_3d_training_button.setEnabled(False)
        # self.save_3d_weights_button.setEnabled(False)
        # self.losses_3d_widget.hide()

        self.train_3d_model_from_scratch_checkbox.stateChanged.connect(
            lambda: self.affect_load_weights("3d")
        )
        self.train_3d_model_from_scratch_checkbox.setChecked(False)

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

        self.radio_button_group = QButtonGroup(self)
        self.radio_button_ws = QRadioButton("waterz")
        self.radio_button_mws = QRadioButton("mwatershed")
        self.radio_button_cc = QRadioButton("connected components")
        self.radio_button_group.addButton(self.radio_button_ws)
        self.radio_button_group.addButton(self.radio_button_mws)
        self.radio_button_group.addButton(self.radio_button_cc)
        self.radio_button_ws.toggled.connect(self.update_post_processing)
        self.radio_button_mws.toggled.connect(self.update_post_processing)
        self.radio_button_cc.toggled.connect(self.update_post_processing)
        self.radio_button_ws.setChecked(True)

        self.aff_bias_label = QLabel("Affinities Bias")
        self.aff_bias_line = QLineEdit(self)
        self.aff_bias_line.setAlignment(Qt.AlignCenter)
        self.aff_bias_line.textChanged.connect(self.adjust_aff_bias)

        self.start_inference_button = QPushButton("Start inference")
        self.start_inference_button.setFixedSize(140, 30)
        self.stop_inference_button = QPushButton("Stop inference")
        self.stop_inference_button.setFixedSize(140, 30)

        self.grid_5.addWidget(self.radio_button_ws, 0, 0, 1, 1)
        self.grid_5.addWidget(self.radio_button_mws, 0, 1, 1, 1)
        self.grid_5.addWidget(self.radio_button_cc, 0, 2, 1, 1)
        self.grid_5.addWidget(self.aff_bias_label, 1, 0, 1, 1)
        self.grid_5.addWidget(self.aff_bias_line, 1, 1, 1, 1)
        self.grid_5.addWidget(self.start_inference_button, 2, 0, 1, 1)
        self.grid_5.addWidget(self.stop_inference_button, 2, 1, 1, 1)
        self.start_inference_button.clicked.connect(
            self.prepare_for_start_inference
        )
        self.stop_inference_button.clicked.connect(
            self.prepare_for_stop_inference
        )

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

        self.stop_2d_training_button.setEnabled(dimension == "2d")
        self.stop_3d_training_button.setEnabled(dimension == "3d")

        self.radio_button_ws.setEnabled(False)
        self.radio_button_mws.setEnabled(False)
        self.radio_button_cc.setEnabled(False)
        self.aff_bias_line.setEnabled(False)
        self.start_inference_button.setEnabled(False)
        self.stop_inference_button.setEnabled(False)

        worker = self.train(dimension)
        setattr(self, f"train_{dimension}_worker", worker)

        worker.yielded.connect(getattr(self, f"on_yield_{dimension}_training"))
        worker.returned.connect(
            lambda: self.prepare_for_stop_training(dimension)
        )

        worker.start()

    def remove_inference_attributes(self, dimension):
        """
        When inference is initiated, then existing predictions are removed.
        """
        if hasattr(self, f"affs_{dimension}"):
            delattr(self, f"affs_{dimension}")
        if hasattr(self, f"lsds_{dimension}"):
            delattr(self, f"lsds_{dimension}")
        if hasattr(self, "segmentation"):
            delattr(self, "segmentation")

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

        model_type = getattr(
            self, f"model_{dimension}_type_selector"
        ).currentText()

        # Create torch dataset
        if dimension == "2d":
            self.napari_dataset = Napari2DDataset(
                raw_layer, labels_layer, model_type
            )
        elif dimension == "3d":
            self.napari_dataset = Napari3DDataset(model_type)

        # Create dataloader
        train_dataloader = torch.utils.data.DataLoader(
            dataset=self.napari_dataset,
            batch_size=int(
                getattr(self, f"batch_size_{dimension}_line").text()
            ),
            drop_last=True,
            num_workers=8, #train_config.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        # Load model and loss
        if dimension == "2d":
            model = get_2d_model(
                num_channels=self.napari_dataset.num_channels,
                num_fmaps=int(
                    getattr(self, f"num_fmaps_{dimension}_line").text()
                ),
                fmap_inc_factor=int(
                    getattr(self, f"fmap_inc_factor_{dimension}_line").text()
                ),
                model_type=model_type,
            )
        elif dimension == "3d":
            model = get_3d_model(
                num_channels=self.napari_dataset.num_channels,
                model_type=model_type,
                num_fmaps=int(
                    getattr(self, f"num_fmaps_{dimension}_line").text()
                ),
                fmap_inc_factor=int(
                    getattr(self, f"fmap_inc_factor_{dimension}_line").text()
                ),
            )
        self.device = torch.device(self.device_combo_box.currentText())
        setattr(self, f"model_{dimension}", model.to(self.device))
        criterion = get_loss(model_type).to(self.device)

        # Initialize model weights
        # if getattr(self, f"train_{dimension}_model_from_scratch_checkbox").isChecked():
        #     for _name, layer in getattr(self, f"model_{dimension}").named_modules():
        #         if isinstance(layer, torch.nn.modules.conv._ConvNd):
        #             torch.nn.init.kaiming_normal_(
        #                 layer.weight, nonlinearity="relu"
        #             )

        # Set optimizer
        setattr(
            self,
            f"optimizer_{dimension}",
            torch.optim.Adam(
                getattr(self, f"model_{dimension}").parameters(),
                lr=1e-4,
                weight_decay=0.01,
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
                    f"Resuming model from {getattr(self, f'pretrained_{dimension}_model_checkpoint').text()}"
                )
                state = torch.load(
                    getattr(
                        self, f"pretrained_{dimension}_model_checkpoint"
                    ).text(),
                    map_location=self.device,
                )
                setattr(
                    self,
                    f"start_iteration_{dimension}",
                    state["iterations"][-1] + 1,
                )
                if "model_state_dict" in state:
                    getattr(self, f"model_{dimension}").load_state_dict(
                        state["model_state_dict"], strict=True
                    )
                    getattr(self, f"optimizer_{dimension}").load_state_dict(
                        state["optim_state_dict"]
                    )
                    setattr(self, f"losses_{dimension}", state["losses"])
                    setattr(
                        self, f"iterations_{dimension}", state["iterations"]
                    )
                else:
                    getattr(self, f"model_{dimension}").load_state_dict(
                        state, strict=True
                    )

        # Call Train Iteration
        for iteration, batch in tqdm(
            zip(
                range(
                    getattr(self, f"start_iteration_{dimension}"),
                    int(
                        getattr(
                            self, f"max_iterations_{dimension}_line"
                        ).text()
                    ),
                ),
                train_dataloader,
                strict=False,
            )
        ):
            loss, outputs = train_iteration(
                batch,
                model=getattr(self, f"model_{dimension}"),
                criterion=criterion,
                optimizer=getattr(self, f"optimizer_{dimension}"),
                device=self.device,
                dimension=dimension,
            )
            if iteration % 1000 == 0:
                self.save_snapshot(batch, outputs, f'/tmp/snapshots_{model_type}.zarr', iteration, dimension)
            yield loss, iteration
        print(f"{dimension.upper()} Training finished!")
        return

    def on_yield_2d_training(self, loss_iteration):
        """
        The loss plot is updated every training iteration.
        """
        loss, iteration = loss_iteration
        print(f"===> Iteration: {iteration}, loss: {loss:.6f}")
        self.iterations_2d.append(iteration)
        self.losses_2d.append(loss)
        self.losses_2d_widget.plot(self.iterations_2d, self.losses_2d)

    def on_yield_3d_training(self, loss_iteration):
        """
        The loss plot is updated every training iteration.
        """
        loss, iteration = loss_iteration
        print(f"===> Iteration: {iteration}, loss: {loss:.6f}")
        self.iterations_3d.append(iteration)
        self.losses_3d.append(loss)
        self.losses_3d_widget.plot(self.iterations_3d, self.losses_3d)

    def prepare_for_stop_training(self, dimension):
        # Re-enable all training buttons and save buttons
        self.start_2d_training_button.setEnabled(True)
        self.stop_2d_training_button.setEnabled(False)
        self.save_2d_weights_button.setEnabled(True)
        self.start_3d_training_button.setEnabled(True)
        self.stop_3d_training_button.setEnabled(False)
        self.save_3d_weights_button.setEnabled(True)

        # Re-enable all other UI elements
        self.radio_button_ws.setEnabled(True)
        self.radio_button_mws.setEnabled(True)
        self.radio_button_cc.setEnabled(True)
        self.aff_bias_line.setEnabled(True)
        self.start_inference_button.setEnabled(True)
        self.stop_inference_button.setEnabled(True)

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
        self.start_2d_training_button.setEnabled(False)
        self.stop_2d_training_button.setEnabled(False)
        self.save_2d_weights_button.setEnabled(False)
        self.start_3d_training_button.setEnabled(False)
        self.stop_3d_training_button.setEnabled(False)
        self.save_3d_weights_button.setEnabled(False)

        self.radio_button_ws.setEnabled(False)
        self.radio_button_mws.setEnabled(False)
        self.radio_button_cc.setEnabled(False)
        self.aff_bias_line.setEnabled(False)
        self.start_inference_button.setEnabled(False)
        self.stop_inference_button.setEnabled(True)

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

        self.radio_button_ws.setEnabled(True)
        self.radio_button_mws.setEnabled(True)
        self.radio_button_cc.setEnabled(True)
        self.aff_bias_line.setEnabled(True)
        self.start_inference_button.setEnabled(True)
        self.stop_inference_button.setEnabled(True)

        if self.inference_worker is not None:
            self.inference_worker.quit()
            self.inference_worker = None

    @thread_worker
    def infer(self):

        model_2d_type = self.model_2d_type_selector.currentText()
        model_3d_type = self.model_3d_type_selector.currentText()
        
        for layer in self.viewer.layers:
            if f"{layer}" == self.raw_selector.currentText():
                raw_image_layer = layer
                num_channels = (
                    raw_image_layer.data.shape[0]
                    if len(raw_image_layer.data.shape) > 3
                    else 1
                )
                break

        # load model, set in eval mode
        if not hasattr(self, "device"):
            self.device = torch.device(self.device_combo_box.currentText())

        if hasattr(self, "model_2d"):
            pass
        elif hasattr(self, "pretrained_2d_model_checkpoint"):
            model_2d = get_2d_model(
                model_type=model_2d_type,
                num_channels=num_channels,
                num_fmaps=int(self.num_fmaps_2d_line.text()),
                fmap_inc_factor=int(self.fmap_inc_factor_2d_line.text()),
            )
            self.model_2d = model_2d.to(self.device)
            print(f"Loading model from {self.pretrained_2d_model_checkpoint}")
            state = torch.load(
                self.pretrained_2d_model_checkpoint, map_location=self.device
            )
            if "model_state_dict" in state:
                self.model_2d.load_state_dict(
                    state["model_state_dict"], strict=True
                )
            else:
                self.model_2d.load_state_dict(state, strict=True)
        else:
            raise ValueError("No checkpoint loaded for 3D model")

        if hasattr(self, "model_3d"):
            pass
        elif hasattr(self, "pretrained_3d_model_checkpoint"):
            model_3d = get_3d_model(
                model_type=model_3d_type,
                num_channels=self.num_channels,
                num_fmaps=int(self.num_fmaps_3d_line.text()),
                fmap_inc_factor=int(self.fmap_inc_factor_3d_line.text()),
            )
            self.model_3d = model_3d.to(self.device)
            print(f"Loading model from {self.pretrained_3d_model_checkpoint}")
            state = torch.load(
                self.pretrained_3d_model_checkpoint, map_location=self.device
            )
            if "model_state_dict" in state:
                self.model_3d.load_state_dict(
                    state["model_state_dict"], strict=True
                )
            else:
                self.model_3d.load_state_dict(state, strict=True)
        else:
            raise ValueError("No checkpoint loaded for 3D model")

        self.model_2d = self.model_2d.to(self.device)
        self.model_2d.eval()

        self.model_3d.to(self.device)
        self.model_3d.eval()

        # set up pipeline
        voxel_size = 1, 1, 1
        offset = 0, 0, 0
        input_shape_2d = 3, 212, 212
        output_shape_2d = 1, 120, 120
        input_shape_3d = 20, 332, 332
        output_shape_3d = 4, 240, 240
        roi = gp.Roi(offset, raw_image_layer.data.shape[-3:])

        raw = gp.ArrayKey("RAW")
        int_lsds = gp.ArrayKey("LSDS_2D")
        int_affs = gp.ArrayKey("AFFS_2D")
        pred_affs = gp.ArrayKey("AFFS_3D")

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

        if "3d_affs_from_2d_lsd" == model_3d_type:
            ins_3d.append(int_lsds)
        elif "3d_affs_from_2d_affs" == model_3d_type:
            ins_3d.append(int_affs)
        elif "3d_affs_from_2d_mtlsd" == model_3d_type:
            ins_3d.append(int_lsds)
            ins_3d.append(int_affs)
        else:
            raise ValueError(f"Invalid 3D model type: {model_3d_type}")

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
            outputs={
                i: k
                for i, k in enumerate(outs_2d)
            },
            array_specs={
                k: gp.ArraySpec(roi=roi, voxel_size=voxel_size)
                for k in outs_2d
            },
            device=self.device_combo_box.currentText(),
        )

        predict_3d = torchPredict(
            self.model_3d,
            inputs={
                i: k
                for i, k in enumerate(ins_3d)
            },
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
                spec=gp.ArraySpec(roi=roi, voxel_size=voxel_size),
            )
            + gp.Normalize(raw, factor=1.0 / 255)  # TODO: unharcode !!
            + gp.Pad(raw, None, "reflect")
            + gp.IntensityScaleShift(raw, 2, -1)
            + gp.Unsqueeze([raw])
            + predict_2d
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
                assert all(inp in outs_2d for inp in ins_3d), f"All 3D inputs {ins_3d} must be in 2D outputs {outs_2d}"
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
                
            pipeline_2 += predict_3d
            pipeline_2 += gp.Squeeze([pred_affs])
            pipeline_2 += gp.Scan(scan_2)

            request_2 = gp.BatchRequest()
            request_2.add(pred_affs, roi.shape)

            with gp.build(pipeline_2):
                batch = pipeline_2.request_batch(request_2)

            self.affs_3d = batch[pred_affs].data

        colormaps = ["red", "green", "blue"]

        # create napari layers for returning
        pred_layers = [
            *[
                (
                    getattr(self, str(out).lower())[:3],
                    {"name": str(out).lower().replace('_', ' '), "colormap": colormaps, "channel_axis": 0},
                    "image",
                )
                for out in outs_2d
            ],
            (
                self.affs_3d[:3],
                {"name": "3d affs", "colormap": colormaps, "channel_axis": 0},
                "image",
            ),
        ]

        # segment affs
        self.segmentation = segment_affs(self.affs_3d, method=self.seg_method)

        return pred_layers + [
            (self.segmentation, {"name": "segmentation"}, "labels")
        ]

    def on_return_infer(self, layers):
        # TODO: check if this actually deletes
        if "2d lsds" in self.viewer.layers:
            del self.viewer.layers["2d lsds"]
        if "2d affs" in self.viewer.layers:
            del self.viewer.layers["2d affs"]
        if "3d affs" in self.viewer.layers:
            del self.viewer.layers["3d affs"]
        if "segmentation" in self.viewer.layers:
            del self.viewer.layers["segmentation"]

        for data, metadata, layer_type in layers:
            if layer_type == "image":
                self.viewer.add_image(data, **metadata)
            elif layer_type == "labels":
                self.viewer.add_labels(data.astype(int), **metadata)

            if metadata["name"] != "segmentation":
                self.viewer.layers[metadata["name"]].visible = False

        self.inference_worker.quit()
        self.prepare_for_stop_inference()

    def adjust_aff_bias(self):
        self.aff_bias = list(map(float, self.aff_bias_line.text().split(",")))

    def update_post_processing(self):
        if self.radio_button_ws.isChecked():
            self.seg_method = "waterz"
        elif self.radio_button_mws.isChecked():
            self.seg_method = "mutex watershed"
        else:
            self.seg_method = "connected components"

        print(f"Segmentation method: {self.seg_method}")

    def load_weights(self, dimension):
        """
        Describes sequence of actions, which ensue after `Load Weights` button is pressed

        """
        file_name, _ = QFileDialog.getOpenFileName(
            caption=f"Load {dimension.upper()} Model Weights"
        )

        setattr(self, f"pretrained_{dimension}_model_checkpoint", file_name)

        print(
            f" {dimension.upper()} Model weights will be loaded from {getattr(self, f'pretrained_{dimension}_model_checkpoint')}"
        )

    def affect_load_weights(self, dimension):
        checkbox = getattr(
            self, f"train_{dimension}_model_from_scratch_checkbox"
        )
        load_button = getattr(self, f"load_{dimension}_model_button")

        if checkbox.isChecked():
            load_button.setEnabled(False)
        else:
            load_button.setEnabled(True)

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

        # self.setFixedWidth(320)
        self.setCentralWidget(self.scroll)

    def save_snapshot(self, batch, outputs, path, iteration, dimension):
        import zarr
        # Save the batch and outputs to a Zarr array
        print(f"Saving snapshot to {path}")
        f = zarr.open(path, mode="a")

        is_2d = dimension == "2d"
        offset = (8, 46, 46) if not is_2d else (0, 46, 46)
        
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
            f[arr_name].attrs["offset"] = (0, 0, 0) if (is_input and idx == 0) else offset
            f[arr_name].attrs["voxel_size"] = (1, 1, 1)
            f[arr_name].attrs["axis_names"] = ["c^", "z", "y", "x"]
        
        # Process batch arrays
        for i, arr in enumerate(batch):
            process_and_save_array(arr, iteration, "arr", i, is_input=True)
        
        # Process output arrays
        for i, arr in enumerate(outputs):
            process_and_save_array(arr, iteration, "outputs", i, is_input=False)

