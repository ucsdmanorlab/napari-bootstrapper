import time
from typing import TYPE_CHECKING

import pyqtgraph as pg
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

if TYPE_CHECKING:
    pass


class Widget(QMainWindow):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.scroll = QScrollArea()
        self.widget = QWidget()
        # initialize outer layout
        layout = QVBoxLayout()

        # initialize individual grid layouts from top to bottom
        self.grid_0 = QGridLayout()  # title
        self.set_grid_0()
        self.grid_1 = QGridLayout()  # device
        self.set_grid_1()
        self.grid_2 = QGridLayout()  # train configs
        self.set_grid_2()
        self.grid_3 = QGridLayout()  # model configs
        self.set_grid_3()
        self.grid_4 = QGridLayout()  # loss plot and train/stop button
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
        Specifies the training configurations.
        """
        raw_label = QLabel(self)
        raw_label.setText("Image layer")
        self.raw_selector = QComboBox(self)
        for layer in self.viewer.layers:  # TODO: only show image layers
            self.raw_selector.addItem(f"{layer}")

        labels_label = QLabel(self)
        labels_label.setText("Labels layer")
        self.labels_selector = QComboBox(self)
        for layer in self.viewer.layers:  # TODO: only show labels layers
            self.labels_selector.addItem(f"{layer}")

        # TODO: add mask layer, and option to quickly generate a mask from labels

        # TODO: include custom voxel size
        # self.voxel_size_label = QLabel(self)
        # self.voxel_size_label.setText("Voxel size (in nm)")
        # self.voxel_size_line = QLineEdit(self)
        # self.voxel_size_line.setAlignment(Qt.AlignCenter)
        # self.voxel_size_line.setText("40,8,8")

        model_dir_label = QLabel(self)
        model_dir_label.setText("Model directory")
        self.model_dir_line = QLineEdit(self)
        self.model_dir_line.setAlignment(Qt.AlignCenter)
        self.model_dir_line.setText("/tmp/models")

        max_iterations_label = QLabel(self)
        max_iterations_label.setText("Max iterations")
        self.max_iterations_line = QLineEdit(self)
        self.max_iterations_line.setAlignment(Qt.AlignCenter)
        self.max_iterations_line.setText("10000")

        save_every_label = QLabel(self)
        save_every_label.setText("Save every")
        self.save_every_line = QLineEdit(self)
        self.save_every_line.setAlignment(Qt.AlignCenter)
        self.save_every_line.setText("1000")

        self.grid_2.addWidget(raw_label, 0, 0, 1, 1)
        self.grid_2.addWidget(self.raw_selector, 0, 1, 1, 1)
        self.grid_2.addWidget(labels_label, 1, 0, 1, 1)
        self.grid_2.addWidget(self.labels_selector, 1, 1, 1, 1)
        self.grid_2.addWidget(model_dir_label, 2, 0, 1, 1)
        self.grid_2.addWidget(self.model_dir_line, 2, 1, 1, 1)
        self.grid_2.addWidget(max_iterations_label, 3, 0, 1, 1)
        self.grid_2.addWidget(self.max_iterations_line, 3, 1, 1, 1)
        self.grid_2.addWidget(save_every_label, 4, 0, 1, 1)
        self.grid_2.addWidget(self.save_every_line, 4, 1, 1, 1)

    def set_grid_3(self):
        """
        Specifies the model configuration.
        """
        model_2d_type = QLabel(self)
        model_2d_type.setText("2D model type")
        self.model_2d_type_selector = QComboBox(self)
        self.model_2d_type_selector.addItems(["2d_lsd", "2d_affs", "2d_mtlsd"])

        model_3d_type = QLabel(self)
        model_3d_type.setText("3D model type")
        self.model_3d_type_selector = QComboBox(self)
        self.model_3d_type_selector.addItems(
            [
                "3d_affs_from_2d_affs",
                "3d_affs_from_2d_lsd",
                "3d_affs_from_2d_mtlsd",
            ]
        )  # TODO: update 3d model options based on selected 2d model type

        self.train_3d_model_from_scratch_checkbox = QCheckBox(
            "Train 3D model from scratch"
        )

        self.train_3d_model_from_scratch_checkbox.stateChanged.connect(
            self.affect_train_3d_start_stop
        )

        self.load_3d_model_button = QPushButton("Load 3D model weights")
        self.load_3d_model_button.clicked.connect(self.load_weights)
        self.train_3d_model_from_scratch_checkbox.setChecked(False)

        # TODO: add advanced model config dropdown. net config params, aff neighborhood, sigma

        self.grid_3.addWidget(model_2d_type, 0, 0, 1, 1)
        self.grid_3.addWidget(self.model_2d_type_selector, 0, 1, 1, 1)
        self.grid_3.addWidget(model_3d_type, 1, 0, 1, 1)
        self.grid_3.addWidget(self.model_3d_type_selector, 1, 1, 1, 1)
        self.grid_3.addWidget(
            self.train_3d_model_from_scratch_checkbox, 2, 0, 1, 2
        )
        self.grid_3.addWidget(self.load_3d_model_button, 3, 0, 1, 2)

    def set_grid_4(self):
        """
        Specifies the loss plot and train/stop button.
        """
        # 2d model
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

        self.grid_4.addWidget(self.losses_2d_widget, 0, 0, 4, 4)
        self.grid_4.addWidget(self.start_2d_training_button, 4, 0, 1, 1)
        self.grid_4.addWidget(self.stop_2d_training_button, 4, 1, 1, 1)
        self.grid_4.addWidget(self.save_2d_weights_button, 4, 2, 1, 1)

        self.start_2d_training_button.clicked.connect(
            self.prepare_for_start_2d_training
        )
        self.stop_2d_training_button.clicked.connect(
            self.prepare_for_stop_2d_training
        )
        self.save_2d_weights_button.clicked.connect(self.save_weights)

        # 3d model
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

        self.grid_4.addWidget(self.losses_3d_widget, 5, 0, 4, 4)
        self.grid_4.addWidget(self.start_3d_training_button, 9, 0, 1, 1)
        self.grid_4.addWidget(self.stop_3d_training_button, 9, 1, 1, 1)
        self.grid_4.addWidget(self.save_3d_weights_button, 9, 2, 1, 1)

        self.start_3d_training_button.setEnabled(False)
        self.stop_3d_training_button.setEnabled(False)
        self.save_3d_weights_button.setEnabled(False)
        self.losses_3d_widget.hide()

        self.start_3d_training_button.clicked.connect(
            self.prepare_for_start_3d_training
        )
        self.stop_3d_training_button.clicked.connect(
            self.prepare_for_stop_3d_training
        )
        self.save_3d_weights_button.clicked.connect(self.save_weights)

    def set_grid_5(self):
        model_2d_iteration_label = QLabel("Model 2D Iteration")
        self.model_2d_iteration_line = QLineEdit(self)
        self.model_2d_iteration_line.textChanged.connect(
            self.update_model_2d_iteration
        )
        self.model_2d_iteration_line.setAlignment(Qt.AlignCenter)
        self.model_2d_iteration_line.setText("latest")

        model_3d_iteration_label = QLabel("Model 3D Iteration")
        self.model_3d_iteration_line = QLineEdit(self)
        self.model_3d_iteration_line.textChanged.connect(
            self.update_model_3d_iteration
        )
        self.model_3d_iteration_line.setAlignment(Qt.AlignCenter)
        self.model_3d_iteration_line.setText("latest")

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

        self.grid_5.addWidget(model_2d_iteration_label, 0, 0, 1, 1)
        self.grid_5.addWidget(self.model_2d_iteration_line, 0, 1, 1, 1)
        self.grid_5.addWidget(model_3d_iteration_label, 1, 0, 1, 1)
        self.grid_5.addWidget(self.model_3d_iteration_line, 1, 1, 1, 1)

        self.grid_5.addWidget(self.radio_button_ws, 2, 0, 1, 1)
        self.grid_5.addWidget(self.radio_button_mws, 2, 1, 1, 1)
        self.grid_5.addWidget(self.radio_button_cc, 2, 2, 1, 1)
        self.grid_5.addWidget(self.aff_bias_label, 3, 0, 1, 1)
        self.grid_5.addWidget(self.aff_bias_line, 3, 1, 1, 1)
        self.grid_5.addWidget(self.start_inference_button, 4, 0, 1, 1)
        self.grid_5.addWidget(self.stop_inference_button, 4, 1, 1, 1)
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

    def prepare_for_start_2d_training(self):
        """
        If training, other buttons, except stop, are disabled.
        """
        self.start_2d_training_button.setEnabled(False)
        self.stop_2d_training_button.setEnabled(True)
        self.save_2d_weights_button.setEnabled(False)
        self.start_3d_training_button.setEnabled(False)
        self.stop_3d_training_button.setEnabled(False)
        self.save_3d_weights_button.setEnabled(False)

        self.model_2d_iteration_line.setEnabled(False)
        self.model_3d_iteration_line.setEnabled(False)
        self.radio_button_ws.setEnabled(False)
        self.radio_button_mws.setEnabled(False)
        self.radio_button_cc.setEnabled(False)
        self.aff_bias_line.setEnabled(False)
        self.start_inference_button.setEnabled(False)
        self.stop_inference_button.setEnabled(False)

        self.train_2d_worker = self.train_2d()
        self.train_2d_worker.yielded.connect(self.on_yield_2d_training)
        self.train_2d_worker.returned.connect(
            self.prepare_for_stop_2d_training
        )
        self.train_2d_worker.start()

    def prepare_for_start_3d_training(self):
        """
        If training, other buttons, except stop, are disabled.
        """
        self.start_3d_training_button.setEnabled(False)
        self.stop_3d_training_button.setEnabled(True)
        self.save_3d_weights_button.setEnabled(False)
        self.start_2d_training_button.setEnabled(False)
        self.stop_2d_training_button.setEnabled(False)
        self.save_2d_weights_button.setEnabled(False)

        self.model_2d_iteration_line.setEnabled(False)
        self.model_3d_iteration_line.setEnabled(False)
        self.radio_button_ws.setEnabled(False)
        self.radio_button_mws.setEnabled(False)
        self.radio_button_cc.setEnabled(False)
        self.aff_bias_line.setEnabled(False)
        self.start_inference_button.setEnabled(False)
        self.stop_inference_button.setEnabled(False)

        self.train_3d_worker = self.train_3d()
        self.train_3d_worker.yielded.connect(self.on_yield_3d_training)
        self.train_3d_worker.returned.connect(
            self.prepare_for_stop_3d_training
        )
        self.train_3d_worker.start()

    @thread_worker
    def train_2d(self):

        if not hasattr(self, "losses_2d"):
            self.losses_2d = []
        if not hasattr(self, "iterations_2d"):
            self.iterations_2d = []
        if not hasattr(self, "start_iteration_2d"):
            self.start_iteration_2d = 0

        # if self.train_2d_model_from_scratch_checkbox.isChecked():
        #     self.losses_2d, self.iterations_2d = [], []
        #     self.start_iteration_2d = 0
        #     self.losses_2d_widget.clear()

        for i in range(
            self.start_iteration_2d, int(self.max_iterations_line.text())
        ):
            time.sleep(1)
            yield float(i) ** 2, i
        return

    @thread_worker
    def train_3d(self):

        if not hasattr(self, "losses_3d"):
            self.losses_3d = []
        if not hasattr(self, "iterations_3d"):
            self.iterations_3d = []
        if not hasattr(self, "start_iteration_3d"):
            self.start_iteration_3d = 0

        # if self.train_2d_model_from_scratch_checkbox.isChecked():
        #     self.losses_2d, self.iterations_2d = [], []
        #     self.start_iteration_2d = 0
        #     self.losses_2d_widget.clear()

        for i in range(
            self.start_iteration_3d, int(self.max_iterations_line.text())
        ):
            time.sleep(1)
            yield float(i) ** 2, i
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

    def prepare_for_stop_2d_training(self):
        """
        This function defines the sequence of events once training is stopped.
        """
        self.start_2d_training_button.setEnabled(True)
        self.stop_2d_training_button.setEnabled(False)
        self.save_2d_weights_button.setEnabled(True)
        self.start_3d_training_button.setEnabled(True)
        self.stop_3d_training_button.setEnabled(False)
        self.save_3d_weights_button.setEnabled(True)

        self.model_2d_iteration_line.setEnabled(True)
        self.model_3d_iteration_line.setEnabled(True)
        self.radio_button_ws.setEnabled(True)
        self.radio_button_mws.setEnabled(True)
        self.radio_button_cc.setEnabled(True)
        self.aff_bias_line.setEnabled(True)
        self.start_inference_button.setEnabled(True)
        self.stop_inference_button.setEnabled(True)

        if self.train_2d_worker is not None:
            # state = {
            #     "model_state_dict": self.model.state_dict(),
            #     "optim_state_dict": self.optimizer.state_dict(),
            #     "iterations": self.iterations_2d,
            #     "losses": self.losses_2d,
            # }
            # checkpoint_file_name = Path("/tmp/models") / "last.pth"
            # torch.save(state, checkpoint_file_name)
            self.train_2d_worker.quit()
            # self.model_2d_config.checkpoint = checkpoint_file_name

    def prepare_for_stop_3d_training(self):
        """
        This function defines the sequence of events once training is stopped.
        """
        self.start_3d_training_button.setEnabled(True)
        self.stop_3d_training_button.setEnabled(False)
        self.save_3d_weights_button.setEnabled(True)
        self.start_2d_training_button.setEnabled(True)
        self.stop_2d_training_button.setEnabled(False)
        self.save_2d_weights_button.setEnabled(True)

        self.model_2d_iteration_line.setEnabled(True)
        self.model_3d_iteration_line.setEnabled(True)
        self.radio_button_ws.setEnabled(True)
        self.radio_button_mws.setEnabled(True)
        self.radio_button_cc.setEnabled(True)
        self.aff_bias_line.setEnabled(True)
        self.start_inference_button.setEnabled(True)
        self.stop_inference_button.setEnabled(True)

        if self.train_3d_worker is not None:
            # state = {
            #     "model_state_dict": self.model.state_dict(),
            #     "optim_state_dict": self.optimizer.state_dict(),
            #     "iterations": self.iterations_3d,
            #     "losses": self.losses_3d,
            # }
            # checkpoint_file_name = Path("/tmp/models") / "last.pth"
            # torch.save(state, checkpoint_file_name)
            self.train_3d_worker.quit()
            # self.model_3d_config.checkpoint = checkpoint_file_name

    def prepare_for_start_inference(self):
        self.start_2d_training_button.setEnabled(False)
        self.stop_2d_training_button.setEnabled(False)
        self.save_2d_weights_button.setEnabled(False)
        self.start_3d_training_button.setEnabled(False)
        self.stop_3d_training_button.setEnabled(False)
        self.save_3d_weights_button.setEnabled(False)

        self.model_2d_iteration_line.setEnabled(False)
        self.model_3d_iteration_line.setEnabled(False)
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

        self.model_2d_iteration_line.setEnabled(True)
        self.model_3d_iteration_line.setEnabled(True)
        self.radio_button_ws.setEnabled(True)
        self.radio_button_mws.setEnabled(True)
        self.radio_button_cc.setEnabled(True)
        self.aff_bias_line.setEnabled(True)
        self.start_inference_button.setEnabled(True)
        self.stop_inference_button.setEnabled(True)

        if self.inference_worker is not None:
            self.inference_worker.quit()

    @thread_worker
    def infer(self):
        import random

        import numpy as np

        pred_2d = random.choice(
            [
                np.random.rand(2, 62, 625, 625),
                np.random.rand(6, 62, 625, 625),
                [
                    np.random.rand(6, 62, 625, 625),
                    np.random.rand(2, 62, 625, 625),
                ],
            ]
        )
        if pred_2d is list:
            preds_2d = [
                (pred, {"name": f"2D pred {i}"}, "image")
                for i, pred in enumerate(pred_2d)
            ]
        else:
            preds_2d = [(pred_2d, {"name": "2D pred"}, "image")]
        affs_3d = np.random.rand(3, 62, 625, 625)
        seg = np.random.rand(62, 625, 625) > 0.5

        return (
            *preds_2d,
            (affs_3d, {"name": "3D affs"}, "image"),
            (seg, {"name": "Segmentation"}, "labels"),
        )

    def on_return_infer(self, layers):
        if "2D pred 0" in self.viewer.layers:
            del self.viewer.layers["2D pred 0"]
            del self.viewer.layers["2D pred 1"]
        if "2D pred" in self.viewer.layers:
            del self.viewer.layers["2D pred"]
        if "3D affs" in self.viewer.layers:
            del self.viewer.layers["3D affs"]
        if "Segmentation" in self.viewer.layers:
            del self.viewer.layers["Segmentation"]

        for data, metadata, layer_type in layers:
            if layer_type == "image":
                self.viewer.add_image(data, **metadata)
            elif layer_type == "labels":
                self.viewer.add_labels(data.astype(int), **metadata)

            if metadata["name"] != "Segmentation":
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

    def update_model_2d_iteration(self):
        if self.model_2d_iteration_line.text() == "latest":
            self.model_2d_iteration = -1
        else:
            self.model_2d_iteration = int(self.model_2d_iteration_line.text())

        print(f"Model 2D iteration: {self.model_2d_iteration}")

    def update_model_3d_iteration(self):
        if self.model_3d_iteration_line.text() == "latest":
            self.model_3d_iteration = -1
        else:
            self.model_3d_iteration = int(self.model_3d_iteration_line.text())

        print(f"Model 3D iteration: {self.model_3d_iteration}")

    def load_weights(self):
        """
        Describes sequence of actions, which ensue after `Load Weights` button is pressed

        """
        file_name, _ = QFileDialog.getOpenFileName(
            caption="Load Model Weights"
        )
        self.pre_trained_model_checkpoint = file_name
        print(
            f"Model weights will be loaded from {self.pre_trained_model_checkpoint}"
        )

    def affect_train_3d_start_stop(self):
        """
        In case `train from scratch` checkbox is selected,
        the `Start/Stop training 3D` button is disabled, and vice versa.
        """
        if self.train_3d_model_from_scratch_checkbox.isChecked():
            self.load_3d_model_button.setEnabled(False)
            self.start_3d_training_button.setEnabled(True)
            self.stop_3d_training_button.setEnabled(False)
            self.save_3d_weights_button.setEnabled(True)
            self.losses_3d_widget.show()
        else:
            self.load_3d_model_button.setEnabled(True)
            self.start_3d_training_button.setEnabled(False)
            self.stop_3d_training_button.setEnabled(False)
            self.save_3d_weights_button.setEnabled(False)
            self.losses_3d_widget.hide()

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

        self.setFixedWidth(320)
        self.setCentralWidget(self.scroll)

    def save_weights(self):
        """
        Describes sequence of actions which ensue, after `Save weights` button is pressed

        """
        checkpoint_file_name, _ = QFileDialog.getSaveFileName(
            caption="Save Model Weights"
        )
        if (
            hasattr(self, "model")
            and hasattr(self, "optimizer")
            and hasattr(self, "iterations")
            and hasattr(self, "losses")
        ):
            # state = {
            #     "model_state_dict": self.model.state_dict(),
            #     "optim_state_dict": self.optimizer.state_dict(),
            #     "iterations": self.iterations,
            #     "losses": self.losses,
            # }
            # torch.save(state, checkpoint_file_name)
            print(f"Model weights will be saved at {checkpoint_file_name}")
