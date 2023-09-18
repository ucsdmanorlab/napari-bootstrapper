import dataclasses
import contextlib

# python built in libraries
from pathlib import Path
from typing import List, Optional
import attrs
from attrs.validators import instance_of

# github repo libraries
import gunpowder as gp

# pip installed libraries
import napari
import numpy as np
import torch
from magicgui import magic_factory
from magicgui.widgets import Container

# widget stuff
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from napari.qt.threading import FunctionWorker, thread_worker
from qtpy.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QHBoxLayout,
    QGroupBox,
)
from qtpy.QtCore import QUrl
from superqt import QCollapsible
from tqdm import tqdm

from .dataset import NapariDataset2D
from ..models.model_1 import Model2D, WeightedMSELoss, ModelConfig2D

from ..gp.nodes.napari_source_2d import NapariSource2D

# local package imports
from .gui_helpers import MplCanvas, layer_choice_widget


@attrs.define
class TrainConfig2D:
    """Train configuration.

    Parameters:

        crop_size:
        input_shape:

            The size of the crops - specified as a list of number of pixels -
            extracted from the raw images, used during training.

        batch_size:

            The number of samples to use per batch.

        max_iterations:

            The maximum number of iterations to train for.

        initial_learning_rate (default = 4e-5):

            Initial learning rate of the optimizer.

        save_model_every (default = 1e3):

            The model weights are saved every few iterations.

        save_snapshot_every (default = 1e3):

            The zarr snapshot is saved every few iterations.

        num_workers (default = 8):

            The number of sub-processes to use for data-loading.

        control_point_spacing (default = 64):

            The distance in pixels between control points used for elastic
            deformation of the raw data during training.

        control_point_jitter (default = 2.0):

            How much to jitter the control points for elastic deformation
            of the raw data during training, given as the standard deviation of
            a normal distribution with zero mean.

        min_masked (default = 0.1):
            
            How much of a requested batch is required to have masked-in labels
            while selecting a random location.

#        train_data_config:
#
#            Configuration object for the training data.

        device (default = 'cuda:0'):

            The device to train on.
            Set to 'cpu' to train without GPU.


    """

#    crop_size: List = attrs.field(default=[252, 252], validator=instance_of(List))
    voxel_size: List = attrs.field(default=[1, 1], validator=instance_of(List))
    batch_size: int = attrs.field(default=8, validator=instance_of(int))
    max_iterations: int = attrs.field(default=100_000, validator=instance_of(int))
    initial_learning_rate: float = attrs.field(
        default=4e-5, validator=instance_of(float)
    )
    save_model_every: int = attrs.field(default=1_000, validator=instance_of(int))
    save_snapshot_every: int = attrs.field(default=1_000, validator=instance_of(int))
    num_workers: int = attrs.field(default=8, validator=instance_of(int))

    control_point_spacing: int = attrs.field(default=64, validator=instance_of(int))
    control_point_jitter: float = attrs.field(default=2.0, validator=instance_of(float))
    min_masked: float = attrs.field(default=0.1, validator=instance_of(float))
    device: str = attrs.field(default="cpu", validator=instance_of(str))
    #device: str = attrs.field(default="cuda:0", validator=instance_of(str))


@dataclasses.dataclass
class TrainingStats:
    iteration: int = 0
    losses: list[float] = dataclasses.field(default_factory=list)
    iterations: list[int] = dataclasses.field(default_factory=list)

    def reset(self):
        self.iteration = 0
        self.losses = []
        self.iterations = []

    def load(self, other):
        self.iteration = other.iteration
        self.losses = other.losses
        self.iterations = other.iterations


################################## GLOBALS ####################################
_train_config = None
_model_config = None
_model = None
_optimizer = None
_scheduler = None
_training_stats = TrainingStats()


def get_training_stats():
    global _training_stats
    return _training_stats


def get_train_config(**kwargs):
    global _train_config
    # set dataset configs to None
#    kwargs["train_data_config"] = None
#    kwargs["validate_data_config"] = None
    if _train_config is None:
        _train_config = TrainConfig2D(**kwargs)
    elif len(kwargs) > 0:
        for k, v in kwargs.items():
            _train_config.__setattr__(k, v)
    return _train_config


@magic_factory(call_button="Save")
def train_config_widget(
#    crop_size: list[int] = [256, 256],
    voxel_size: list[int] = [1,1],
    batch_size: int = 8,
    max_iterations: int = 100_000,
    initial_learning_rate: float = 4e-5,
    save_model_every: int = 1_000,
    save_snapshot_every: int = 1_000,
    num_workers: int = 8,
    control_point_spacing: int = 64,
    control_point_jitter: float = 2.0,
):
    get_train_config(
#        crop_size=crop_size,
        voxel_size=voxel_size,
        batch_size=batch_size,
        max_iterations=max_iterations,
        initial_learning_rate=initial_learning_rate,
        save_model_every=save_model_every,
        save_snapshot_every=save_snapshot_every,
        num_workers=num_workers,
        control_point_spacing=control_point_spacing,
        control_point_jitter=control_point_jitter,
    )


def get_model_config(**kwargs):
    global _model_config
    if _model_config is None:
        _model_config = ModelConfig2D(**kwargs)
    elif len(kwargs) > 0:
        for k, v in kwargs.items():
            _model_config.__setattr__(k, v)
    return _model_config


@magic_factory
def model_config_widget(
    num_fmaps: int = 12,
    fmap_inc_factor: int = 5,
    downsampling_factors: list[list[int]] = [[2, 2], [2, 2], [2, 2]],
):
    get_model_config(
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        downsampling_factors=downsampling_factors,
    )


def get_training_state(dataset: Optional[NapariDataset2D] = None):
    global _model
    global _optimizer
    global _scheduler
    global _model_config
    global _train_config
    if _model_config is None:
        _model_config = ModelConfig2D(
            num_fmaps=12, 
            fmap_inc_factor=5,
            downsampling_factors= [[2, 2], [2, 2], [2, 2]]
        )
    if _train_config is None:
        _train_config = get_train_config()
    if _model is None:
        # Build model
        _model = Model2D(
            in_channels=dataset.get_num_channels(), #TODO: check, maybe hardcode
            out_channels=6, 
            num_fmaps=_model_config.num_fmaps,
            fmap_inc_factor=_model_config.fmap_inc_factor,
            downsampling_factors=[
                tuple(factor) for factor in _model_config.downsampling_factors
            ],
        )#.cuda()

        # Weight initialization
        for _name, layer in _model.named_modules():
            if isinstance(layer, torch.nn.modules.conv._ConvNd):
                torch.nn.init.kaiming_normal_(
                    layer.weight, nonlinearity="relu"
                )

        _optimizer = torch.optim.Adam(
            _model.parameters(),
            lr=_train_config.initial_learning_rate,
        )

        def lambda_(iteration):
            return pow((1 - ((iteration) / _train_config.max_iterations)), 0.9)

        _scheduler = torch.optim.lr_scheduler.LambdaLR(
            _optimizer, lr_lambda=lambda_
        )
    return (_model, _optimizer, _scheduler)


class TrainWidget(QWidget):
    def __init__(self, napari_viewer):
        # basic initialization
        self.viewer = napari_viewer
        super().__init__()

        # initialize state variables
        self.__training_generator = None

        # Widget layout
        layout = QVBoxLayout()

        # add loss/iterations widget
        self.progress_plot = MplCanvas(self, width=5, height=3, dpi=100)
        toolbar = NavigationToolbar(self.progress_plot, self)
        progress_plot_layout = QVBoxLayout()
        progress_plot_layout.addWidget(toolbar)
        progress_plot_layout.addWidget(self.progress_plot)
        self.loss_plot = None
        self.val_plot = None
        plot_container_widget = QWidget()
        plot_container_widget.setLayout(progress_plot_layout)
        layout.addWidget(plot_container_widget)

        # add raw layer choice
        self.raw_selector = layer_choice_widget(
            self.viewer,
            annotation=napari.layers.Image,
            name="raw",
        )
        layout.addWidget(self.raw_selector.native)	

        # add labels layer choice
        self.labels_selector = layer_choice_widget(
            self.viewer,
            annotation=napari.layers.Labels,
            name="labels",
        )
        layout.addWidget(self.labels_selector.native)	

#        self.s_checkbox = QCheckBox('s')
#        self.c_checkbox = QCheckBox('c')
#        self.t_checkbox = QCheckBox('t')
#        self.z_checkbox = QCheckBox('z')
#        self.y_checkbox = QCheckBox('y')
#        self.x_checkbox = QCheckBox('x')
#
#        axis_layout = QHBoxLayout()
#        axis_layout.addWidget(self.s_checkbox)
#        axis_layout.addWidget(self.c_checkbox)
#        axis_layout.addWidget(self.t_checkbox)
#        axis_layout.addWidget(self.z_checkbox)
#        axis_layout.addWidget(self.y_checkbox)
#        axis_layout.addWidget(self.x_checkbox)
#
#        self.axis_selector = QGroupBox("Axis Names:")
#        self.axis_selector.setLayout(axis_layout)
#        layout.addWidget(self.axis_selector)

        # add buttons
        self.train_button = QPushButton("Train!", self)
        self.train_button.clicked.connect(self.train) # TODO: what is self.train??
        layout.addWidget(self.train_button)

        # add save and load widgets
        collapsable_save_load_widget = QCollapsible("Save/Load", self)
        collapsable_save_load_widget.addWidget(self.save_widget.native)
        collapsable_save_load_widget.addWidget(self.load_widget.native)

        layout.addWidget(collapsable_save_load_widget)

        # Add inference widget
        collapsable_inference_widget = QCollapsible("Run Inference", self)
        collapsable_inference_widget.addWidget(self.inference_widget)
        layout.addWidget(collapsable_inference_widget)

        # activate layout
        self.setLayout(layout)

        # Widget state
        self.model = None
        self.reset_training_state()

        # TODO: Can we do this better?
        # connect napari events
        self.viewer.layers.events.inserted.connect(
            self.__inference_widget.raw.reset_choices
        )
        self.viewer.layers.events.removed.connect(
            self.__inference_widget.raw.reset_choices
        )

        # handle button activations and deactivations
        # buttons: save, load, (train/pause), inference
        self.save_button = self.__save_widget.call_button.native
        self.load_button = self.__load_widget.call_button.native
        self.inference_button = self.__inference_widget.call_button.native
        self.inference_button.clicked.connect(
            lambda: self.set_buttons("inferenceing")
        )

        self.set_buttons("initial")

    def set_buttons(self, state: str):
        if state == "training":
            self.train_button.setText("Pause!")
            self.train_button.setEnabled(True)
            self.save_button.setText("Stop training to save!")
            self.save_button.setEnabled(False)
            self.load_button.setText("Stop training to load!")
            self.load_button.setEnabled(False)
            self.inference_button.setText("Stop training to run inference!")
            self.inference_button.setEnabled(False)
        if state == "paused":
            self.train_button.setText("Train!")
            self.train_button.setEnabled(True)
            self.save_button.setText("Save")
            self.save_button.setEnabled(True)
            self.load_button.setText("Load")
            self.load_button.setEnabled(True)
            self.inference_button.setText("Inference")
            self.inference_button.setEnabled(True)
        if state == "inferenceing":
            self.train_button.setText("Can't train while inferenceing!")
            self.train_button.setEnabled(False)
            self.save_button.setText("Can't save while inferenceing!")
            self.save_button.setEnabled(False)
            self.load_button.setText("Can't load while inferenceing!")
            self.load_button.setEnabled(False)
            self.inference_button.setText("Running inference...")
            self.inference_button.setEnabled(False)
        if state == "initial":
            self.train_button.setText("Train!")
            self.train_button.setEnabled(True)
            self.save_button.setText("No state to Save!")
            self.save_button.setEnabled(False)
            self.load_button.setText(
                "Load data and test data before loading an old model!"
            )
            self.load_button.setEnabled(False)
            self.inference_button.setText("Inference")
            self.inference_button.setEnabled(True)

    @property
    def inference_widget(self):
        @magic_factory(call_button="Inference")
        def inference(
            raw: napari.layers.Image,
            checkpoint: Path = Path("checkpoint.pt"),
            crop_size: list[int] = [252, 252],
        ) -> FunctionWorker[List[napari.types.LayerDataTuple]]:
            # TODO: do this better?
            @thread_worker(
                connect={"returned": lambda: self.set_buttons("paused")},
                progress={"total": 0, "desc": "Inferenceing"},
            )
            def async_inference(
                raw: napari.layers.Image,
                section: int,
                checkpoint: Path,
                crop_size: list[int],
            ) -> np.ndarray: #List[napari.types.LayerDataTuple]:
                raw.data = raw.data.astype(np.float32)

                global _model
                assert (
                    _model is not None
                ), "You must train a model before running inference"
                model = _model
                model.eval()

                num_spatial_dims = 2 #len(raw.data.shape) - 2 #TODO
                num_channels = raw.data.shape[0] if len(raw.data.shape) == 4 else 1 #raw.data.shape[0] #TODO

                voxel_size = gp.Coordinate((1,) * num_spatial_dims)
#                model.set_infer(
#                    p_salt_pepper=p_salt_pepper,
#                    num_infer_iterations=num_infer_iterations,
#                )

                # prediction crop size is the size of the scanned tiles to be provided to the model
                input_shape = gp.Coordinate((1, num_channels, *crop_size))

                output_shape = gp.Coordinate(
                    model(
                        torch.zeros(
                            (
                                1,
                                num_channels,
                                *crop_size,
                            ),
                            dtype=torch.float32,
                        ).to(device)
                    ).shape
                )

                input_size = gp.Coordinate(input_shape[2:]) * voxel_size
                output_size = gp.Coordinate(output_shape[2:]) * voxel_size

                context = (input_size - output_size) / 2

                raw_key = gp.ArrayKey("RAW")
                prediction_key = gp.ArrayKey("PREDICT")

                scan_request = gp.BatchRequest()
                scan_request.add(raw_key, input_size)
                scan_request.add(prediction_key, output_size)

                source = NapariSource2D(raw, raw_key, section)

                with gp.build(source):
                    total_input_roi = source.spec[raw].roi
                    total_output_roi = source.spec[raw].roi.grow(-context,-context)

#                for i in range(len(voxel_size)):
#                    assert total_output_roi.get_shape()[i]/voxel_size[i] >= output_shape[i], \
#                        f"total output (write) ROI cannot be smaller than model's output shape, \noffending index: {i}\ntotal_output_roi: {total_output_roi.get_shape()}, \noutput_shape: {output_shape}, \nvoxel size: {voxel_size}"

                predict = gp.torch.Predict(
                    model,
                    inputs={"raw": raw_key},
                    outputs={0: prediction_key},
                    array_specs={
                        prediction_key: gp.ArraySpec(voxel_size=voxel_size)
                    },
                )

                channel_unsqueeze = gp.Unsqueeze([raw_key])
                if (num_channels == 1 and len(raw.data.shape) == 4) or len(raw.data.shape) == 4:
                    channel_unsqueeze += gp.Squeeze([raw_key])
                
                pipeline = (
                    source
                    + gp.Normalize(raw_key)
                    + gp.Pad(raw_key, context)
                    + gp.Unsqueeze([raw_key])
                    + channel_unsqueeze
                    + predict
                    + gp.Squeeze([raw_key])
                    + gp.Scan(scan_request)
                )

                # request to pipeline for ROI of whole image/volume
                request = gp.BatchRequest()
                request.add(raw_key, raw.data.shape[2:])
                request.add(prediction_key, raw.data.shape[2:])
                with gp.build(pipeline):
                    batch = pipeline.request_batch(request)

                prediction = batch.arrays[prediction_key].data
                print("TEST")
                print(prediction.shape)
                return prediction

            available_sections = list(np.where(np.any(raw.data, axis=(-2,-1)))[0])
            #predictions = [] 

            return np.stack([
                async_inference(raw,section,checkpoint,crop_size=crop_size) 
                for section in available_sections
            ], axis=0).transpose(1,0,2,3).astype(np.float32)
        
#            for section in available_sections:
#                predictions.append(
#                    async_inference(
#                        raw,
#                        section,
#                        checkpoint,
#                        crop_size=crop_size,
#                    )
#                )
#
#            stacked_pred = np.stack(predictions, axis=0)
#            stacked_pred = np.transpose(stacked_pred,axes=[1, 0, 2, 3]).astype(np.float32)
#            print(stacked_pred.shape, " !!!!!!!!!!!!!!!!")
#    
#            return [
#                (stacked_pred, {"name": "Stacked 2D prediction"}, "image")
#            ] 
 
        if not hasattr(self, "__inference_widget"):
            self.__inference_widget = inference()
            self.__inference_widget_native = self.__inference_widget.native
        return self.__inference_widget_native

    @property
    def save_widget(self):
        # TODO: block buttons on call. This shouldn't take long, but other operations such
        # as continuing to train should be blocked until this is done.
        def on_return():
            self.set_buttons("paused")

        @magic_factory(call_button="Save")
        def save(path: Path = Path("checkpoint.pt")) -> FunctionWorker[None]:
            @thread_worker(
                connect={"returned": lambda: on_return()},
                progress={"total": 0, "desc": "Saving"},
            )
            def async_save(path: Path = Path("checkpoint.pt")) -> None:
                model, optimizer, scheduler = get_training_state()
                training_stats = get_training_stats()
                torch.save(
                    (
                        model.state_dict(),
                        optimizer.state_dict(),
                        scheduler.state_dict(),
                        training_stats,
                    ),
                    path,
                )

            return async_save(path)

        if not hasattr(self, "__save_widget"):
            self.__save_widget = save()

        return self.__save_widget

    @property
    def load_widget(self):
        # TODO: block buttons on call. This shouldn't take long, but other operations such
        # as continuing to train should be blocked until this is done.
        def on_return():
            self.update_progress_plot()
            self.set_buttons("paused")

        @magic_factory(call_button="Load")
        def load(path: Path = Path("checkpoint.pt")) -> FunctionWorker[None]:
            @thread_worker(
                connect={"returned": on_return},
                progress={"total": 0, "desc": "Saving"},
            )
            def async_load(path: Path = Path("checkpoint.pt")) -> None:
                model, optimizer, scheduler = get_training_state()
                training_stats = get_training_stats()
                state_dicts = torch.load(
                    path,
                )
                model.load_state_dict(state_dicts[0])
                optimizer.load_state_dict(state_dicts[1])
                scheduler.load_state_dict(state_dicts[2])
                training_stats.load(state_dicts[3])

            return async_load(path)

        if not hasattr(self, "__load_widget"):
            self.__load_widget = load()

        return self.__load_widget

    @property
    def training(self) -> bool:
        try:
            return self.__training
        except AttributeError:
            return False

    @training.setter
    def training(self, training: bool):
        self.__training = training
        if training:
            if self.__training_generator is None:
                self.start_training_loop()
            assert self.__training_generator is not None
            self.__training_generator.resume()
            self.set_buttons("training")
        else:
            if self.__training_generator is not None:
                self.__training_generator.send("stop")
                # button state handled by on_return

    def reset_training_state(self, keep_stats=False):
        if self.__training_generator is not None:
            self.__training_generator.quit()
        self.__training_generator = None
        if not keep_stats:
            training_stats = get_training_stats()
            training_stats.reset()
            if self.loss_plot is None:
                self.loss_plot = self.progress_plot.axes.plot(
                    [],
                    [],
                    label="Training Loss",
                )[0]
                self.progress_plot.axes.legend()
                self.progress_plot.axes.set_title("Training Progress")
                self.progress_plot.axes.set_xlabel("Iterations")
                self.progress_plot.axes.set_ylabel("Loss")
            self.update_progress_plot()

    def update_progress_plot(self):
        training_stats = get_training_stats()
        self.loss_plot.set_xdata(training_stats.iterations)
        self.loss_plot.set_ydata(training_stats.losses)
        self.progress_plot.axes.relim()
        self.progress_plot.axes.autoscale_view()
        with contextlib.suppress(np.linalg.LinAlgError):
            # matplotlib seems to throw a LinAlgError on draw sometimes. Not sure
            # why yet. Seems to only happen when initializing models without any
            # layers loaded. No idea whats going wrong.
            # For now just avoid drawing. Seems to work as soon as there is data to plot
            self.progress_plot.draw()

    def start_training_loop(self):
        self.reset_training_state(keep_stats=True)
        training_stats = get_training_stats()

        self.__training_generator = self.train_model(
            self.raw_selector.value,
            self.labels_selector.value,
            iteration=training_stats.iteration,
        )
        self.__training_generator.yielded.connect(self.on_yield)
        self.__training_generator.returned.connect(self.on_return)
        self.__training_generator.start()

    def train(self):
        self.training = not self.training

    def snapshot(self): #TODO: implement!
        self.__training_generator.send("snapshot")
        self.training = True

    def create_train_widget(self, viewer):
        # inputs:
        raw = layer_choice_widget(
            viewer,
            annotation=napari.layers.Image,
            name="raw",
        )
        labels = layer_choice_widget(
            viewer,
            annotation=napari.layers.Labels,
            name="labels",
        )
        train_widget = Container(widgets=[raw,labels])

        return train_widget

    def on_yield(self, step_data):
        iteration, loss, *layers = step_data
        if len(layers) > 0:
            self.add_layers(layers)
        if iteration is not None and loss is not None:
            training_stats = get_training_stats()
            training_stats.iteration = iteration
            training_stats.iterations.append(iteration)
            training_stats.losses.append(loss)
            self.update_progress_plot()

    def on_return(self, weights_path: Path):
        """
        Update model to use provided returned weights
        """
        global _model
        global _optimizer
        global _scheduler
        assert (
            _model is not None
            and _optimizer is not None
            and _scheduler is not None
        )
        model_state_dict, optim_state_dict, scheduler_state_dict = torch.load(
            weights_path
        )
        _model.load_state_dict(model_state_dict)
        _optimizer.load_state_dict(optim_state_dict)
        _scheduler.load_state_dict(scheduler_state_dict)
        self.reset_training_state(keep_stats=True)
        self.set_buttons("paused")

    def add_layers(self, layers):
        viewer_axis_labels = self.viewer.dims.axis_labels

        for data, metadata, layer_type in layers:
            # then try to update the viewer layer with that name.
            name = metadata.pop("name")
            axes = metadata.pop("axes")
            overwrite = metadata.pop("overwrite", False)
            slices = metadata.pop("slices", None)
            shape = metadata.pop("shape", None)

            # handle viewer axes if still default numerics
            # TODO: Support using xarray axis labels as soon as napari does
            if len(set(viewer_axis_labels).intersection(set(axes))) == 0:
                spatial_axes = [
                    axis for axis in axes if axis not in ["batch", "channel"]
                ]
                assert (
                    len(viewer_axis_labels) - len(spatial_axes) <= 1
                ), f"Viewer has axes: {viewer_axis_labels}, but we expect ((channels), {spatial_axes})"
                viewer_axis_labels = (
                    ("channels", *spatial_axes)
                    if len(viewer_axis_labels) > len(spatial_axes)
                    else spatial_axes
                )
                self.viewer.dims.axis_labels = viewer_axis_labels

            batch_dim = axes.index("batch") if "batch" in axes else -1
            assert batch_dim in [
                -1,
                0,
            ], "Batch dim must be first"
            if batch_dim == 0:
                data = data[0]

            if slices is not None and shape is not None:
                # strip channel dimension from slices and shape
                slices = (slice(None, None), *slices[1:])
                shape = (data.shape[0], *shape[1:])

                # create new data array with filled in chunk
                full_data = np.zeros(shape, dtype=data.dtype)
                full_data[slices] = data

            else:
                slices = tuple(slice(None, None) for _ in data.shape)
                full_data = data

            try:
                # add to existing layer
                layer = self.viewer.layers[name]

                if overwrite:
                    layer.data[slices] = data
                    layer.refresh()
                else:
                    # concatenate along batch dimension
                    layer.data = np.concatenate(
                        [
                            layer.data.reshape(-1, *full_data.shape),
                            full_data.reshape(-1, *full_data.shape).astype(
                                layer.data.dtype
                            ),
                        ],
                        axis=0,
                    )
                # make first dimension "batch" if it isn't
                if not overwrite and viewer_axis_labels[0] != "batch":
                    viewer_axis_labels = ("batch", *viewer_axis_labels)
                    self.viewer.dims.axis_labels = viewer_axis_labels

            except KeyError:  # layer not in the viewer
                # TODO: Support defining layer axes as soon as napari does
                if layer_type == "image":
                    self.viewer.add_image(full_data, name=name, **metadata)
                elif layer_type == "labels":
                    self.viewer.add_labels(
                        full_data.astype(int), name=name, **metadata
                    )

    @thread_worker
    def train_model(
        self,
        raw,
        labels,
        iteration=0,
    ):
        train_config = get_train_config()
        # Turn layer into dataset:
        train_dataset = NapariDataset2D(
            raw,
            labels,
#            crop_size=train_config.crop_size,
            voxel_size=train_config.voxel_size,
            control_point_spacing=train_config.control_point_spacing,
            control_point_jitter=train_config.control_point_jitter,
            min_masked=train_config.min_masked,
        )
        model, optimizer, scheduler = get_training_state(train_dataset)

        # TODO: How to display profiling stats
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model = model.to(device)
        model.train()

        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=train_config.batch_size,
            drop_last=True,
            num_workers=train_config.num_workers,
            pin_memory=True,
        )

        # set loss
        criterion = WeightedMSELoss()

        def train_iteration(
            batch,
            model,
            criterion,
            optimizer,
        ):
            prediction = model(batch[0].to(device))#.cuda())
            target = batch[1].to(device)#.cuda()
            weights = batch[2].to(device)#.cuda()
            loss = criterion(prediction, target, weights)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss.item(), prediction

        mode = yield (None, None)
        # call `train_iteration`
        for iteration, batch in tqdm(
            zip(
                range(iteration, train_config.max_iterations),
                train_dataloader,
            )
        ):
            train_loss, prediction = train_iteration(
                [x.float() for x in batch],
                model=model,
                criterion=criterion,
                optimizer=optimizer,
            )
            scheduler.step()

            if mode is None:
                mode = yield (
                    iteration,
                    train_loss,
                )

            elif mode == "stop":
                checkpoint = Path(f"/tmp/checkpoints/{iteration}.pt")
                if not checkpoint.parent.exists():
                    checkpoint.parent.mkdir(parents=True)
                torch.save(
                    (
                        model.state_dict(),
                        optimizer.state_dict(),
                        scheduler.state_dict(),
                    ),
                    checkpoint,
                )
                return checkpoint
