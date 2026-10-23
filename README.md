# napari-bootstrapper

[![License BSD-3](https://img.shields.io/pypi/l/napari-bootstrapper.svg?color=green)](https://github.com/ucsdmanorlab/napari-bootstrapper/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-bootstrapper.svg?color=green)](https://pypi.org/project/napari-bootstrapper)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-bootstrapper.svg?color=green)](https://python.org)
[![tests](https://github.com/ucsdmanorlab/napari-bootstrapper/workflows/tests/badge.svg)](https://github.com/ucsdmanorlab/napari-bootstrapper/actions)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-bootstrapper)](https://napari-hub.org/plugins/napari-bootstrapper)

This napari plugin generates dense 3D segmentations from sparse 2D labels.
The plugin uses the 2D-to-3D method in [_Sparse Annotation is Sufficient for Bootstrapping Dense Segmentation_](https://www.biorxiv.org/content/10.1101/2024.06.14.599135v2).

For larger volumes, dedicated 3D models, and block-wise processing, see the [Bootstrapper CLI](https://github.com/ucsdmanorlab/bootstrapper).

![cremi30](https://github.com/user-attachments/assets/db0e9ef7-0826-4184-9f00-8203e7bf48ec)

| Dataset | Data type | Video |
|---------|-----------|-------|
| CREMI C | 3D volumetric stack | [![CREMI C](https://img.youtube.com/vi/n0KkhZ-oBTs/0.jpg)](https://www.youtube.com/watch?v=n0KkhZ-oBTs) |
| Fluo-C2DL-Huh7 | 2D + time series | [![Fluo-C2DL-Huh7](https://img.youtube.com/vi/vThjwJR_RNg/0.jpg)](https://www.youtube.com/watch?v=vThjwJR_RNg) |

## Install

Use Python 3.12 or later and scikit-image 0.26 or later.
The plugin requires Zarr format 2 (`zarr<3`). Create a separate environment:

```bash
conda create -n napari-bootstrapper python=3.12
conda activate napari-bootstrapper
python -m pip install "napari-bootstrapper==1.0.0" pyqt5
```

To install the development version instead:

```bash
python -m pip install "napari-bootstrapper @ git+https://github.com/ucsdmanorlab/napari-bootstrapper.git" pyqt5
```

## Launch

```bash
conda activate napari-bootstrapper
napari -w napari-bootstrapper Bootstrapper
```

You can also open the Bootstrapper widget from napari's Plugins menu.
The File > Open Sample menu includes CREMI C (`cremi_c`) and Fluo-C2DL-Huh7 (`fluo_c2dl_huh7`) data, which ship with the plugin and need no download.

## Segmentation

### 1. Data

Load a 3D image, or a 4D image with one channel axis, and select it as the Image layer.
Label objects on one or more sections and select that layer as the Labels layer.
Click Make mask to mark the labelled voxels for training, then select the new mask as the Mask layer.
You can make labels with napari's drawing tools or [micro-sam](https://github.com/computational-cell-analytics/micro-sam).

### 2. 2D model

Choose a task and click Train. The loss plot appears below the buttons while the model trains.
To use a checkpoint from an earlier run, clear Train from scratch and click Load model.

| Task | Output |
|------|--------|
| `2d_affs` | Affinities |
| `2d_lsd` | Local shape descriptors (LSDs) |
| `2d_mtlsd` | Both LSDs and affinities |

The Device list offers CPU, plus CUDA and MPS when they are available.
For CPU training without data-loader processes, set Num Workers to `0` under Training in the 2D model's Advanced Config.

### 3. 3D model

Click Download to get the pretrained 3D model for the selected 2D task, or click Load model to use your own checkpoint.
The 3D model converts 2D predictions into 3D affinities.
To train a 3D model yourself, open Train the 3D model.
The plugin saves downloads and training checkpoints in `~/.napari_bootstrapper`.

### 4. Segmentation

Choose mutex watershed or connected components as the Method, then click Start to predict and segment the volume.
Advanced Config sets the affinity channels, biases and other parameters of the method.
The widget adds the predictions and segmentation as napari layers.

## Proofreading

Open the Proofreading widget from the Plugins menu, or run:

```bash
napari -w napari-bootstrapper Proofreading
```

Choose the labels layer to edit. Select objects with a points layer or with label IDs, then choose an operation and click its button.
The widget can merge, split, delete, dilate, erode, open, close, and fill holes.
These operations change the current section, or the full volume when Apply in 3D is checked.
The Filter operation removes objects by size, by section count, or by how many standard deviations their size lies from the mean object size.
Filter always changes the full volume, so Apply in 3D is disabled for it.
A filter set to Off is skipped. Undo reverts the last operation.
The widget uses [fastmorph](https://github.com/seung-lab/fastmorph), [fastremap](https://github.com/seung-lab/fastremap), and [cc3d](https://github.com/seung-lab/connected-components-3d).

## Tests

From a source checkout, install the test dependencies and run the headless suite:

```bash
python -m pip install ".[testing]"
QT_QPA_PLATFORM=offscreen CUDA_VISIBLE_DEVICES="" python -m pytest
```

The tests build the widget, load the bundled samples, train small CPU models, and run both segmentation methods.
They also download, verify, and run all three published checkpoints.
The tests run the widget on a napari ViewerModel, which has no canvas, so they do not check rendered pixels.
To run without checkpoint downloads, use `python -m pytest -m "not network"`.

## Citation

Please cite the [preprint](https://www.biorxiv.org/content/10.1101/2024.06.14.599135v2) when you use the plugin in research:

```bibtex
@article {Thiyagarajan2024.06.14.599135,
	author = {Thiyagarajan, Vijay Venu and Sheridan, Arlo and Harris, Kristen M. and Manor, Uri},
	title = {Sparse Annotation is Sufficient for Bootstrapping Dense Segmentation},
	year = {2024},
	doi = {10.1101/2024.06.14.599135},
	URL = {https://www.biorxiv.org/content/10.1101/2024.06.14.599135v2},
}
```

## Issues

[File an issue](https://github.com/ucsdmanorlab/napari-bootstrapper/issues) with the error message, plugin version, and steps to reproduce the problem.

## Acknowledgements

- [micro-sam](https://github.com/computational-cell-analytics/micro-sam) provides SAM annotation tools.
- [empanada-napari](https://github.com/volume-em/empanada-napari) inspired the proofreading widget.
- [napari-cellulus](https://github.com/funkelab/napari-cellulus) provided the original plugin structure.

## Funding

This project received funding from the Chan-Zuckerberg Imaging Scientist Award (https://doi.org/10.37921/694870itnyzk).
The award comes from the Chan Zuckerberg Initiative DAF, an advised fund of Silicon Valley Community Foundation.
The funder DOI is 10.13039/100014989.

The project also received funding from the NSF NeuroNex Technology Hub Award (1707356) and the NSF NeuroNex2 Award (2014862).

![image](https://github.com/ucsdmanorlab/bootstrapper/assets/64760651/4b4a6029-e1ba-42bb-ab8b-d9357cc46239)
