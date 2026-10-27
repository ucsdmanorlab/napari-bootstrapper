<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ucsdmanorlab/napari-bootstrapper/main/src/napari_bootstrapper/resources/logo-dark.svg">
  <img alt="Bootstrapper" src="https://raw.githubusercontent.com/ucsdmanorlab/napari-bootstrapper/main/src/napari_bootstrapper/resources/logo.svg" width="252">
</picture>

# napari-bootstrapper

[![License BSD-3](https://img.shields.io/pypi/l/napari-bootstrapper.svg?color=green)](https://github.com/ucsdmanorlab/napari-bootstrapper/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-bootstrapper.svg?color=green)](https://pypi.org/project/napari-bootstrapper)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-bootstrapper.svg?color=green)](https://python.org)
[![tests](https://github.com/ucsdmanorlab/napari-bootstrapper/workflows/tests/badge.svg)](https://github.com/ucsdmanorlab/napari-bootstrapper/actions)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-bootstrapper)](https://napari-hub.org/plugins/napari-bootstrapper)

Dense 3D segmentation from sparse 2D labels, in napari.
You label a few objects on a few sections and train a 2D model on them, and a pretrained 3D model turns its predictions into a 3D segmentation.
The method is from [_Sparse Annotation is Sufficient for Bootstrapping Dense Segmentation_](https://www.biorxiv.org/content/10.1101/2024.06.14.599135v2).

https://github.com/user-attachments/assets/c501d37a-762b-422f-93d0-4b2415b8c8b0

## Install

```bash
conda create -n bootstrapper python=3.12 -y
conda activate bootstrapper
pip install napari-bootstrapper pyqt5
```

## Run

```bash
napari -w napari-bootstrapper Bootstrapper
```

To try it on example data, open File > Open Sample and pick `cremi_c` (EM) or `fluo_c2dl_huh7` (2D + time).
Video walkthroughs of both are on YouTube: [CREMI C](https://www.youtube.com/watch?v=n0KkhZ-oBTs) and [Fluo-C2DL-Huh7](https://www.youtube.com/watch?v=vThjwJR_RNg).

## Use

1. Data: choose the image and the labels, then click Make mask. Draw labels with the napari brush or with [micro-sam](https://github.com/computational-cell-analytics/micro-sam).
2. 2D model: choose a task and click Train. The tasks predict affinities (`2d_affs`), local shape descriptors (`2d_lsd`), or both (`2d_mtlsd`).
3. 3D model: click Download to get the pretrained model for that task.
4. Segmentation: click Start.

Each Advanced Config button opens the settings for its step.
Models and checkpoints are saved in `~/.napari_bootstrapper`.

To fix the result, open the Proofreading widget (`napari -w napari-bootstrapper Proofreading`).
It can merge, split, delete, dilate, erode, open, close and fill holes, on one section or in 3D, and it can filter objects by size.
Undo reverts the last edit.

For large volumes, 3D models and block-wise processing, use the [Bootstrapper CLI](https://github.com/ucsdmanorlab/bootstrapper).

## Develop

```bash
pip install -e ".[testing]"
QT_QPA_PLATFORM=offscreen pytest -m "not network"
```

Leave out `-m "not network"` to also download and test the published checkpoints.
Report bugs on the [issue tracker](https://github.com/ucsdmanorlab/napari-bootstrapper/issues).

## Cite

```bibtex
@article {Thiyagarajan2024.06.14.599135,
	author = {Thiyagarajan, Vijay Venu and Sheridan, Arlo and Harris, Kristen M. and Manor, Uri},
	title = {Sparse Annotation is Sufficient for Bootstrapping Dense Segmentation},
	year = {2024},
	doi = {10.1101/2024.06.14.599135},
	URL = {https://www.biorxiv.org/content/10.1101/2024.06.14.599135v2},
}
```

## Acknowledgements

The plugin structure comes from [napari-cellulus](https://github.com/funkelab/napari-cellulus), and the Proofreading widget is modeled on [empanada-napari](https://github.com/volume-em/empanada-napari).
The edits use [fastmorph](https://github.com/seung-lab/fastmorph), [fastremap](https://github.com/seung-lab/fastremap) and [cc3d](https://github.com/seung-lab/connected-components-3d).

Funded by the Chan Zuckerberg Initiative Imaging Scientist Award ([10.37921/694870itnyzk](https://doi.org/10.37921/694870itnyzk)) from the Chan Zuckerberg Initiative DAF, an advised fund of Silicon Valley Community Foundation (funder DOI 10.13039/100014989), and by the NSF NeuroNex Technology Hub Award (1707356) and NeuroNex2 Award (2014862).

![image](https://github.com/ucsdmanorlab/bootstrapper/assets/64760651/4b4a6029-e1ba-42bb-ab8b-d9357cc46239)
