# napari-bootstrapper

[![License BSD-3](https://img.shields.io/pypi/l/napari-bootstrapper.svg?color=green)](https://github.com/ucsdmanorlab/napari-bootstrapper/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-bootstrapper.svg?color=green)](https://pypi.org/project/napari-bootstrapper)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-bootstrapper.svg?color=green)](https://python.org)
[![tests](https://github.com/ucsdmanorlab/napari-bootstrapper/workflows/tests/badge.svg)](https://github.com/ucsdmanorlab/napari-bootstrapper/actions)
[![codecov](https://codecov.io/gh/ucsdmanorlab/napari-bootstrapper/branch/main/graph/badge.svg)](https://codecov.io/gh/ucsdmanorlab/napari-bootstrapper)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-bootstrapper)](https://napari-hub.org/plugins/napari-bootstrapper)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

- [Introduction](#introduction)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Citation](#citation)
- [Issues](#issues)
- [Funding](#funding)

## Introduction

`napari-bootstrapper` is a tool to quickly generate dense 3D labels using sparse 2D labels within napari.

Dense 3D segmentations are generated using the 2D->3D method described in the preprint titled [_Sparse Annotation is Sufficient for Bootstrapping Dense Segmentation_](https://www.biorxiv.org/content/10.1101/2024.06.14.599135v2). In the preprint, we show sparse 2D annotations made in ~10 minutes on a single section can generate dense 3D segmentations that are reasonably good starting points for refining or bootstrapping.

This plugin is limited to the 2D->3D method and is intended for small volumes that can fit in memory. For more complex bootstrapping workflows, dedicated 3D models, and block-wise processing of large volumes, we recommend using the [_Bootstrapper_](https://github.com/ucsdmanorlab/bootstrapper) CLI tool.

## Installation

We recommend installing `napari-bootstrapper` via conda and [pip]:

1. Create a new environment called `napari-bootstrapper`:

```bash
conda create -n napari-bootstrapper -c conda-forge python==3.11 napari pyqt
```

2. Activate the newly-created environment:

```
conda activate napari-bootstrapper
```

3. You can install `napari-bootstrapper` via [pip]:

```bash
pip install napari-bootstrapper
```
   - Or you can install the latest development version from github:

```bash
pip install git+https://github.com/ucsdmanorlab/napari-bootstrapper.git
```


## Getting Started
Run the following in your terminal:
```bash
conda activate napari-bootstrapper
napari
```

## Citation

If you find Bootstrapper useful in your research, please consider citing our **[preprint](https://www.biorxiv.org/content/10.1101/2024.06.14.599135v1)**:
```
@article {Thiyagarajan2024.06.14.599135,
	author = {Thiyagarajan, Vijay Venu and Sheridan, Arlo and Harris, Kristen M. and Manor, Uri},
	title = {Sparse Annotation is Sufficient for Bootstrapping Dense Segmentation},
	year = {2024},
	doi = {10.1101/2024.06.14.599135},
	URL = {https://www.biorxiv.org/content/10.1101/2024.06.14.599135v2},
}
```


## Issues

If you encounter any problems, please [file an issue](https://github.com/ucsdmanorlab/napari-bootstrapper/issues) along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/


## Funding
Chan-Zuckerberg Imaging Scientist Award DOI https://doi.org/10.37921/694870itnyzk from the Chan Zuckerberg Initiative DAF, an advised fund of Silicon Valley Community Foundation (funder DOI 10.13039/100014989).

NSF NeuroNex Technology Hub Award (1707356), NSF NeuroNex2 Award (2014862)

![image](https://github.com/ucsdmanorlab/bootstrapper/assets/64760651/4b4a6029-e1ba-42bb-ab8b-d9357cc46239)
