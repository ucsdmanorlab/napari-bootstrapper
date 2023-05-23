# napari-bootstrapper

[![License BSD-3](https://img.shields.io/pypi/l/napari-bootstrapper.svg?color=green)](https://github.com/yajivunev/napari-bootstrapper/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-bootstrapper.svg?color=green)](https://pypi.org/project/napari-bootstrapper)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-bootstrapper.svg?color=green)](https://python.org)
[![tests](https://github.com/yajivunev/napari-bootstrapper/workflows/tests/badge.svg)](https://github.com/yajivunev/napari-bootstrapper/actions)
[![codecov](https://codecov.io/gh/yajivunev/napari-bootstrapper/branch/main/graph/badge.svg)](https://codecov.io/gh/yajivunev/napari-bootstrapper)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-bootstrapper)](https://napari-hub.org/plugins/napari-bootstrapper)

A plugin to quickly generate ground truth with sparse labels

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `napari-bootstrapper` in a conda environment by following these commands:

    conda create -n napari python=3.10
    conda activate napari
    conda install pytorch pytorch-cuda=11.8 "numpy<=1.23.5" boost -c pytorch -c nvidia
        
    pip install cython zarr matplotlib mahotas
    pip install git+https://github.com/funkelab/funlib.geometry.git
    pip install git+https://github.com/funkelab/funlib.persistence.git
    pip install git+https://github.com/funkelab/daisy.git
    pip install git+https://github.com/funkey/gunpowder.git
    pip install git+https://github.com/funkelab/funlib.math.git
    pip install git+https://github.com/funkelab/funlib.evaluate.git
    pip install git+https://github.com/funkelab/funlib.learn.torch.git
    pip install git+https://github.com/htem/waterz@hms
    pip install git+https://github.com/funkelab/funlib.segment.git
    pip install git+https://github.com/funkelab/lsd.git
    pip install neuroglancer
    pip install git+https://github.com/funkelab/funlib.show.neuroglancer.git
    pip install tensorboard tensorboardx
    pip install jsmin
    pip install magic-class
    pip install git+https://github.com/yajivunev/autoseg
    pip install "napari[all]"
    pip install git+https://github.com/salkmanorlab/napari-bootstrapper


## Usage

1. Open napari with `napari` and load in your data.
2. Create a new Labels Layer napari.
    * Paint some labels on a single section (or a few) using napari's brush and fill tools. 
    * Focus on critical areas of the image (ambiguous and obvious examples of boundaries and not boundaries).
3. **Save Your Data** as a zarr.
    * Open `Plugins` -> `napari-bootstrapper`. Open `Bootstrapper`.
    * Click on `Save Data`.
    * Set the filepath of the zarr to be create. 
    * Set the voxel size (or resolution) in world units (nanometers/pixel)
4. Train model 1
    * Make sure voxel size is same as what you saved. If your saved data has voxel size `50, 8, 8`, then set it here as `8, 8`
    * `Run`.
5. Train model 2
    * Make sure voxel size is same as what you saved. If your saved data has voxel size `50,8,8`, then set it here as `50, 8, 8`
    * `Run`.
5. Get a coffee. 
6. `Run Inference`.
    * Make sure all the dataset names, model checkpoint paths, and voxel size are all correct.
    * `Run`
7. `Watershed`
8. `Segment`
9. Update resulting segmentation using other widgets in i`Plugins` -> `napari-bootstrapper`. You can create a Points layer in napari and place points in 3D to:
    * Keep selected labels
    * Delete selected labels
    * Merge selected labels
    * Split selected label between two point
    * Morph selected labels
        * Dilation, Erosion 
        * Morphological opening, closing
        * Remove small objects
10. **Save Your Data**. Use it to train again!
    * Refine, Rinse, Repeat
    * Make many 3D segmentations to feed to the same or a full 3D model.
    * You are now bootstrapping 3D ground truth segmentation.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-bootstrapper" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
