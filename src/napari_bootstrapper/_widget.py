import napari
import numpy as np
import zarr
from magicclass import magicclass, magicmenu, magicgui, MagicTemplate
from magicclass.ext.napari import to_napari
from magicgui import magic_factory
from napari.layers import Image, Points
from napari.types import LabelsData, ImageData
from pathlib import Path
from skimage.measure import label
from typing import List

from utils import get_segmentation, watershed_from_lsds

@to_napari
@magicclass
class Bootstrapper:

    @magicclass
    class LoadData:
        def Load_Data(
                self,
                dataset: str = 'pred_lsds',
                channel_start: int = 0,
                channel_end: int = 3) -> ImageData:

            f = zarr.open('gp/test_prediction.zarr')

            # will render as rgb but cause problems when segmenting, fix later
            # lsds = f['pred_lsds'][channel_start:channel_end].T

            lsds = f['pred_lsds'][:]

            return lsds

    @magicclass
    class Relabel:
        def Relabel(self, labels: LabelsData) -> LabelsData:
            relabelled = label(labels, connectivity=1)
            return relabelled

    @magicclass
    class SaveData:
        def Save_Data(
            self,
            image: ImageData,
            labels: LabelsData,
            save_container: Path = 'training_data/test.zarr',
            offset: List[int] = [0]*3,
            resolution: List[int] = [1]*3) -> None:

            f = zarr.open(save_container, 'a')

            unlabelled = (labels > 0).astype(np.uint8)

            for ds_name, data in [
                    ('image', image),
                    ('labels', labels),
                    ('unlabelled', unlabelled)]:

                if ds_name == 'labels':
                    data = data.astype(np.uint64)

                f[ds_name] = data
                f[ds_name].attrs['offset'] = offset
                f[ds_name].attrs['resolution'] = resolution

            return None

    @magicclass
    class TrainModel:
        def Train_Model(self):
            pass

    @magicclass
    class RunInference:
        def Run_Inference(self):
            pass

    @magicclass
    class Segment:
        def Segment(
                self,
                lsds: ImageData) -> LabelsData:

            affs, frags, boundary_mask, boundary_distances = watershed_from_lsds(lsds)

            threshold = 0.1

            seg = get_segmentation(
                affinities=np.expand_dims(affs,axis=1),
                frags=np.expand_dims(frags, axis=0), threshold=threshold)

            return seg

napari.run()
