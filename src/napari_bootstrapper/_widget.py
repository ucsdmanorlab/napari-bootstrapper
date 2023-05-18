import napari
import numpy as np
import os
import time
import zarr
import re

from magicclass import magicclass
from magicclass.ext.napari import to_napari
from napari.layers import Image, Points
from napari.qt.threading import thread_worker, FunctionWorker
from napari.types import LabelsData, ImageData
from pathlib import Path
from skimage.measure import label
from typing import List

from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate

# functions for new two net approach
from gp.train_net_1 import lsd_outpainting_pipeline
from gp.train_net_2 import fake_lsds_pipeline
from gp.predict_net_1 import lsd_outpaint_predict
from gp.predict_net_2 import fake_lsds_predict

from segment import watershed_from_affinities, get_segmentation


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def _pad_data(in_data, target_shape, offset, resolution):

    # todo: make this nicer - seems like napari doesn't handle offsets so
    # we need to pad with zeros instead
    offset = [int(i / j) for i, j in zip(offset, resolution)]

    padded = np.zeros(shape=(in_data.shape[0],) + tuple(target_shape))

    padded[
        :,
        offset[0] : offset[0] + in_data.shape[1],
        offset[1] : offset[1] + in_data.shape[2],
        offset[2] : offset[2] + in_data.shape[3],
    ] = in_data

    return padded



@to_napari
@magicclass
class Bootstrapper:
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
            save_container: Path = "training_data/test.zarr",
            offset: List[int] = [0] * 3,
            resolution: List[int] = [1] * 3,
        ) -> None:

            f = zarr.open(save_container,"a")

            # temporary handle type for gp
            image = np.asarray(image,dtype=np.uint8)
            labels = np.asarray(labels,dtype=np.uint64)

            assert image.shape == labels.shape, "unequal shapes not implemented"
            shape = tuple(image.shape)

            # create unlabelled mask
            unlabelled = (labels > 0).astype(np.uint8)

            # write 3d datasets
            for ds_name, data in [
                ("volumes/image", image),
                ("volumes/labels", labels),
                ("volumes/unlabelled", unlabelled),
            ]:
                f[ds_name] = data
                f[ds_name].attrs["offset"] = offset
                f[ds_name].attrs["resolution"] = resolution

            # write 2d datasets
            for ds_name, data in [
                ("image", image),
                ("labels", labels),
                ("unlabelled", unlabelled),
            ]:
                for i, section in enumerate(data):
                    section_number = int(offset[0]/resolution[0] + i)

                    if np.any(section):
                        f[f"{ds_name}/{section_number}"] = section
                        f[f"{ds_name}/{section_number}"].attrs["offset"] = offset[1:]
                        f[f"{ds_name}/{section_number}"].attrs["resolution"] = resolution[1:]

            return None


    @magicclass
    class TrainModel:
        def Train_Model(
            self,
            zarr_container: Path = "training_data/test.zarr",
            image_dataset: str = "image",
            labels_dataset: str = "labels",
            unlabelled_dataset: str = "unlabelled",
            model_1_iters: int = 3000,
            model_2_iters: int = 1000,
            model_1_vs: List[int] = [8, 8],
            model_2_vs: List[int] = [40, 8, 8],
            min_masked: float = 0.1,
            batch_size: int = 5,
            #pre_cache,
            model_1_save_every: int = 1000,
            model_2_save_every: int = 1000,
            model_1_save_name: str = "training_data/lsd_outpainting",
            model_2_save_name: str = "training_data/fake_lsds",
        ):

            # todo: handle parameters cleanly - maybe change pipelines to
            # classes and have a config class? quick and dirty for now, here are
            # some example parameters:

            os.makedirs("checkpoints", exist_ok=True)

            self._run_model_1(
                zarr_container,
                image_dataset,
                labels_dataset,
                unlabelled_dataset,
                model_1_iters,
                model_1_vs,
                min_masked,
                batch_size,
                model_1_save_every,
                model_1_save_name,
            ).start()

            self._run_model_2(
                model_2_iters,
                model_2_vs,
                model_2_save_every,
                model_2_save_name,
            ).start()

        @thread_worker
        def _run_model_1(
            self, zarr_file, raw_ds, labels_ds, unlabelled_ds, iters, vs, min_masked, batch_size, save_every, save_name
        ):

            lsd_outpainting_pipeline(
                zarr_file, raw_ds, labels_ds, unlabelled_ds, iters, vs, min_masked, batch_size, save_every, save_name
            )

        @thread_worker
        def _run_model_2(self, iters, vs, save_every, save_name):

            fake_lsds_pipeline(iters, vs, save_every, save_name)

    @magicclass
    class RunInference:
        def Run_Inference(
            self,
            zarr_container: Path = "training_data/test.zarr",
            image_dataset: str = "image",
            model_1_checkpoint: int = 3000,
            model_2_checkpoint: int = 1000,
            voxel_size: List[int] = [40, 8, 8],
        ) -> ImageData:

            self.zarr_container = str(zarr_container)
            self.image_dataset = image_dataset
            self.model_1_checkpoint = str(model_1_checkpoint)
            self.model_2_checkpoint = str(model_2_checkpoint)
            self.voxel_size = voxel_size

            worker = self._run_prediction()
            worker.returned.connect(self._on_return)
            worker.start()

        def _on_return(self, value):
            # callback function to add returned data to the viewer
            viewer = self.parent_viewer
            viewer.add_image(value[0], name="lsds")
            viewer.add_image(value[1], name="affs")
            # ADD TRANSLATE. napari viewer can do offset

        @thread_worker
        def _run_prediction(self):

            # open zarr container
            out_file = zarr.open(self.zarr_container, "r+")
            voxel_size = Coordinate(out_file[f"volumes/{self.image_dataset}"].attrs["resolution"])
            image_shape = out_file[f"volumes/{self.image_dataset}"].shape

            # get available sections
            available_sections = natural_sort([x for x in os.listdir(os.path.join(self.zarr_container,self.image_dataset)) if '.' not in x])
            print(f"Doing net 1 inference on sections: {available_sections}")

            image_sources = [(self.zarr_container,f"{self.image_dataset}/{section}") for section in available_sections]

            # network 1: raw -> lsds
            full_lsds = []

            for z in available_sections:
                raw_dataset = f"self.image_dataset/{z}"

                lsds, lsds_roi = lsd_outpaint_predict(
                    self.zarr_container, raw_dataset, self.model_1_checkpoint, self.voxel_size
                )

                full_lsds.append(lsds)

            # currently z,c,y,x
            # transpose to c,z,y,x for second net
            full_lsds = (
                np.array(full_lsds).transpose((1, 0, 2, 3)).astype(np.float32)
            )

            # offset should be 3d for padding
            lsds_offset = [int(available_sections[0])*voxel_size[0]] + list(lsds_roi.offset)
            lsds_roi = Roi(Coordinate(lsds_offset),Coordinate(full_lsds.shape[1:])*voxel_size)

            # write lsds to zarr
            out_lsds = prepare_ds(
                    self.zarr_container,
                    "lsds",
                    Roi(lsds_offset,lsds_roi.shape),
                    voxel_size,
                    dtype=np.uint8,
                    delete=True,
                    num_channels=6)

            out_lsds[lsds_roi] = full_lsds

            # hacky way for knowing how much to pad segmentation by later, todo:
            # handle this correctly
            out_file["lsds"].attrs["raw_shape"] = image_shape

            # network 2: lsds -> affs
            affs, affs_offset = fake_lsds_predict(
                self.zarr_container,
                "lsds",
                self.zarr_container,
                "affs",
                self.model_2_checkpoint,
                self.voxel_size,
                grow=True
            )

            # napari doesn't handle offsets?? pad with zeros...
            full_lsds = _pad_data(
                in_data=full_lsds,
                target_shape=image_shape,
                offset=lsds_offset,
                resolution=resolution,
            )

            affs = _pad_data(
                in_data=affs,
                target_shape=image_shape,
                offset=affs_offset,
                resolution=resolution,
            )

            # c,z,y,x
            # think napari wants channels last to render rgb...
            full_lsds = full_lsds[0:3].transpose((1, 2, 3, 0)).astype(np.uint8)
            affs = affs.transpose((1, 2, 3, 0)).astype(np.uint8)

            return [full_lsds, affs]

    @magicclass
    class Watershed:
        def Watershed(
                self, 
                zarr_container: Path = "training_data/test.zarr",
                affs_dataset: str = "affs") -> LabelsData:

            # hacky way of getting the base image shape, im sure there is a
            # better way to access this info via the viewer (without having to
            # pass as an argument...

            f = zarr.open(zarr_container, "a")
            affinities = f[affs_dataset][:]
            affinities = (affinities / np.max(affinities)).astype(np.float32)

            raw_shape = f["lsds"].attrs["raw_shape"]
            resolution = f["lsds"].attrs["resolution"]
            offset = f[affs_dataset].attrs["offset"]
            roi = Roi(Coordinate(offset),Coordinate(affinities.shape[1:])*voxel_size)

            out_frags = prepare_ds(
                    zarr_container,
                    "fragments",
                    roi,
                    resolution,
                    dtype=np.uint64,
                    delete=True)
            
            fragments = watershed_from_affinities(affinities)[0]

            out_frags[roi] = fragments
            
            fragments = _pad_data(
                fragments, raw_shape, offset, resolution
            )

            return fragments

    @magicclass
    class Segment:
        def Segment(
                self, 
                zarr_container: Path = "training_data/test.zarr",
                threshold: float = 0.5) -> LabelsData:


            # hacky way of getting the base image shape, im sure there is a
            # better way to access this info via the viewer (without having to
            # pass as an argument...
            raw_shape = zarr.open(f_name)["lsds"].attrs["raw_shape"]
            resolution = zarr.open(f_name)["lsds"].attrs["resolution"]
            offset = zarr.open(f_name)["affs"].attrs["offset"]

            f = zarr.open(zarr_container, "r")
            affs = f["affs"][:]
            frags = f["frags"][:]

            affs = (affs / np.max(affs)).astype(np.float32)
            seg = get_segmentation(affs, frags, threshold=threshold)

            seg = _pad_data(seg, raw_shape, offset, resolution)

            return seg

napari.run()
