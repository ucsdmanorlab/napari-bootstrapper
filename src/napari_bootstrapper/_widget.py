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

from autoseg.segment.watershed import watershed_from_affinities
from autoseg.segment.hierarchical import run as get_segmentation

import autoseg


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
    class TrainModel1:
        def Train_Model_1(
            self,
            zarr_container: Path = "training_data/test.zarr",
            image_dataset: str = "image",
            labels_dataset: str = "labels",
            unlabelled_dataset: str = "unlabelled",
            model_1_iters: int = 3000,
            model_1_vs: List[int] = [8, 8],
            min_masked: float = 0.1,
            #batch_size: int = 5,
            pre_cache: List[int] = [10,20],
            model_1_save_every: int = 1000,
            model_1_save_name: str = "lsd_outpainting",
        ):
            self._run_model_1(
                model_1_iters,
                model_1_save_every,
                zarr_container,
                image_dataset,
                labels_dataset,
                unlabelled_dataset,
                min_masked,
                pre_cache,
                model_1_save_name
            ).start()
            
        @thread_worker
        def _run_model_1(
                self,
                iters,
                save_every,
                zarr_file,
                raw_ds,
                labels_ds,
                unlabelled_ds,
                min_masked,
                pre_cache,
                checkpoint_basename
        ):

            model_1 = autoseg.model_paths["membrane"]["lsd_2d_unet"]

            # get available sections
            available_sections = [x for x in os.listdir(os.path.join(zarr_file,labels_ds)) if '.' not in x]
            print(f"Training on sections: {available_sections}")
            
            zarr_file = str(zarr_file)

            sources = [{
                "raw": [zarr_file,f"{raw_ds}/{section}"],
                "labels": [zarr_file, f"{labels_ds}/{section}"],
                "unlabelled": [zarr_file, f"{unlabelled_ds}/{section}"]
            } for section in available_sections]

            autoseg.train(
                    iters,
                    save_every,
                    sources,
                    model_1,
                    min_masked=min_masked,
                    pre_cache=pre_cache,
                    checkpoint_basename=checkpoint_basename)


    @magicclass
    class TrainModel2:
        def Train_Model_2(
            self,
            model_2_iters: int = 1000,
            model_2_vs: List[int] = [50, 8, 8],
            pre_cache: List[int] = [10,20],
            model_2_save_every: int = 1000,
            model_2_save_name: str = "fake_lsds",
        ):
            
            self._run_model_2(
                model_2_iters,
                model_2_save_every,
                pre_cache,
                model_2_save_name
            ).start()

        @thread_worker
        def _run_model_2(
                self,
                iters,
                save_every,
                pre_cache,
                checkpoint_basename
        ):

            model_2 = autoseg.model_paths["membrane"]["2d_lsds_to_3d_affs_unet"]

            autoseg.train(
                    iters,
                    save_every,
                    None,
                    model_2,
                    pre_cache=pre_cache,
                    checkpoint_basename=checkpoint_basename,
                    snapshots_dir="snapshots")

    @magicclass
    class RunInference1:
        def Run_Inference_1(
            self,
            zarr_container: Path = "training_data/test.zarr",
            image_dataset: str = "image",
            model_1_checkpoint: Path = "training_data/lsd_outpainting_checkpoint_2000",
        ) -> ImageData:

            self.zarr_container = str(zarr_container)
            self.image_dataset = image_dataset
            self.model_1_checkpoint = str(model_1_checkpoint)

            worker = self._run_prediction()
            worker.returned.connect(self._on_return)
            worker.start()

        def _on_return(self, value):
            # callback function to add returned data to the viewer
            viewer = self.parent_viewer
            viewer.add_image(value, name="lsds")
            # ADD TRANSLATE 

        @thread_worker
        def _run_prediction(self):

            # open zarr container
            out_file = zarr.open(self.zarr_container, "a")
            voxel_size = Coordinate(out_file[f"volumes/{self.image_dataset}"].attrs["resolution"])
            image_shape = out_file[f"volumes/{self.image_dataset}"].shape

            # network 1: raw -> lsds
            model_1 = autoseg.model_paths["membrane"]["lsd_2d_unet"]

            # get available sections
            available_sections = natural_sort([x for x in os.listdir(os.path.join(self.zarr_container,self.image_dataset)) if '.' not in x])
            print(f"Doing net 1 inference on sections: {available_sections}")

            image_sources = [(self.zarr_container,f"{self.image_dataset}/{section}") for section in available_sections]

            full_lsds = []
            for source in image_sources:

                lsds, _, lsds_roi = autoseg.predict(
                    [source],
                    self.zarr_container,
                    self.model_1_checkpoint,
                    model_1,
                    increase=(128,128),
                    write=None,
                    return_arrays=True)

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

            # napari doesn't handle offsets?? pad with zeros...
            full_lsds = _pad_data(
                in_data=full_lsds,
                target_shape=image_shape,
                offset=lsds_offset,
                resolution=voxel_size,
            )

            # c,z,y,x
            # think napari wants channels last to render rgb...
            full_lsds = full_lsds[0:3].transpose((1, 2, 3, 0))

            return full_lsds.astype(np.uint8)


    @magicclass
    class RunInference2:
        def Run_Inference_2(
            self,
            zarr_container: Path = "training_data/test.zarr",
            lsds_dataset: str = "lsds",
            model_2_checkpoint: Path = "training_data/affs_net_checkpoint_5000",
        ) -> ImageData:

            self.zarr_container = str(zarr_container)
            self.lsds_dataset = lsds_dataset
            self.model_2_checkpoint = str(model_2_checkpoint)

            worker = self._run_prediction()
            worker.returned.connect(self._on_return)
            worker.start()

        def _on_return(self, value):
            # callback function to add returned data to the viewer
            viewer = self.parent_viewer
            viewer.add_image(value, name="affs")
            # ADD TRANSLATE 

        @thread_worker
        def _run_prediction(self):

            # open zarr container
            out_file = zarr.open(self.zarr_container, "a")
            voxel_size = Coordinate(out_file[f"{self.lsds_dataset}"].attrs["resolution"])
            image_shape = out_file[f"{self.lsds_dataset}"].attrs["raw_shape"]

            lsds_source = [(self.zarr_container,self.lsds_dataset)]

            # network 2: lsds -> affs
            model_2 = autoseg.model_paths["membrane"]["2d_lsds_to_3d_affs_unet"]
           
            print("Doing net 2 inference")
            affs, _, affs_roi = autoseg.predict(
                lsds_source,
                self.zarr_container,
                self.model_2_checkpoint,
                model_2,
                increase=(16,128,128),
                return_arrays=True)

            # napari doesn't handle offsets?? pad with zeros...
            affs = _pad_data(
                in_data=affs,
                target_shape=image_shape,
                offset=affs_roi.offset,
                resolution=voxel_size,
            )

            # c,z,y,x
            # think napari wants channels last to render rgb...
            affs = affs.transpose((1, 2, 3, 0))

            return affs.astype(np.uint8)

    @magicclass
    class Watershed:
        def Watershed(self) -> LabelsData:

            # todo: we should handle this via layers rather than always going
            # back and forth between files. also make padding function a util
            # function...

            f_name = "test_prediction.zarr"

            # hacky way of getting the base image shape, im sure there is a
            # better way to access this info via the viewer (without having to
            # pass as an argument...
            raw_shape = zarr.open(f_name)["lsds"].attrs["raw_shape"]
            resolution = zarr.open(f_name)["lsds"].attrs["resolution"]
            offset = zarr.open(f_name)["affs"].attrs["offset"]

            f = zarr.open(f_name, "a")
            affinities = f["affs"][:]

            fragments = watershed_from_affinities(affinities)[0]

            f["frags"] = fragments
            f["frags"].attrs["offset"] = offset
            f["frags"].attrs["resolution"] = resolution

            fragments = _pad_data(
                fragments, raw_shape, offset, resolution
            )

            return fragments

    @magicclass
    class Segment:
        def Segment(self, threshold: float = 0.5) -> LabelsData:

            f_name = "test_prediction.zarr"

            # hacky way of getting the base image shape, im sure there is a
            # better way to access this info via the viewer (without having to
            # pass as an argument...
            raw_shape = zarr.open(f_name)["lsds"].attrs["raw_shape"]
            resolution = zarr.open(f_name)["lsds"].attrs["resolution"]
            offset = zarr.open(f_name)["affs"].attrs["offset"]

            f = zarr.open(f_name, "a")
            affs = f["affs"][:]
            frags = f["frags"][:]

            seg = get_segmentation(affs, frags, threshold=threshold)

            seg = _pad_data(seg, raw_shape, offset, resolution)

            return seg


napari.run()
