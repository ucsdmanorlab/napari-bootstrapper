from .models import lsd_outpainting, lsd_to_affs
from .segment import watershed_from_affinities, get_segmentation

from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate

import napari

import numpy as np
import os
import time
import zarr
import re

from magicclass import magicclass, wraps
from napari.layers import Layer, Image, Labels, Points
from napari.qt.threading import thread_worker#, FunctionWorker
from napari.types import LabelsData, ImageData
from pathlib import Path
from skimage.measure import label
from typing import List, Literal


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def _pad_data(in_data, target_shape, offset, resolution):

    # todo: make this nicer - seems like napari doesn't handle offsets so
    # we need to pad with zeros instead
    offset = [int(i / j) for i, j in zip(offset, resolution)]

    if len(in_data.shape) == 4:
        padded = np.zeros(shape=(in_data.shape[0],) + tuple(target_shape)).astype(in_data.dtype)
    else:
        padded = np.zeros(tuple(target_shape)).astype(in_data.dtype)

    if len(in_data.shape) == 4:
        padded[
            :,
            offset[0] : offset[0] + in_data.shape[1],
            offset[1] : offset[1] + in_data.shape[2],
            offset[2] : offset[2] + in_data.shape[3],
        ] = in_data
    else:
        padded[
            offset[0] : offset[0] + in_data.shape[0],
            offset[1] : offset[1] + in_data.shape[1],
            offset[2] : offset[2] + in_data.shape[2],
        ] = in_data

    return padded


@magicclass
class Bootstrapper:
    @magicclass
    class Relabel:
        def Relabel(self, labels: LabelsData) -> LabelsData:
            relabelled = label(labels, connectivity=1)
            return relabelled
   
#    @magicclass
#    class TranslateLayer:
#        def Translate_Layer(
#                self, 
#                image: ImageData,
#                labels: LabelsData,
#                translation: List[int] = [0] * 3) -> None:
#
#            # temporary handle type for gp
#            image = np.asarray(image,dtype=np.uint8)
#            labels = np.asarray(labels,dtype=np.uint64)
#
#            print(f"Image shape: {image.shape}")
#            print(f"Labels shape: {labels.shape}")
#
#            # pad or crop labels array so shapes match
#            shape = tuple(image.shape)
#            
#            if len(labels.shape) == 2:
#                labels = np.expand_dims(labels, axis=0)
#
#            translated_labels = np.zeros(shape, dtype=np.uint64)
#            
#            slices = np.s_[
#                    translation[0]:translation[0]+labels.shape[0],
#                    translation[1]:translation[1]+labels.shape[1],
#                    translation[2]:translation[2]+labels.shape[2]]
#
#            translated_labels[slices] = labels
#
#            #labels = _pad_or_crop_array(shape,labels)
#            assert image.shape == translated_labels.shape, "unequal shapes not implemented"
#
#            return translated_labels

#        def _pad_or_crop_array(self, shape, B):
#                
#                A_shape = shape
#                B_shape = B.shape
#                
#                if A_shape == B_shape:
#                    return B  # No padding or cropping required
#                
#                # Calculate the required padding or cropping
#                pad_width = [(0, max(0, A_shape[i] - B_shape[i])) for i in range(len(A_shape))]
#                crop_width = [(0, max(0, B_shape[i] - A_shape[i])) for i in range(len(A_shape))]
#                
#                # Pad or crop the array accordingly
#                if np.sum(crop_width) > 0:
#                    slices = tuple(slice(crop_width[i][0], B_shape[i] - crop_width[i][1]) for i in range(len(A_shape)))
#                    B = B[slices]
#                elif np.sum(pad_width) > 0:
#                    B = np.pad(B, pad_width, mode='constant')
#                
#                return B


    @magicclass
    class SaveData:
        def Save_Data(
            self,
            image: ImageData,
            labels: LabelsData,
            image_name: str = "image",
            labels_name: str = "labels",
            save_container: Path = "data.zarr",
            offset: List[int] = [0] * 3,
            resolution: List[int] = [40, 8, 8],
        ) -> None:

            f = zarr.open(save_container,"a")

            # temporary handle type for gp
            image = np.asarray(image,dtype=np.uint8)
            labels = np.asarray(labels,dtype=np.uint64)

            # pad or crop labels array so shapes match
            shape = tuple(image.shape)
            assert image.shape == labels.shape, "unequal shapes not implemented"

            # create unlabelled mask
            unlabelled = (labels > 0).astype(np.uint8)
            unlabelled_name = "unlabelled"

            if len(shape) == 3:
                # write 3d datasets

                print("Writing 3d datasets...")
                for ds_name, data in [
                    (f"volumes/{image_name}", image),
                    (f"volumes/{labels_name}", labels),
                    (f"volumes/{unlabelled_name}", unlabelled),
                ]:
                    f[ds_name] = data
                    f[ds_name].attrs["offset"] = offset
                    f[ds_name].attrs["resolution"] = resolution

            print("Writing 2d datasets...")
            # write 2d datasets
            for ds_name, data in [
                (image_name, image),
                (labels_name, labels),
                (unlabelled_name, unlabelled),
            ]:
        
                if len(shape) == 2:
                    # add z dim
                    data = np.expand_dims(data, axis=0)

                for i, section in enumerate(data):
                    section_number = int(offset[0]/resolution[0] + i)

                    if np.any(section):
                        f[f"{ds_name}/{section_number}"] = section
                        f[f"{ds_name}/{section_number}"].attrs["offset"] = offset[1:]
                        f[f"{ds_name}/{section_number}"].attrs["resolution"] = resolution[1:]

            print("Done!")
            return None
            

    @magicclass
    class TrainModel1:
        def Train_Model_1(
            self,
            zarr_container: Path = "data.zarr",
            image_dataset: str = "image",
            labels_dataset: str = "labels",
            unlabelled_dataset: str = "unlabelled",
            iters: int = 5001,
            vs: List[int] = [8, 8],
            min_masked: float = 0.1,
            batch_size: int = 5,
            num_workers: int = 10,
            save_every: int = 5000,
            save_name: str = "lsd_outpainting",
        ):

            self._run_model_1(
                zarr_container,
                image_dataset,
                labels_dataset,
                unlabelled_dataset,
                iters,
                vs,
                min_masked,
                batch_size,
                num_workers,
                save_every,
                save_name,
            ).start()

        @thread_worker
        def _run_model_1(
            self, zarr_container, image_dataset, labels_dataset, unlabelled_dataset, iters, vs, min_masked, batch_size, num_workers, save_every, save_name
        ):

            lsd_outpainting.train(
                zarr_container,
                image_dataset,
                labels_dataset,
                unlabelled_dataset,
                iters,
                vs,
                min_masked,
                save_every,
                batch_size,
                num_workers,
                save_name)

    @magicclass
    class TrainModel2:
        def Train_Model_2(
            self,
            iters: int = 5001,
            vs: List[int] = [40, 8, 8],
            #batch_size: int = 5,
            num_workers: int = 10,
            save_every: int = 1000,
            save_name: str = "lsd_to_affs",
        ):

            self._run_model_2(
                iters,
                vs,
                save_every,
                save_name,
                num_workers
            ).start()
        
        @thread_worker
        def _run_model_2(self, iters, vs, save_every, save_name, num_workers):

            lsd_to_affs.train(iters, vs, save_every, save_name, num_workers)

    @magicclass
    class RunInference:
        def Run_Inference(
            self,
            zarr_container: Path = "data.zarr",
            image_dataset: str = "image",
            lsds_dataset: str = "lsds",
            affs_dataset: str = "affs",
            model_1_checkpoint: Path = "lsd_outpainting_checkpoint_5000",
            model_2_checkpoint: Path = "lsd_to_affs_checkpoint_5000",
            voxel_size: List[int] = [40, 8, 8],
            grow: bool = False,
        ) -> ImageData:

            self.zarr_container = str(zarr_container)
            self.image_dataset = image_dataset
            self.lsds_dataset = lsds_dataset
            self.affs_dataset = affs_dataset
            self.model_1_checkpoint = str(model_1_checkpoint)
            self.model_2_checkpoint = str(model_2_checkpoint)
            self.voxel_size = voxel_size
            self.grow = grow

            worker = self._run_prediction()
            worker.returned.connect(self._on_return)
            worker.start()

        def _on_return(self, value):
            # callback function to add returned data to the viewer
            viewer = self.parent_viewer
            viewer.add_image(value[0], name=self.lsds_dataset)
            viewer.add_image(value[1], name=self.affs_dataset)
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

            for f,ds in image_sources:

                lsds, lsds_roi = lsd_outpainting.predict(
                    f, ds, self.model_1_checkpoint, self.voxel_size, self.grow
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
                    self.lsds_dataset,
                    lsds_roi,
                    voxel_size,
                    dtype=np.uint8,
                    delete=True,
                    num_channels=6)

            out_lsds[lsds_roi] = full_lsds
            print("Lsds inference complete!")

            # network 2: lsds -> affs
            affs, affs_roi = lsd_to_affs.predict(
                self.zarr_container,
                self.lsds_dataset,
                self.zarr_container,
                self.affs_dataset,
                self.model_2_checkpoint,
                self.voxel_size,
                grow=self.grow
            )

            # napari doesn't handle offsets?? pad with zeros...
            full_lsds = _pad_data(
                in_data=full_lsds,
                target_shape=image_shape,
                offset=lsds_offset,
                resolution=self.voxel_size,
            )

            affs = _pad_data(
                in_data=affs,
                target_shape=image_shape,
                offset=affs_roi.get_begin(),
                resolution=self.voxel_size,
            )

            # c,z,y,x
            # think napari wants channels last to render rgb...
            full_lsds = full_lsds[0:3].transpose((1, 2, 3, 0)).astype(np.uint8)
            affs = affs.transpose((1, 2, 3, 0)).astype(np.uint8)

            # hacky way for knowing how much to pad segmentation by later, todo:
            # handle this correctly
            out_file[self.affs_dataset].attrs["raw_shape"] = image_shape

            return [full_lsds, affs]

    @magicclass
    class Watershed:
        def Watershed(
                self, 
                zarr_container: Path = "data.zarr",
                affs_dataset: str = "affs",
                fragments_dataset: str = "frags") -> LabelsData:

            # hacky way of getting the base image shape, im sure there is a
            # better way to access this info via the viewer (without having to
            # pass as an argument...
            zarr_container = str(zarr_container)

            f = zarr.open(zarr_container, "a")
            print("Loading affinities...")
            affinities = f[affs_dataset][:]
            affinities = (affinities / np.max(affinities)).astype(np.float32)

            raw_shape = f[affs_dataset].attrs["raw_shape"]
            resolution = f[affs_dataset].attrs["resolution"]
            offset = f[affs_dataset].attrs["offset"]
            roi = Roi(Coordinate(offset),Coordinate(affinities.shape[1:])*Coordinate(resolution))

            out_frags = prepare_ds(
                    zarr_container,
                    fragments_dataset,
                    roi,
                    resolution,
                    dtype=np.uint64,
                    delete=True)
            
            print("Making fragments...")
            fragments = watershed_from_affinities(affinities)[0]

            out_frags[roi] = fragments
            
            fragments = _pad_data(
                fragments, raw_shape, offset, resolution
            )

            print("Done!")
            return fragments

    @magicclass
    class Segment:
        def Segment(
                self, 
                zarr_container: Path = "data.zarr",
                affs_dataset: str = "affs",
                fragments_dataset: str = "frags",
                threshold: float = 0.5,
                merge_function: Literal["mean", "hist_quant_50", "hist_quant_75"] = "mean") -> LabelsData:

            f = zarr.open(zarr_container, "r")
            
            # hacky way of getting the base image shape, im sure there is a
            # better way to access this info via the viewer (without having to
            # pass as an argument...
            raw_shape = f[affs_dataset].attrs["raw_shape"]
            resolution = f[affs_dataset].attrs["resolution"]
            offset = f[affs_dataset].attrs["offset"]

            print("Loading affinities and fragments...")
            affs = f[affs_dataset][:]
            frags = f[fragments_dataset][:]

            affs = (affs / np.max(affs)).astype(np.float32)

            print("Performing hierarchical agglomeration...")
            seg = get_segmentation(affs, frags, threshold=threshold, merge_function=merge_function)
            seg = _pad_data(seg, raw_shape, offset, resolution)

            print("Done!")
            return seg

if __name__ == "__main__":
    napari.run()
