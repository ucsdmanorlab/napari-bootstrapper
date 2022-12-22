import napari
import numpy as np
import os
import time
import zarr

from magicclass import magicclass
from magicclass.ext.napari import to_napari
from napari.layers import Image, Points
from napari.qt.threading import thread_worker, FunctionWorker
from napari.types import LabelsData, ImageData
from pathlib import Path
from skimage.measure import label
from typing import List

# functions for new two net approach
from train_net_1 import lsd_outpainting_pipeline
from train_net_2 import fake_lsds_pipeline
from predict_net_1 import lsd_outpaint_predict
from predict_net_2 import fake_lsds_predict
from segment import watershed_from_affinities, get_segmentation


@to_napari
@magicclass
class Bootstrapper:
    @magicclass
    class LoadData:
        def Load_Data(self):
            # todo: add logic for loading various datatypes, eg a single
            # function that can load image, image stack, zarr, etc
            pass

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

            f = zarr.open(save_container, "a")

            # temporary handle type for gp
            image = image.astype(np.uint8)

            # create unlabelled mask
            unlabelled = (labels > 0).astype(np.uint8)

            # just so we have source for net 2
            zeros = np.zeros(shape=(10, 600, 600)).astype(np.uint8)

            for ds_name, data in [
                ("image", image),
                ("labels", labels),
                ("unlabelled", unlabelled),
                ("zeros", zeros),
            ]:

                if ds_name == "labels":
                    data = data.astype(np.uint64)

                f[ds_name] = data

                # todo: clean up
                if ds_name == "zeros":
                    f[ds_name].attrs["offset"] = offset
                    f[ds_name].attrs["resolution"] = resolution
                else:
                    f[ds_name].attrs["offset"] = offset[1:]
                    f[ds_name].attrs["resolution"] = resolution[1:]

            return None

    @magicclass
    class TrainModel:
        def Train_Model(
            self,
            model_1_iters: int = 3000,
            model_2_iters: int = 1000,
            model_1_vs: List[int] = [8, 8],
            model_2_vs: List[int] = [40, 8, 8],
            min_masked: float = 0.1,
            batch_size: int = 5,
            model_1_save_every: int = 1000,
            model_2_save_every: int = 1000,
            model_1_save_name: str = "checkpoints/lsd_outpainting",
            model_2_save_name: str = "checkpoints/fake_lsds",
        ):

            # todo: handle parameters cleanly - maybe change pipelines to
            # classes and have a config class? quick and dirty for now, here are
            # some example parameters:

            os.makedirs("checkpoints", exist_ok=True)

            self._run_model_1(
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
            self, iters, vs, min_masked, batch_size, save_every, save_name
        ):

            lsd_outpainting_pipeline(
                iters, vs, min_masked, batch_size, save_every, save_name
            )

        @thread_worker
        def _run_model_2(self, iters, vs, save_every, save_name):

            fake_lsds_pipeline(iters, vs, save_every, save_name)

    @magicclass
    class RunInference:
        def Run_Inference(
            self,
            image: ImageData,
            model_1_checkpoint: int = 3000,
            model_2_checkpoint: int = 1000,
            grow: bool = False,
        ) -> ImageData:

            # another option for param passing
            self.image = image
            self.model_1_checkpoint = model_1_checkpoint
            self.model_2_checkpoint = model_2_checkpoint
            self.grow = grow

            worker = self._run_prediction()
            worker.returned.connect(self._on_return)
            worker.start()

        def _on_return(self, value):
            # callback function to add returned data to the viewer
            viewer = self.parent_viewer
            viewer.add_image(value[0], name="lsds")
            viewer.add_image(value[1], name="affs")

        def _pad_data(self, in_data, target_data, offset, resolution):

            # todo: make this nicer - seems like napari doesn't handle offsets so
            # we need to pad with zeros instead
            offset = [int(i / j) for i, j in zip(offset, resolution)]

            padded = np.zeros(shape=(in_data.shape[0],) + target_data.shape)

            padded[
                :,
                offset[0] : offset[0] + in_data.shape[1],
                offset[1] : offset[1] + in_data.shape[2],
                offset[2] : offset[2] + in_data.shape[3],
            ] = in_data

            return padded

        @thread_worker
        def _run_prediction(self):

            raw_file = "test_data.zarr"
            out = zarr.open(raw_file, "a")

            # todo: make nicer, this is because gunpowder expects zarr source
            # so we write images from layer to zarr then read them back for
            # prediction
            for z in range(self.image.shape[0]):
                print(f"saving section {z}")

                out[f"full_raw_2d/{z}"] = np.array(self.image[z]).astype(
                    np.uint8
                )
                out[f"full_raw_2d/{z}"].attrs["offset"] = [0, 0]
                out[f"full_raw_2d/{z}"].attrs["resolution"] = [8, 8]

            # network 1: raw -> lsds
            full_lsds = []

            for z in range(self.image.shape[0]):
                raw_dataset = f"full_raw_2d/{z}"

                lsds, lsds_offset = lsd_outpaint_predict(
                    raw_file, raw_dataset, self.model_1_checkpoint
                )

                full_lsds.append(lsds)

            # offset should be 3d for padding
            lsds_offset = [0] + list(lsds_offset)

            # currently z,c,y,x
            # transpose to c,z,y,x for second net
            full_lsds = (
                np.array(full_lsds).transpose((1, 0, 2, 3)).astype(np.float32)
            )

            save_name = "test_prediction.zarr"
            out_file = zarr.open(save_name, "a")

            resolution = [40, 8, 8]

            out_file["lsds"] = full_lsds
            out_file["lsds"].attrs["offset"] = lsds_offset
            out_file["lsds"].attrs["resolution"] = resolution

            # hacky way for knowing how much to pad segmentation by later, todo:
            # handle this correctly
            out_file["lsds"].attrs["raw_shape"] = self.image.shape

            # network 2: lsds -> affs
            affs, affs_offset = fake_lsds_predict(
                save_name,
                "lsds",
                save_name,
                "affs",
                self.model_2_checkpoint,
                self.grow,
            )

            # napari doesn't handle offsets?? pad with zeros...
            full_lsds = self._pad_data(
                in_data=full_lsds,
                target_data=self.image,
                offset=lsds_offset,
                resolution=resolution,
            )

            affs = self._pad_data(
                in_data=affs,
                target_data=self.image,
                offset=affs_offset,
                resolution=resolution,
            )

            # c,z,y,x
            # think napari wants channels last to render rgb...
            full_lsds = full_lsds[0:3].transpose((1, 2, 3, 0))
            affs = affs.transpose((1, 2, 3, 0))

            return [full_lsds, affs]

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

            fragments = self._pad_data(
                fragments, raw_shape, offset, resolution
            )

            return fragments

        def _pad_data(self, in_data, target_data_shape, offset, resolution):

            # todo: make this nicer - seems like napari doesn't handle offsets so
            # we need to pad with zeros instead
            offset = [int(i / j) for i, j in zip(offset, resolution)]

            padded = np.zeros(shape=target_data_shape).astype(in_data.dtype)

            padded[
                offset[0] : offset[0] + in_data.shape[0],
                offset[1] : offset[1] + in_data.shape[1],
                offset[2] : offset[2] + in_data.shape[2],
            ] = in_data

            return padded

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

            seg = self._pad_data(seg, raw_shape, offset, resolution)

            return seg

        def _pad_data(self, in_data, target_data_shape, offset, resolution):

            # todo: make this nicer - seems like napari doesn't handle offsets so
            # we need to pad with zeros instead
            offset = [int(i / j) for i, j in zip(offset, resolution)]

            padded = np.zeros(shape=target_data_shape).astype(in_data.dtype)

            padded[
                offset[0] : offset[0] + in_data.shape[0],
                offset[1] : offset[1] + in_data.shape[1],
                offset[2] : offset[2] + in_data.shape[2],
            ] = in_data

            return padded


napari.run()
