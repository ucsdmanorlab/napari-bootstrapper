"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory

from skimage.measure import label

if TYPE_CHECKING:
    import napari

@magic_factory
def relabel_cc(labels: "napari.types.LabelsData") -> "napari.types.LabelsData":
    relabelled = label(labels, connectivity=1)
    return relabelled
