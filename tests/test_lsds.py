import gunpowder as gp
import numpy as np
import pytest

from napari_bootstrapper.gp.add_2d_lsds import Add2DLSDs
from napari_bootstrapper.gp.add_obfuscated_2d_lsds import AddObfuscated2DLSDs
from napari_bootstrapper.gp.np_source import NpArraySource


@pytest.mark.parametrize("node", [Add2DLSDs, AddObfuscated2DLSDs])
def test_lsds_match_bootstrapper_on_odd_context_and_sparse_labels(
    node, monkeypatch
):
    monkeypatch.setattr(np.random, "random", lambda: 0.0)
    labels = np.zeros((2, 17, 19), dtype=np.uint32)
    labels[:, 2:12, 2:9] = 1
    labels[:, 7:16, 10:17] = 2
    labels[1, 9, 9] = 3
    segmentation = gp.ArrayKey("LABELS")
    descriptor = gp.ArrayKey("LSDS")
    weights = gp.ArrayKey("WEIGHTS")
    pipeline = NpArraySource(
        labels,
        gp.ArraySpec(roi=gp.Roi((0, 0, 0), (6, 34, 38)), voxel_size=(3, 2, 2)),
        segmentation,
    ) + node(
        segmentation,
        descriptor,
        lsds_mask=weights,
        sigma=(0, 4, 4),
        downsample=2,
    )
    roi = gp.Roi((3, 12, 12), (3, 10, 14))
    request = gp.BatchRequest()
    for key in (segmentation, descriptor, weights):
        request[key] = gp.ArraySpec(roi=roi)
    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    result = batch[descriptor].data
    assert result.shape == (6, 1, 5, 7)
    assert result.dtype == np.float32
    assert batch[descriptor].spec.roi == roi
    bootstrapper_1_0_add_2d_lsds_values = np.array(
        [
            [0.5, 0.44237125, 0.92431164, 0.73377895, 0.5, 0.92881966],
            [
                0.75970936,
                0.75970936,
                0.44205284,
                0.44205475,
                0.50000215,
                0.48933542,
            ],
            [0, 0, 0.0000625, 0.0000625, 0.5, 1],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    actual = np.array(
        [result[:, 0, y, x] for y, x in ((0, 0), (2, 5), (3, 3), (0, 3))]
    )
    np.testing.assert_allclose(actual, bootstrapper_1_0_add_2d_lsds_values, atol=1e-6)
    np.testing.assert_array_equal(
        batch[weights].data[0, 0], labels[1, 6:11, 6:13] != 0
    )
