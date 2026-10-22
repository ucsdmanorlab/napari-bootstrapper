import random
import subprocess
import sys
import textwrap

import numpy as np
import torch

from napari_bootstrapper.datasets import stack_samples
from napari_bootstrapper.datasets.napari_2d_dataset import Napari2DDataset


def test_sparse_2d_targets_leave_background_at_zero(
    sparse_layers, small_2d_config
):
    dataset = Napari2DDataset(
        *sparse_layers,
        model_type="2d_lsd",
        adj_slices=3,
        input_shape=[24, 24],
        output_shape=[16, 16],
        **small_2d_config["task"],
    )
    random.seed(0)
    np.random.seed(0)
    samples = iter(dataset)
    try:
        for _ in range(8):
            _, targets, weights = next(samples)
            assert np.any(weights > 0)
            assert np.any(weights == 0)
            np.testing.assert_array_equal(targets[weights == 0], 0)
    finally:
        samples.close()


def test_2d_dataset_feeds_a_spawned_worker(sparse_layers, small_2d_config):
    dataset = Napari2DDataset(
        *sparse_layers,
        model_type="2d_mtlsd",
        adj_slices=3,
        input_shape=[24, 24],
        output_shape=[16, 16],
        **small_2d_config["task"],
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        multiprocessing_context="spawn",
    )
    raw, *targets = next(iter(loader))
    assert raw.shape == (1, 3, 24, 24)
    assert len(targets) == 4


def test_numpy_batches_match_default_collate(sparse_layers, small_2d_config):
    dataset = Napari2DDataset(
        *sparse_layers,
        model_type="2d_mtlsd",
        adj_slices=3,
        input_shape=[24, 24],
        output_shape=[16, 16],
        **small_2d_config["task"],
    )
    samples = iter(dataset)
    try:
        batch = [next(samples) for _ in range(2)]
    finally:
        samples.close()
    expected = torch.utils.data.default_collate(batch)
    stacked = [torch.as_tensor(array) for array in stack_samples(batch)]
    assert len(stacked) == len(expected)
    for actual, want in zip(stacked, expected, strict=True):
        assert actual.dtype == want.dtype
        assert torch.equal(actual, want)


def test_small_synthetic_source_does_not_corrupt_memory(tmp_path):
    code = textwrap.dedent("""
        import gunpowder as gp
        import numpy as np
        from napari_bootstrapper.gp.create_labels import CreateLabels

        labels = gp.ArrayKey("LABELS")
        pipeline = CreateLabels(labels, shape=(5, 24, 24), voxel_size=(1, 1, 1))
        with gp.build(pipeline):
            for seed in range(32):
                request = gp.BatchRequest(random_seed=seed)
                request.add(labels, (5, 24, 24))
                array = pipeline.request_batch(request)[labels].data
                assert array.shape == (5, 24, 24)
                assert array.dtype == np.uint32
                assert array.max() > 0
    """)
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
