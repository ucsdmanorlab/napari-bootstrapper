import hashlib
import io
import zipfile

import pytest
import requests
import torch

import napari_bootstrapper.widget as widget_module


@pytest.fixture
def checkpoint_download(monkeypatch):
    model_type = "3d_affs_from_2d_affs"
    checkpoint = b"test checkpoint"
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr(
            f"{model_type}/model_checkpoint_25000.ckpt", checkpoint
        )
    payload = data.getvalue()
    hashes = {
        model_type: {
            "archive": hashlib.sha256(payload).hexdigest(),
            "checkpoint": hashlib.sha256(checkpoint).hexdigest(),
        }
    }
    monkeypatch.setattr(
        widget_module, "PRETRAINED_3D_MODEL_SHA256", hashes, raising=False
    )

    def get(url, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response.url = url
        response.headers["content-length"] = str(len(payload))
        response.raw = io.BytesIO(payload)
        return response

    monkeypatch.setattr(widget_module.requests, "get", get)
    return model_type, checkpoint, hashes


@pytest.mark.parametrize("part", ["archive", "checkpoint"])
def test_download_rejects_wrong_hash(widget, checkpoint_download, part):
    model_type, _, hashes = checkpoint_download
    hashes[model_type][part] = "0" * 64
    widget.model_3d_type_selector.setCurrentText(model_type)
    with pytest.raises(ValueError, match="SHA-256"):
        widget.download_pretrained_model()
    assert not hasattr(widget, "pretrained_3d_model_checkpoint")
    assert not list(widget.tmp_dir.glob("**/*.ckpt"))
    assert not list(widget.tmp_dir.glob("*.zip"))


def test_download_repairs_corrupt_cache_and_reuses_verified_weights(
    widget, checkpoint_download, monkeypatch
):
    model_type, expected, _ = checkpoint_download
    widget.model_3d_type_selector.setCurrentText(model_type)
    widget.download_pretrained_model()
    checkpoint = widget.pretrained_3d_model_checkpoint
    assert checkpoint.read_bytes() == expected
    checkpoint.write_bytes(b"damaged")
    widget.download_pretrained_model()
    assert checkpoint.read_bytes() == expected

    def offline(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(widget_module.requests, "get", offline)
    widget.download_pretrained_model()
    assert widget.pretrained_3d_model_checkpoint.read_bytes() == expected


@pytest.mark.network
@pytest.mark.parametrize(
    "task,channels",
    [
        ("3d_affs_from_2d_affs", (6,)),
        ("3d_affs_from_2d_lsd", (6,)),
        ("3d_affs_from_2d_mtlsd", (6, 6)),
    ],
)
def test_published_checkpoint_predicts_on_cpu(widget, task, channels):
    widget.model_2d_type_selector.setCurrentText("2d_mtlsd")
    widget.model_3d_type_selector.setCurrentText(task)
    widget.device = torch.device("cpu")
    widget.download_pretrained_model()
    widget.reload_model_if_needed("3d", task, sum(channels))
    widget.model_3d.eval()
    inputs = [torch.rand(1, count, 24, 148, 148) for count in channels]
    with torch.no_grad():
        output = widget.model_3d(*inputs)
    assert output.shape == (1, 9, 4, 56, 56)
    assert torch.isfinite(output).all()
    assert torch.std(output.flatten(-3), dim=-1).max() > 0
