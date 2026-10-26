import contextlib
import warnings

import numpy as np
import pytest
import torch
from qtpy.QtWidgets import QCheckBox, QDialog, QLabel

from napari_bootstrapper import widget as widget_module
from napari_bootstrapper.widget import autocast, train_iteration

mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="needs an MPS device"
)


def mps_runs_conv3d():
    if not torch.backends.mps.is_available():
        return False
    try:
        x = torch.zeros((1, 2, 4, 12, 12), device="mps", requires_grad=True)
        torch.nn.Conv3d(2, 3, 3, padding=1).to("mps")(x).sum().backward()
    except RuntimeError:
        return False
    return True


# The virtual GPU on GitHub's macOS runners cannot compile the Metal Conv3d
# kernel, so the 3D case runs only where MPS can run a Conv3d.
mps_conv3d = pytest.mark.skipif(
    not mps_runs_conv3d(), reason="MPS cannot run a Conv3d here"
)


@pytest.mark.parametrize(
    "device,enabled,bf16_supported,capability,expected",
    [
        ("cpu", True, True, (8, 0), False),
        ("mps", False, True, (8, 0), False),
        ("mps", True, True, (8, 0), True),
        ("cuda", True, True, (8, 0), True),
        # A T4 reports bfloat16 support, but only through emulation.
        ("cuda", True, True, (7, 5), False),
        ("cuda", True, False, (8, 0), False),
        ("cuda", False, True, (8, 0), False),
    ],
)
def test_autocast_only_on_gpus_with_bfloat16(
    monkeypatch, device, enabled, bf16_supported, capability, expected
):
    monkeypatch.setattr(
        torch.cuda,
        "is_bf16_supported",
        lambda including_emulation=True: bf16_supported,
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )
    with warnings.catch_warnings():
        # Without a CUDA device, torch warns and turns the cuda context off.
        warnings.simplefilter("ignore")
        context = autocast(torch.device(device), enabled)
    if expected:
        assert isinstance(context, torch.autocast)
        assert context.fast_dtype == torch.bfloat16
    else:
        assert isinstance(context, contextlib.nullcontext)


class TwoHeads(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 3, padding=1)
        self.forward_dtypes = []

    def forward(self, x):
        y = self.conv(x)
        self.forward_dtypes.append(y.dtype)
        return y[:, :1], y[:, 1:]


class TwoInputs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(2, 3, 3, padding=1)
        self.forward_dtypes = []

    def forward(self, x, y):
        out = self.conv(torch.cat([x, y], dim=1))
        self.forward_dtypes.append(out.dtype)
        return out


def mse_2d(pred_a, target_a, weights_a, pred_b, target_b, weights_b):
    return ((pred_a - target_a) ** 2 * weights_a).mean() + (
        (pred_b - target_b) ** 2 * weights_b
    ).mean()


def mse_3d(pred, target, weights):
    return ((pred - target) ** 2 * weights).mean()


def batch_2d(seed):
    g = torch.Generator().manual_seed(seed)
    shape = (2, 1, 16, 16)
    return [torch.rand(shape, generator=g) for _ in range(5)]


def batch_3d(seed):
    g = torch.Generator().manual_seed(seed)
    return [
        torch.rand((1, 1, 4, 12, 12), generator=g),
        torch.rand((1, 1, 4, 12, 12), generator=g),
        torch.rand((1, 3, 4, 12, 12), generator=g),
        torch.rand((1, 3, 4, 12, 12), generator=g),
    ]


def run(model_class, criterion, batch, dimension, device, mixed_precision):
    torch.manual_seed(0)
    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for seed in range(3):
        loss, outputs = train_iteration(
            batch(seed),
            model,
            criterion,
            optimizer,
            torch.device(device),
            dimension,
            mixed_precision=mixed_precision,
        )
        losses.append(loss)
    return model, losses, outputs


@pytest.mark.parametrize(
    "model_class,criterion,batch,dimension",
    [(TwoHeads, mse_2d, batch_2d, "2d"), (TwoInputs, mse_3d, batch_3d, "3d")],
)
def test_mixed_precision_leaves_cpu_training_unchanged(
    model_class, criterion, batch, dimension
):
    model_a, losses_a, _ = run(
        model_class, criterion, batch, dimension, "cpu", False
    )
    model_b, losses_b, _ = run(
        model_class, criterion, batch, dimension, "cpu", True
    )
    assert losses_a == losses_b
    for a, b in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert torch.equal(a, b)
    assert set(model_b.forward_dtypes) == {torch.float32}


@mps
@pytest.mark.parametrize(
    "mixed_precision,forward_dtype",
    [(True, torch.bfloat16), (False, torch.float32)],
)
@pytest.mark.parametrize(
    "model_class,criterion,batch,dimension",
    [
        (TwoHeads, mse_2d, batch_2d, "2d"),
        pytest.param(TwoInputs, mse_3d, batch_3d, "3d", marks=mps_conv3d),
    ],
)
def test_mixed_precision_flag_sets_the_mps_forward_dtype(
    model_class, criterion, batch, dimension, mixed_precision, forward_dtype
):
    model, losses, outputs = run(
        model_class, criterion, batch, dimension, "mps", mixed_precision
    )
    assert set(model.forward_dtypes) == {forward_dtype}
    assert all(np.isfinite(loss) for loss in losses)
    outputs = [outputs] if isinstance(outputs, torch.Tensor) else outputs
    assert all(output.dtype == torch.float32 for output in outputs)
    assert all(p.dtype == torch.float32 for p in model.parameters())


@pytest.mark.parametrize("dimension", ["2d", "3d"])
def test_widget_passes_the_config_flag(
    widget,
    sparse_layers,
    small_2d_config,
    small_3d_config,
    monkeypatch,
    dimension,
):
    flags = []

    def record(*args, mixed_precision, **kwargs):
        flags.append(mixed_precision)
        return train_iteration(
            *args, mixed_precision=mixed_precision, **kwargs
        )

    monkeypatch.setattr(widget_module, "train_iteration", record)
    widget.model_2d_type_selector.setCurrentText("2d_mtlsd")
    widget.model_3d_type_selector.setCurrentText("3d_affs_from_2d_mtlsd")
    widget.model_2d_config = small_2d_config
    widget.model_3d_config = small_3d_config
    widget.train_3d_model_from_scratch_checkbox.setChecked(True)
    assert widget.train(dimension).work() is None
    getattr(widget, f"model_{dimension}_config")["mixed_precision"] = False
    assert widget.train(dimension).work() is None
    assert flags == [True, False]


@pytest.mark.parametrize("config_has_key", [True, False])
@pytest.mark.parametrize("dimension", ["2d", "3d"])
def test_advanced_config_shows_mixed_precision(
    widget, monkeypatch, dimension, config_has_key
):
    if not config_has_key:
        # Configs saved before this setting existed have no key.
        del getattr(widget, f"model_{dimension}_config")["mixed_precision"]
    dialogs = []
    monkeypatch.setattr(
        QDialog, "exec_", lambda dialog: dialogs.append(dialog)
    )
    widget.open_model_options(dimension)
    config = getattr(widget, f"model_{dimension}_config")
    assert config["mixed_precision"] is True
    labels = [
        label
        for label in dialogs[0].findChildren(QLabel)
        if label.text() == "Mixed Precision"
    ]
    assert len(labels) == 1
    boxes = [
        box
        for box in dialogs[0].findChildren(QCheckBox)
        if box.toolTip() == labels[0].toolTip()
    ]
    assert len(boxes) == 1
    assert boxes[0].isChecked()
    assert "bfloat16" in boxes[0].toolTip()
