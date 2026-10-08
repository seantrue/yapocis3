import os
import sys
from pathlib import Path

import numpy as np
import pytest


# Ensure we can import the external package "vimp" from yapocis_external
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT.parent / "yapocis_external"
if str(EXTERNAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_ROOT))


from vimp.common import (  # noqa: E402  # isort: skip
    get_project_root,
    load_config,
    load_npz,
    resolve_param,
    save_npz,
    show_array,
)


def test_load_config_returns_section_when_exists(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
        edges:
          method: zc
          threshold: 0.5
        other:
          foo: bar
        """
    )

    data = load_config(cfg, "edges")
    assert data == {"method": "zc", "threshold": 0.5}


def test_load_config_missing_or_empty_returns_empty(tmp_path: Path):
    # Missing file
    assert load_config(tmp_path / "missing.yaml", "edges") == {}

    # Empty file
    cfg = tmp_path / "empty.yaml"
    cfg.write_text("")
    assert load_config(cfg, "edges") == {}


def test_resolve_param_precedence():
    config = {"k": 2}
    # CLI value wins
    assert resolve_param(10, config, "k", 0) == 10
    # Config when CLI is None
    assert resolve_param(None, config, "k", 0) == 2
    # Default when not in config and CLI is None
    assert resolve_param(None, config, "missing", 7) == 7


def test_show_array_saves_grayscale_with_normalize_and_invert(tmp_path: Path):
    arr = np.array([[0.0, 0.5], [0.5, 1.0]], dtype=np.float32)
    out = show_array(arr, name="gray_inv", queue_dir=tmp_path, normalize=True, invert=True)
    # TODO: show_array should return the output path
    assert out.exists()
    # Verify it saved a PNG file
    assert out.suffix == ".png"


def test_show_array_rgb_normalization(tmp_path: Path):
    # Values > 1 trigger normalization path (division by 255 when normalize=True)
    rgb = np.array(
        [[[0, 128, 255], [10, 20, 30]]], dtype=np.float32
    )  # shape (1,2,3)
    # TODO: show_array should return the output path
    out = show_array(rgb, name="rgb_norm", queue_dir=tmp_path, normalize=True)
    assert out.exists()


def test_show_array_rgba_passthrough_unit_range(tmp_path: Path):
    # Values <= 1.0 should pass through when normalize=True
    rgba = np.random.rand(2, 3, 4).astype(np.float32)
    out = show_array(rgba, name="rgba", queue_dir=tmp_path, normalize=True)
    # TODO: show_array should return the output path
    assert out.exists()


def test_show_array_unsupported_channels_and_dims(tmp_path: Path):
    with pytest.raises(ValueError):
        show_array(np.zeros((2, 2, 5), dtype=np.float32), name="badc", queue_dir=tmp_path)
    with pytest.raises(ValueError):
        show_array(np.zeros((1,), dtype=np.float32), name="baddim", queue_dir=tmp_path)


def test_show_array_default_queue_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # With no queue_dir, yapocis saves to its VIEWING_QUEUE (fixed at import);
    # point that at a temp dir so the test doesn't write into the real queue.
    import yapocis.utils.basic as basic

    queue = tmp_path / "queue"
    queue.mkdir()
    monkeypatch.setattr(basic, "VIEWING_QUEUE", str(queue))
    # Non-zero: yapocis show_array rejects empty pictures (checkzero)
    arr = np.array([[0.0, 0.5], [0.5, 1.0]], dtype=np.float32)
    out = show_array(arr, name="default_queue")
    assert out.exists()
    assert out.parent == queue


def test_save_and_load_npz_and_metadata_yaml(tmp_path: Path):
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.array([1, 2, 3], dtype=np.int16)
    out = tmp_path / "data.npz"

    save_npz(out, a=a, b=b)

    # Main file exists
    assert out.exists()
    # Metadata YAML exists and contains expected keys
    yml = out.with_suffix(".yaml")
    assert yml.exists()
    meta = yml.read_text()
    assert "arrays:" in meta
    assert "a:" in meta and "b:" in meta
    assert "shape:" in meta and "dtype:" in meta

    # Loading back yields identical arrays
    loaded = load_npz(out)
    assert set(loaded.keys()) == {"a", "b"}
    assert np.array_equal(loaded["a"], a)
    assert np.array_equal(loaded["b"], b)


def test_get_project_root_env_and_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Env var case
    env_root = tmp_path / "env_root"
    env_root.mkdir()
    monkeypatch.setenv("YAPOCIS_PROJECT", str(env_root))
    assert get_project_root() == env_root

    # No env var -> current working directory
    monkeypatch.delenv("YAPOCIS_PROJECT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert get_project_root() == tmp_path
