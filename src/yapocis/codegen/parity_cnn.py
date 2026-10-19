#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.
"""
Library parity generators for the CNN layer (`cnn` interface, VGG16, ResNet50).

``cnn``: every `cnn` kernel called once on small seeded inputs. Each record lists
the user arguments in call order (``{"array": key}``, ``{"scalar": v}`` or
``{"out": key, "shape": [...]}`` for `out` parameters, which are passed as zero
arrays) and the returned arrays in return order (``ret_<i>`` keys). A consumer
replays the record through its untyped call path and compares.

``models``: VGG16 and ResNet50 end to end on a packaged image, only where the
model weights are installed (``YAPOCIS_MODELS``). Stores the decoded image, the
preprocessed tensor, class probabilities, top-5 and the embedding so a consumer
can check its preprocessing (loosely: resampling differs between libraries) and
its network (tightly, from the stored preprocessed tensor).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np

from yapocis.core import load_program
from yapocis.core.interface import get_interface, load_interface

from .parity_lib import generator

SEED = 20261019

# kernel -> (array shapes by parameter name, scalar values by parameter name)
C, H, W = 4, 8, 6
OUT = 5
CASES: dict[str, tuple[dict[str, tuple[int, ...]], dict[str, Any]]] = {
    "conv2d_3x3": (
        {
            "input": (C, H, W),
            "weights": (OUT, C, 3, 3),
            "bias": (OUT,),
            "output": (OUT, H, W),
        },
        {"output_channels": OUT},
    ),
    "conv2d_3x3_optimized": (
        {
            "input": (C, H, W),
            "weights": (OUT, C, 3, 3),
            "bias": (OUT,),
            "output": (OUT, H, W),
        },
        {"output_channels": OUT},
    ),
    "conv2d_1x1": (
        {
            "input": (C, H, W),
            "weights": (OUT, C, 1, 1),
            "bias": (OUT,),
            "output": (OUT, H, W),
        },
        {"output_channels": OUT},
    ),
    "depthwise_conv2d_3x3": (
        {
            "input": (C, H, W),
            "weights": (C, 1, 3, 3),
            "bias": (C,),
            "output": (C, H, W),
        },
        {},
    ),
    "conv2d_7x7_stride2": (
        {
            "input": (3, 16, 12),
            "weights": (OUT, 3, 7, 7),
            "bias": (OUT,),
            "output": (OUT, 8, 6),
        },
        {"output_channels": OUT},
    ),
    "conv2d_3x3_stride2": (
        {
            "input": (C, 16, 12),
            "weights": (OUT, C, 3, 3),
            "bias": (OUT,),
            "output": (OUT, 8, 6),
        },
        {"output_channels": OUT},
    ),
    "conv2d_1x1_stride2": (
        {
            "input": (C, 16, 12),
            "weights": (OUT, C, 1, 1),
            "bias": (OUT,),
            "output": (OUT, 8, 6),
        },
        {"output_channels": OUT},
    ),
    "relu_activation": ({"input": (C, H, W)}, {}),
    "gelu": ({"input": (C, H, W)}, {}),
    "softmax": ({"input": (37,)}, {}),
    "sigmoid": ({"input": (C, H, W)}, {}),
    "max_pool_2x2": ({"input": (C, 16, 12), "output": (C, 8, 6)}, {}),
    "max_pool_3x3_stride2": ({"input": (C, 16, 12), "output": (C, 8, 6)}, {}),
    "global_avg_pool": ({"input": (C, H, W), "output": (C,)}, {}),
    "global_max_pool": ({"input": (C, H, W), "output": (C,)}, {}),
    "batch_norm": (
        {
            "input": (C, H, W),
            "gamma": (C,),
            "beta": (C,),
            "mean": (C,),
            "variance": (C,),
        },
        {"epsilon": 1e-5},
    ),
    "layer_norm": (
        {"input": (C, H, W), "gamma": (C,), "beta": (C,), "output": (C, H, W)},
        {"epsilon": 1e-5},
    ),
    "layer_norm_1d": (
        {"input": (33,), "gamma": (33,), "beta": (33,), "output": (33,)},
        {"epsilon": 1e-5},
    ),
    "fully_connected": (
        {"input": (24,), "weights": (10, 24), "bias": (10,), "output": (10,)},
        {},
    ),
    "matrix_multiply_simd": ({"A": (6, 8), "B": (8, 5), "bias": (5,), "C": (6, 5)}, {}),
    "add_tensors": ({"input1": (C, H, W), "input2": (C, H, W)}, {}),
    "dropout": (
        {"input": (C, H, W), "output": (C, H, W)},
        {"rate": 0.5, "seed": 7, "training": 0},
    ),
    "spatial_dropout": (
        {"input": (C, H, W), "output": (C, H, W)},
        {"rate": 0.5, "seed": 7, "training": 0},
    ),
    "upsample_2x": ({"input": (C, H, W), "output": (C, 2 * H, 2 * W)}, {}),
}

IMAGE = "golden_retriever.jpg"


def _as_tuple(result: Any) -> tuple[np.ndarray, ...]:
    if result is None:
        return ()
    return result if isinstance(result, tuple) else (result,)


def _run(callable_: Callable[..., Any], args: list[Any]) -> tuple[np.ndarray, ...]:
    copies = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    return _as_tuple(callable_(*copies))


@generator("cnn")
def cnn_kernels(
    out_dir: str, backend: str, log: Callable[..., None]
) -> list[dict[str, Any]]:
    prog = load_program("cnn", backend=backend)
    spec = get_interface(load_interface("cnn"))
    kernels = {k.name: k for k in spec.kernels}
    cases: list[dict[str, Any]] = []
    for name, (shapes, scalars) in CASES.items():
        kernel = kernels[name]
        rng = np.random.default_rng(SEED)
        args: list[Any] = []
        arg_records: list[dict[str, Any]] = []
        arrays: dict[str, np.ndarray] = {}
        for p in kernel.parameters:
            if p.derived:
                continue
            if p.hint:
                shape = shapes[p.name]
                if p.hint == "out":
                    arrays[f"out_{p.name}"] = np.zeros(shape, dtype=np.float32)
                    args.append(arrays[f"out_{p.name}"])
                    arg_records.append({"out": f"out_{p.name}", "shape": list(shape)})
                else:
                    if p.name == "variance":
                        a = rng.uniform(0.5, 2.0, size=shape).astype(np.float32)
                    else:
                        a = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
                    arrays[f"in_{p.name}"] = a
                    args.append(a)
                    arg_records.append({"array": f"in_{p.name}"})
            else:
                value = scalars[p.name]
                args.append(value)
                arg_records.append({"scalar": value})
        callable_ = prog.callable[name]
        try:
            first = _run(callable_, args)
            second = _run(callable_, args)
        except Exception as e:  # noqa: BLE001
            log(f"cnn.{name}: SKIP {type(e).__name__}: {e}")
            continue
        if len(first) != len(second) or any(
            not np.array_equal(a, b, equal_nan=True)
            for a, b in zip(first, second, strict=True)
        ):
            log(f"cnn.{name}: SKIP not reproducible")
            continue
        if any(np.isnan(o).any() for o in first):
            log(f"cnn.{name}: SKIP NaN output")
            continue
        for i, o in enumerate(first):
            arrays[f"ret_{i}"] = np.ascontiguousarray(o)
        fname = f"cnn.{name}.npz"
        np.savez_compressed(os.path.join(out_dir, fname), **arrays)  # type: ignore[arg-type]
        cases.append(
            {
                "name": name,
                "kernel": name,
                "file": fname,
                "args": arg_records,
                "outputs": [f"ret_{i}" for i in range(len(first))],
            }
        )
    prog.purge()
    return cases


def _load_image() -> np.ndarray:
    from PIL import Image

    from yapocis.images import find_image

    path = find_image(IMAGE)
    return np.array(Image.open(path).convert("RGB"))


@generator("models")
def models(
    out_dir: str, backend: str, log: Callable[..., None]
) -> list[dict[str, Any]]:
    from yapocis.models import get_weights_path

    cases: list[dict[str, Any]] = []
    image = _load_image()

    weights = get_weights_path("resnet50")
    if weights is not None:
        from yapocis.lib.resnet50 import ResNet50

        net = ResNet50(str(weights))
        pre = net.preprocess_image(image)
        probs = net.forward(pre)
        emb = net.forward(pre, stop_at="pool")
        top = np.argsort(probs)[::-1][:5]
        fname = "models.resnet50.npz"
        np.savez_compressed(
            os.path.join(out_dir, fname),
            image=image,
            preprocessed=pre,
            probs=probs,
            embedding=emb,
            top5_indices=top.astype(np.int64),
            top5_probs=probs[top],
        )
        cases.append(
            {
                "name": "resnet50",
                "model": "resnet50",
                "file": fname,
                "image": IMAGE,
                "top1": int(top[0]),
            }
        )
        log(f"models.resnet50: top-1 {int(top[0])} p={float(probs[top[0]]):.4f}")
    else:
        log("models.resnet50: SKIP no weights (set YAPOCIS_MODELS)")

    weights = get_weights_path("vgg16")
    if weights is not None:
        from yapocis.lib.vgg16 import VGG16

        vgg = VGG16(str(weights))
        pre = vgg.preprocess_image(image)
        probs = vgg.forward(pre)
        fc6 = vgg.forward(pre, stop_at="fc6")
        top = np.argsort(probs)[::-1][:5]
        fname = "models.vgg16.npz"
        np.savez_compressed(
            os.path.join(out_dir, fname),
            image=image,
            preprocessed=pre,
            probs=probs,
            embedding=fc6,
            top5_indices=top.astype(np.int64),
            top5_probs=probs[top],
        )
        cases.append(
            {
                "name": "vgg16",
                "model": "vgg16",
                "file": fname,
                "image": IMAGE,
                "top1": int(top[0]),
            }
        )
        log(f"models.vgg16: top-1 {int(top[0])} p={float(probs[top[0]]):.4f}")
    else:
        log("models.vgg16: SKIP no weights (set YAPOCIS_MODELS)")
    return cases
