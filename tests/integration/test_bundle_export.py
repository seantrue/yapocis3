"""
Kernel bundle export (yapocis-tools bundle) and derivation fixtures.

These tests render interfaces and kernel templates but never touch a device, so
they run everywhere, including on machines with no GPU backend installed.
"""

from __future__ import annotations

import filecmp
import hashlib
import json
import os
import sys

import numpy as np
import pytest

from yapocis.codegen import bundle as bundle_mod
from yapocis.codegen.bundle import (
    BUNDLE_BACKENDS,
    DEFAULT_BUILD_OPTIONS,
    SCHEMA_VERSION,
    DispatchExprError,
    evaluate_dispatch_ast,
    export_bundle,
    load_bundle_config,
    load_manifest,
    packaged_interface_names,
    parse_dispatch_expr,
)
from yapocis.codegen.fixtures import (
    CASES,
    derivation_fixtures,
    write_derivation_fixtures,
)
from yapocis.core import get_test_backends
from yapocis.core.dispatch_expr import (
    evaluate_dispatch_expression,
    evaluate_dispatch_size,
)
from yapocis.core.interface import KernelParamSpec, get_interface
from yapocis.core.kmgr_base import prepare_program


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("bundle")
    manifest = export_bundle(str(out), log=lambda *a: None)
    return out, manifest


def _params(text: str, kernel: str) -> list[KernelParamSpec]:
    spec = get_interface(text)
    return next(k for k in spec.kernels if k.name == kernel).parameters


CNN_LIKE = """
interface t {
    kernel k(in float input, in float weights, out float output, int n);
        with { dispatch_size (dim_1_of input, dim_2_of input, dim_0_of output); };
};
"""


def test_packaged_interface_names_are_package_only():
    names = packaged_interface_names()
    assert "demo" in names and "graphics" in names
    # Interfaces from YAPOCIS_PROJECT / cwd must not leak into the default bundle
    assert all(
        os.path.exists(
            os.path.join(bundle_mod.packaged_interface_dir(), f"{n}.interface")
        )
        for n in names
    )


def test_prepare_program_renders_without_device():
    for backend in BUNDLE_BACKENDS:
        interface, merged, source = prepare_program(backend, "demo")
        assert interface.name == "demo"
        assert "sum" in source
        assert merged == {}


def test_export_covers_all_packaged_interfaces(bundle_dir):
    out, manifest = bundle_dir
    assert manifest.schema_version == SCHEMA_VERSION
    assert manifest.backends == list(BUNDLE_BACKENDS)
    exported = {e.name for e in manifest.interfaces}
    assert exported == set(packaged_interface_names())
    config = load_bundle_config()
    # Every configured variant is present
    for name, variants in config.variants.items():
        for variant in variants:
            vid = name if variant == "default" else f"{name}.{variant}"
            assert any(e.id == vid for e in manifest.interfaces), vid
    # Sources exist and hashes match
    for entry in manifest.interfaces:
        for src in entry.sources.values():
            path = out / src.path
            assert path.exists(), path
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == src.sha256
    # The manifest round-trips
    data = load_manifest(str(out / "manifest.json"))
    assert len(data["interfaces"]) == len(manifest.interfaces)


def test_demo_manifest_entry(bundle_dir):
    _, manifest = bundle_dir
    demo = next(e for e in manifest.interfaces if e.id == "demo")
    names = [k.name for k in demo.kernels]
    assert names == ["sum", "sum_res"]
    s, res = demo.kernels
    assert s.symbol == "sum" and res.symbol == "sum"
    assert [p.kind for p in s.params] == ["buffer", "buffer", "out_like"]
    assert s.user_args == ["a", "b"] and s.returns == ["out_like_a"]
    assert s.params[2].of == "a" and s.params[2].dtype == "float32"
    assert res.user_args == ["a", "b", "c"] and res.returns == ["c"]
    assert s.cuda == {"takes_global_size": True}
    assert demo.sources["cuda"].build_options == DEFAULT_BUILD_OPTIONS["cuda"]


def test_graphics_dim_params_precede_their_array(bundle_dir):
    _, manifest = bundle_dir
    graphics = next(e for e in manifest.interfaces if e.id == "graphics")
    lines = next(k for k in graphics.kernels if k.name == "lines")
    kinds = [(p.name, p.kind, p.dim, p.role) for p in lines.params[:3]]
    assert kinds == [
        ("a", "dim", "width_of", None),
        ("a", "dim", "height_of", None),
        ("a", "buffer", None, "inout"),
    ]
    assert lines.user_args[0] == "a" and lines.returns == ["a"]
    res = next(k for k in graphics.kernels if k.name == "lines_res")
    assert res.symbol == "lines" and res.returns == []


def test_fft_build_options_override(bundle_dir):
    _, manifest = bundle_dir
    fft = next(e for e in manifest.interfaces if e.id == "fft")
    assert fft.sources["ocl"].build_options == ["-cl-opt-disable"]
    assert fft.sources["cuda"].build_options == DEFAULT_BUILD_OPTIONS["cuda"]


def test_convolve_variants_bake_weights(bundle_dir):
    out, manifest = bundle_dir
    box3 = next(e for e in manifest.interfaces if e.id == "convolve.box3")
    assert box3.variant == "box3" and box3.context["name"] == "box3"
    assert [k.name for k in box3.kernels] == ["box3", "box3_fast"]
    assert "0.3333333333" in (out / box3.sources["ocl"].path).read_text()


def test_export_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    export_bundle(str(a), names=["demo", "cnn", "convolvesep"], log=lambda *x: None)
    export_bundle(str(b), names=["demo", "cnn", "convolvesep"], log=lambda *x: None)
    cmp = filecmp.dircmp(a, b)
    assert not cmp.diff_files and not cmp.left_only and not cmp.right_only
    for sub in cmp.subdirs.values():
        assert not sub.diff_files and not sub.left_only and not sub.right_only


@pytest.mark.parametrize(
    "expr",
    [
        "64",
        "dim_1_of input",
        "height_of input / 2",
        "width_of input",
        "sizeof output",
        "dim_0_of output * 2 + 1",
        "dim_2_of input // 3",
        "dim_1_of input % 4",
        "(dim_1_of input - 1) / 2",
        "dim_1_of input * dim_2_of input",
        "sizeof output // dim_0_of input",
    ],
)
def test_dispatch_ast_matches_dispatch_expr(expr):
    params = _params(CNN_LIKE, "k")
    args = (
        np.zeros((3, 7, 5), np.float32),
        np.zeros((4, 4), np.float32),
        np.zeros((9, 7, 5), np.float32),
        3,
    )
    expected = evaluate_dispatch_expression(expr, args, params)
    ast = parse_dispatch_expr(expr, params)
    assert int(evaluate_dispatch_ast(ast, args)) == expected


def test_dispatch_ast_records_param_index_like_python():
    params = _params(CNN_LIKE, "k")
    ast = parse_dispatch_expr("dim_0_of output", params)
    # dispatch_expr indexes the user args with the position in the full parameter list
    assert ast == {"dim": "dim_0_of", "of": "output", "param_index": 2}
    with pytest.raises(DispatchExprError):
        parse_dispatch_expr("dim_0_of nosuch", params)


def test_cnn_dispatch_sizes_in_manifest(bundle_dir):
    _, manifest = bundle_dir
    cnn = next(e for e in manifest.interfaces if e.id == "cnn")
    spec = get_interface(cnn.interface_text)
    with_dispatch = [k for k in cnn.kernels if k.dispatch_size]
    assert with_dispatch, "cnn kernels declare dispatch_size"
    for kentry in with_dispatch:
        kernel = next(k for k in spec.kernels if k.name == kentry.name)
        args = tuple(
            np.zeros((3, 8, 6), np.float32) if p.hint else 2
            for p in kernel.parameters
            if not p.derived
        )
        expected = evaluate_dispatch_size(
            kernel.traits["dispatch_size"], args, kernel.parameters
        )
        got = tuple(int(evaluate_dispatch_ast(n, args)) for n in kentry.dispatch_size)
        assert got == tuple(expected), kentry.name


def test_derivation_fixtures(bundle_dir):
    out, manifest = bundle_dir
    path = write_derivation_fixtures(str(out), manifest)
    data = json.load(open(path))
    assert data["cases"] == {k: list(v) for k, v in CASES.items()}
    demo = data["interfaces"]["demo"]["sum"]
    case = demo["cases"]["2d"]
    assert case["args"] == [
        {"shape": [7, 5], "dtype": "float32"},
        {"shape": [7, 5], "dtype": "float32"},
    ]
    assert case["kernel_args"][2] == {
        "param": "out_like_a",
        "kind": "out_like",
        "arg": 0,
        "shape": [7, 5],
        "dtype": "float32",
    }
    assert case["default_global_size"] == {"ocl": [35], "cuda": [35], "metal": [35, 1]}
    assert case["cuda_launch"] == {"grid": [1, 1, 1], "block": [256, 1, 1]}
    assert case["metal_launch"] == {"grid": [35, 1, 1], "tgs": [35, 1, 1]}
    # graphics: width_of/height_of are shape[0]/shape[1] (kernel_base semantics)
    lines = data["interfaces"]["graphics"]["lines"]["cases"]["2d"]["kernel_args"]
    assert lines[0] == {
        "param": "a",
        "kind": "dim",
        "dim": "width_of",
        "arg": 0,
        "value": 7,
    }
    assert lines[1] == {
        "param": "a",
        "kind": "dim",
        "dim": "height_of",
        "arg": 0,
        "value": 5,
    }
    one_d = data["interfaces"]["graphics"]["lines"]["cases"]["1d"]["kernel_args"]
    assert one_d[1]["value"] == 1  # height_of a 1-D array is 1
    # cnn: dispatch sizes recorded for the 3-D case
    conv = data["interfaces"]["cnn"]["conv2d_3x3"]["cases"]["3d"]
    assert conv["dispatch_global_size"] == [7, 5, 3]


def test_derivation_fixtures_cover_every_kernel(bundle_dir):
    _, manifest = bundle_dir
    data = derivation_fixtures(manifest)
    for entry in manifest.interfaces:
        assert set(data["interfaces"][entry.id]) == {k.name for k in entry.kernels}


def test_cli_bundle(tmp_path, monkeypatch):
    from yapocis.scripts import tools

    out = tmp_path / "cli"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "yapocis-tools",
            "bundle",
            "--out",
            str(out),
            "--backends",
            "ocl",
            "--fixtures",
            "demo",
        ],
    )
    assert tools.main() == 0
    data = load_manifest(str(out / "manifest.json"))
    assert data["backends"] == ["ocl"]
    assert [e["id"] for e in data["interfaces"]] == ["demo"]
    assert (out / "ocl" / "demo.cl").exists()
    assert (out / "fixtures" / "derivation.json").exists()


# --- parity fixtures (yapocis-tools bundle --parity) -------------------------


def test_parity_eligibility_and_cases():
    from yapocis.codegen import parity
    from yapocis.core.interface import load_interface

    demo = get_interface(load_interface("demo"))
    sum_k = next(k for k in demo.kernels if k.name == "sum")
    res_k = next(k for k in demo.kernels if k.name == "sum_res")
    assert parity.eligible(sum_k)  # in, in, out_like
    assert not parity.eligible(res_k)  # has an `out` parameter
    args, kwargs = parity.build_case("demo", sum_k, seed=1)
    assert [a.shape for a in args] == [parity.SHAPE, parity.SHAPE]
    assert kwargs == {}
    hsi = get_interface(load_interface("hsi"))
    rgb2hsi3 = next(k for k in hsi.kernels if k.name == "rgb2hsi3")
    args3, kwargs3 = parity.build_case("hsi", rgb2hsi3, seed=1)
    assert args3[0].shape == (48, 40, 3)
    assert kwargs3 == {"global_size": 48 * 40}
    again, _ = parity.build_case("demo", sum_k, seed=1)
    assert np.array_equal(args[0], again[0])
    assert parity.skipped("anisodiff", "filterImage3")
    assert not parity.skipped("anisodiff", "filterImage4")


@pytest.mark.parametrize("backend", get_test_backends())
def test_parity_export_on_backend(tmp_path, backend):
    from yapocis.codegen import parity
    from yapocis.codegen.parity import export_parity

    manifest = export_bundle(
        str(tmp_path), names=["demo", "operators"], log=lambda *a: None
    )
    path = export_parity(str(tmp_path), manifest, backend, log=lambda *a: None)
    index = json.load(open(path))
    assert index["backend"] == backend
    # fixtures live under the device they were recorded on, named as a string
    assert not index["device"].startswith("<")
    device_dir = tmp_path / "parity" / backend / parity.device_slug(index["device"])
    assert os.path.dirname(path) == str(device_dir)
    kernels = {(c["interface"], c["kernel"]) for c in index["cases"]}
    assert ("demo", "sum") in kernels
    assert ("operators", "add") in kernels
    case = next(c for c in index["cases"] if c["kernel"] == "sum")
    data = np.load(device_dir / case["file"])
    a, b, out = data["in_0"], data["in_1"], data["out_0"]
    np.testing.assert_allclose(out, a + b, rtol=1e-6, atol=1e-6)


# --- external project bundles (yapocis-tools bundle --project-only) ---------


def test_project_only_exports_the_project_interfaces(tmp_path, monkeypatch):
    """A project's bundle holds only the interfaces the package does not ship.

    impressionist-rs vendors yapocis_external's bundle this way, beside the
    package's own; the two must not overlap."""
    import shutil

    project = tmp_path / "project"
    (project / "interfaces").mkdir(parents=True)
    package = os.path.dirname(bundle_mod.packaged_interface_dir())
    demo = open(os.path.join(package, "interfaces", "demo.interface")).read()
    (project / "interfaces" / "projdemo.interface").write_text(
        demo.replace("interface demo", "interface projdemo")
    )
    for backend in ("ocl", "metal", "cuda"):
        (project / "kernels" / backend).mkdir(parents=True)
        shutil.copy(
            os.path.join(package, "kernels", backend, "demo.mako"),
            project / "kernels" / backend / "projdemo.mako",
        )
    monkeypatch.setenv("YAPOCIS_PROJECT", str(project))
    # The interface registry caches its search directories; an earlier test (or a
    # YAPOCIS_PROJECT in the environment) must not leak into this one
    import yapocis.interfaces

    monkeypatch.setattr(yapocis.interfaces.am, "cached_asset_dirs", None)
    monkeypatch.setattr(yapocis.interfaces.am, "cached_assets", None)

    assert bundle_mod.project_interface_names() == ["projdemo"]
    assert "projdemo" not in packaged_interface_names()

    out = tmp_path / "bundle"
    manifest = export_bundle(str(out), project_only=True, log=lambda *a: None)
    assert [e.id for e in manifest.interfaces] == ["projdemo"]
    assert (out / "ocl" / "projdemo.cl").exists()

    # The CLI flag reaches export_bundle the same way
    from yapocis.scripts import tools

    cli = tmp_path / "cli"
    monkeypatch.setattr(
        sys, "argv", ["yapocis-tools", "bundle", "--out", str(cli), "--project-only"]
    )
    assert tools.main() == 0
    data = load_manifest(str(cli / "manifest.json"))
    assert [e["id"] for e in data["interfaces"]] == ["projdemo"]
