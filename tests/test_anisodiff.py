#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

from yapocis.core.manager import get_backend
from yapocis.lib.anisodiff import Anisodiff
from yapocis.lib.channels import join_planes, split_planes
from yapocis.lib.hsi import HSI
from yapocis.utils.config import get_device_config
from yapocis.utils.stereo_typing import *
from . import StopWatcher, image_test

N = 100


def get_available_device_names():
    if get_backend() == "ocl":
        from yapocis.ocl.device import get_available_device_names as gadn

        return gadn()
    return [""]


def setup_test(device_name: Optional[str] = None):
    image = image_test()
    # Ensure that we don't have fortunate addressing!
    assert image.shape[0] != image.shape[1]
    return image, Anisodiff(device_name=device_name)


def test_I():
    rgb, aniso = setup_test()
    hsis = HSI()
    hsi = hsis.rgb2hsi3(rgb)
    H, S, I = split_planes(hsi)
    with StopWatcher(f"[generator {N}]") as sw:
        for i, fI in enumerate(aniso.anisodiff(I)):
            if i >= N:
                break
        hsi = join_planes(H, S, fI)
        rgb2 = hsis.hsi2rgb3(hsi)
        sw.show(aniso, rgb2)
    with StopWatcher(f"[{N}]") as sw:
        fI = aniso.anisodiff_n(I, N)
        hsi = join_planes(H, S, fI)
        rgb2 = hsis.hsi2rgb3(hsi)
        sw.show(aniso, rgb2)


def test_IS():
    for device_name in get_available_device_names():
        rgb, aniso = setup_test(device_name=device_name)
        with StopWatcher(f"[generator {N} {device_name[:24]}]") as sw:
            for i, rgb2 in enumerate(aniso.anisodiff_rgbIS(rgb)):
                if i >= N:
                    break
            sw.show(aniso, rgb2)
        with StopWatcher(f"[{N}] {device_name[:24]}") as sw:
            rgb2 = aniso.anisodiff_rgbIS_n(rgb, N)
            sw.show(aniso, rgb2)


def test_1plane():
    for device_name in get_available_device_names():
        rgb, aniso = setup_test(device_name=device_name)
        hsis = HSI(device_name=device_name)
        hsi = hsis.rgb2hsi3(rgb)
        H, S, I = split_planes(hsi)
        with StopWatcher(f"[{N}] on {device_name[:24]}") as sw:
            fI = aniso.anisodiff_n(I, N)
            hsi = join_planes(H, S, fI)
            rgb2 = hsis.hsi2rgb3(hsi)
            sw.show(aniso, rgb2)


def test_2plane():
    for device_name in get_available_device_names():
        config = get_device_config(device_name=device_name)
        if config["preferred_vector_width_float"] > 2:
            print(
                "preferred_vector_width_float", config["preferred_vector_width_float"]
            )
            print(f"The {config['__name__']} OpenCL engine does not support float2")
            continue
        rgb, aniso = setup_test(device_name=device_name)
        r, g, b = split_planes(rgb)
        rg = join_planes(r, g)
        with StopWatcher(f"[{N}] on {device_name[:24]}") as sw:
            rg2 = aniso.anisodiff_n(rg, N)
            r2, g2 = split_planes(rg2)
            rgb2 = join_planes(r2, g2, b)
            sw.show(aniso, rgb2)


def test_3plane():
    from yapocis.core.manager import get_backend

    device_names = get_available_device_names()
    backend = get_backend()
    if backend == "ocl":
        for device_name in get_available_device_names():
            config = get_device_config(device_name=device_name)
            if config["preferred_vector_width_float"] > 3:
                print(
                    "preferred_vector_width_float",
                    config["preferred_vector_width_float"],
                )
                print(f"The {config['__name__']} OpenCL engine does not support float3")
                continue
            if config.get("compute_capability_major_nv", ""):
                print("Nvidia drivers don't handle float3 well")
                continue
            rgb, aniso = setup_test(device_name=device_name)
            with StopWatcher(f"[{N}] on {backend} {device_name[:24]}") as sw:
                rgb2 = aniso.anisodiff_n(rgb, N)
                sw.show(aniso, rgb2)
    else:
        with StopWatcher(f"[{N}] on {backend}") as sw:
            rgb, aniso = setup_test()
            rgb2 = aniso.anisodiff_n(rgb, N)
            sw.show(aniso, rgb2)


def test_4plane():
    from yapocis.ocl.device import get_available_device_names

    for device_name in get_available_device_names():
        rgb, aniso = setup_test(device_name=device_name)
        shape4 = list(rgb.shape)
        shape4[2] = 4
        rgba = np.zeros((shape4), dtype=np.float32)
        rgba[:, :, :3] = rgb
        with StopWatcher(f"[{N}] on {device_name[:24]}") as sw:
            rgba2 = aniso.anisodiff_n(rgba, N)
            sw.show(aniso, rgba2[:, :, :3])
