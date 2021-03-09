#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from yapocis.lib.anisodiff import Anisodiff
from yapocis.lib.channels import split_planes, join_planes
from yapocis.lib.hsi import HSI
from yapocis.utils.config import get_device_config
from yapocis.utils.typing import *
from . import image_test, TS, show_snapshots


N = 100


def setup(device_name:Optional[str]=None):
    image = image_test()
    return image, Anisodiff(device_name=device_name)

def test_anisodiff_I():
    rgb, aniso = setup()
    hsis = HSI()
    hsi = hsis.rgb2hsi3(rgb)
    H,S,I = split_planes(hsi)
    with TS(f"[generator {N}]") as show:
        for i, fI in enumerate(aniso.anisodiff(I)):
            if i >= N:
                break
        hsi = join_planes(H,S,fI)
        rgb2 = hsis.hsi2rgb3(hsi)
        show(aniso, rgb2)
    with TS(f"[{N}]") as show:
        fI = aniso.anisodiff_n(I, N)
        hsi = join_planes(H,S,fI)
        rgb2 = hsis.hsi2rgb3(hsi)
        show(aniso, rgb2)

def test_anisodiff_IS():
    from yapocis.rpc.device import get_available_device_names
    for device_name in get_available_device_names():
        rgb, aniso = setup(device_name=device_name)
        with TS(f"[generator {N} {device_name[:24]}]") as show:
            for i, rgb2 in enumerate(aniso.anisodiff_rgbIS(rgb)):
                if i >= N:
                    break
            show(aniso, rgb2)
        with TS(f"[{N}] {device_name[:24]}") as show:
            rgb2 = aniso.anisodiff_rgbIS_n(rgb, N)
            show(aniso, rgb2)

def test_anisodiff1():
    from yapocis.rpc.device import get_available_device_names
    for device_name in get_available_device_names():
        rgb, aniso = setup(device_name=device_name)
        hsis = HSI(device_name=device_name)
        hsi = hsis.rgb2hsi3(rgb)
        H,S,I = split_planes(hsi)
        with TS(f"[{N}] on {device_name[:24]}") as show:
            fI = aniso.anisodiff_n(I, N)
            hsi = join_planes(H,S,fI)
            rgb2 = hsis.hsi2rgb3(hsi)
            show(aniso, rgb2)


def test_anisodiff2():
    from yapocis.rpc.device import get_available_device_names
    for device_name in get_available_device_names():
        config = get_device_config(device_name=device_name)
        if config["preferred_vector_width_float"] > 3:
            print("preferred_vector_width_float", config["preferred_vector_width_float"])
            print(f"The {config['__name__']} OpenCL engine does not support float3")
            continue
        rgb, aniso = setup(device_name=device_name)
        r,g,b = split_planes(rgb)
        rg = join_planes(r,g)
        with TS(f"[{N}] on {device_name[:24]}") as show:
            rg2 = aniso.anisodiff_n(rg, N)
            r2,g2 = split_planes(rg2)
            rgb2 = join_planes(r2,g2, b)
            show(aniso, rgb2)


def test_anisodiff3():
    from yapocis.rpc.device import get_available_device_names
    for device_name in get_available_device_names():
        config = get_device_config(device_name=device_name)
        if config["preferred_vector_width_float"] > 3:
            print("preferred_vector_width_float", config["preferred_vector_width_float"])
            print(f"The {config['__name__']} OpenCL engine does not support float3")
            continue
        rgb, aniso = setup(device_name=device_name)
        with TS(f"[{N}] on {device_name[:24]}") as show:
            rgb2 = aniso.anisodiff_n(rgb,N)
            show(aniso, rgb2)


def test_anisodiff4():
    from yapocis.rpc.device import get_available_device_names
    for device_name in get_available_device_names():
        rgb, aniso = setup(device_name=device_name)
        shape4 = list(rgb.shape)
        shape4[2] = 4
        rgba = np.zeros((shape4), dtype=np.float32)
        rgba[:, :, :3] = rgb
        with TS(f"[{N}] on {device_name[:24]}") as show:
            rgba2 = aniso.anisodiff_n(rgba, N)
            show(aniso,rgba2[:, :, :3])
