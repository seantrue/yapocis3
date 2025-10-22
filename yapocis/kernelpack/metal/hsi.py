"""
Auto-generated Metal kernel package for hsi interface.

Generated from: yapocis/interfaces/hsi.interface
Backend: Metal
Generation date: 2025-10-17

This package provides zero-overhead calls to HSI/RGB conversion kernels.
"""

import numpy as np
from typing import Tuple
from yapocis.metallic.device_manager import DeviceManager


class HsiPackage:
    """
    Compiled kernel package for HSI/RGB color space conversion operations.

    Interface pattern:
        kernel rgb2hsi(in float r, in float g, in float b, out_like float r, out_like float r, out_like float r);
        kernel hsi2rgb(in float h, in float s, in float i, out_like h, out_like h, out_like h);
        kernel rgb2hsi3(depth_of rgb, in float rgb, out_like rgb);
        kernel hsi2rgb3(depth_of hsi, in float hsi, out_like hsi);
    """

    def __init__(self, dmgr: DeviceManager, program):
        self.dmgr = dmgr
        self.queue = dmgr.queue
        self.autoflatten = dmgr.autoflatten
        self.program = program
        self._kernels: dict[str, any] = {}

    def load(self, kernel_names: list[str]) -> None:
        """Load specific kernels from the program."""
        for name in kernel_names:
            self._kernels[name] = getattr(self.program, name)

    def rgb2hsi(self, r: np.ndarray, g: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert RGB to HSI (separate channels)."""
        r = r.astype(np.float32, copy=False)
        g = g.astype(np.float32, copy=False)
        b = b.astype(np.float32, copy=False)

        buf_r = self.dmgr.write(r, wait=True)
        buf_g = self.dmgr.write(g, wait=True)
        buf_b = self.dmgr.write(b, wait=True)

        h = np.empty_like(r, dtype=np.float32)
        s = np.empty_like(r, dtype=np.float32)
        i = np.empty_like(r, dtype=np.float32)

        buf_h = self.dmgr.ensure(h)
        buf_s = self.dmgr.ensure(s)
        buf_i = self.dmgr.ensure(i)

        global_size = (r.size,) if self.autoflatten else r.shape
        kernel = self._kernels['rgb2hsi']
        kernel(self.queue, global_size, None, buf_r, buf_g, buf_b, buf_h, buf_s, buf_i)

        self.dmgr.read(h)
        self.dmgr.read(s)
        self.dmgr.read(i)
        return h, s, i

    def hsi2rgb(self, h: np.ndarray, s: np.ndarray, i: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert HSI to RGB (separate channels)."""
        h = h.astype(np.float32, copy=False)
        s = s.astype(np.float32, copy=False)
        i = i.astype(np.float32, copy=False)

        buf_h = self.dmgr.write(h, wait=True)
        buf_s = self.dmgr.write(s, wait=True)
        buf_i = self.dmgr.write(i, wait=True)

        r = np.empty_like(h, dtype=np.float32)
        g = np.empty_like(h, dtype=np.float32)
        b = np.empty_like(h, dtype=np.float32)

        buf_r = self.dmgr.ensure(r)
        buf_g = self.dmgr.ensure(g)
        buf_b = self.dmgr.ensure(b)

        global_size = (h.size,) if self.autoflatten else h.shape
        kernel = self._kernels['hsi2rgb']
        kernel(self.queue, global_size, None, buf_h, buf_s, buf_i, buf_r, buf_g, buf_b)

        self.dmgr.read(r)
        self.dmgr.read(g)
        self.dmgr.read(b)
        return r, g, b

    def rgb2hsi3(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to HSI (packed 3-channel array)."""
        depth = np.int32(rgb.shape[2] if len(rgb.shape) > 2 else 1)

        rgb = rgb.astype(np.float32, copy=False)
        buf_rgb = self.dmgr.write(rgb, wait=True)

        hsi = np.empty_like(rgb, dtype=np.float32)
        buf_hsi = self.dmgr.ensure(hsi)

        global_size = (rgb.size,) if self.autoflatten else rgb.shape
        kernel = self._kernels['rgb2hsi3']
        kernel(self.queue, global_size, None, depth, buf_rgb, buf_hsi)

        self.dmgr.read(hsi)
        return hsi

    def hsi2rgb3(self, hsi: np.ndarray) -> np.ndarray:
        """Convert HSI to RGB (packed 3-channel array)."""
        depth = np.int32(hsi.shape[2] if len(hsi.shape) > 2 else 1)

        hsi = hsi.astype(np.float32, copy=False)
        buf_hsi = self.dmgr.write(hsi, wait=True)

        rgb = np.empty_like(hsi, dtype=np.float32)
        buf_rgb = self.dmgr.ensure(rgb)

        global_size = (hsi.size,) if self.autoflatten else hsi.shape
        kernel = self._kernels['hsi2rgb3']
        kernel(self.queue, global_size, None, depth, buf_hsi, buf_rgb)

        self.dmgr.read(rgb)
        return rgb
