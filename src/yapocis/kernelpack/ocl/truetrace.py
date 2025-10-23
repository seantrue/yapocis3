"""
Auto-generated OpenCL kernel package for truetrace interface.

Generated from: yapocis/interfaces/truetrace.interface
Backend: OpenCL
Generation date: 2025-10-17

This package provides zero-overhead calls to signed distance field / jump flood algorithm kernels.
"""

import numpy as np
from typing import Tuple
from yapocis.ocl.device_manager import DeviceManager


class TruetracePackage:
    """
    Compiled kernel package for signed distance field operations using Jump Flood Algorithm.

    Interface pattern:
        kernel init_seed_from_mask(width_of mask, height_of mask, in int mask, out float seed, out float dist);
        kernel jfa_step(width_of seed_in, height_of seed_in, int jump, in float seed_in, inout float dist, out float seed_out);
        kernel marching_squares(width_of dist, height_of dist, in float dist, float iso, inout uint32 seg_counter, out float segments_out);
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

    def init_seed_from_mask(
        self,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize jump-flood seeds and distance from a binary mask.

        Args:
            mask: Binary mask (int32)

        Returns:
            Tuple of (seed, dist) arrays
        """
        # Param 0: width_of mask
        width = np.int32(mask.shape[0])

        # Param 1: height_of mask
        height = np.int32(mask.shape[1])

        # Param 2: in int mask
        mask = mask.astype(np.int32, copy=False)
        buf_mask = self.dmgr.write(mask, wait=True)

        # Param 3: out float seed (float4 = 4 components)
        seed = np.empty((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
        buf_seed = self.dmgr.ensure(seed)

        # Param 4: out float dist
        dist = np.empty_like(mask, dtype=np.float32)
        buf_dist = self.dmgr.ensure(dist)

        # Execute kernel
        global_size = (mask.size,) if self.autoflatten else mask.shape
        kernel = self._kernels['init_seed_from_mask']
        kernel(
            self.queue,
            global_size,
            None,
            width,
            height,
            buf_mask,
            buf_seed,
            buf_dist
        )

        self.dmgr.read(seed)
        self.dmgr.read(dist)
        return seed, dist

    def jfa_step(
        self,
        seed_in: np.ndarray,
        dist: np.ndarray,
        jump: int
    ) -> np.ndarray:
        """
        Perform one Jump Flood Algorithm step.

        Args:
            seed_in: Input seed array (Hx Wx2)
            dist: Distance array (inout, HxW)
            jump: Jump length for this step

        Returns:
            seed_out: Output seed array (HxWx2)
        """
        # Param 0: width_of seed_in
        width = np.int32(seed_in.shape[0])

        # Param 1: height_of seed_in
        height = np.int32(seed_in.shape[1])

        # Param 2: int jump
        jump_i32 = np.int32(jump)

        # Param 3: in float seed_in
        seed_in = seed_in.astype(np.float32, copy=False)
        buf_seed_in = self.dmgr.write(seed_in, wait=True)

        # Param 4: inout float dist
        dist = dist.astype(np.float32, copy=False)  # No copy needed
        buf_dist = self.dmgr.write(dist, wait=True)

        # Param 5: out float seed_out
        seed_out = np.empty_like(seed_in, dtype=np.float32)
        buf_seed_out = self.dmgr.ensure(seed_out)

        # Execute kernel
        global_size = (seed_in.shape[0] * seed_in.shape[1],) if self.autoflatten else (seed_in.shape[0], seed_in.shape[1])
        kernel = self._kernels['jfa_step']
        kernel(
            self.queue,
            global_size,
            None,
            width,
            height,
            jump_i32,
            buf_seed_in,
            buf_dist,
            buf_seed_out
        )

        self.dmgr.read(seed_out)
        self.dmgr.read(dist)  # Read back modified dist
        return seed_out

    def marching_squares(
        self,
        dist: np.ndarray,
        iso: float,
        seg_counter: np.ndarray,
        segments_out: np.ndarray
    ) -> None:
        """
        Extract line segments at iso-level using marching squares.

        Args:
            dist: Distance field (HxW)
            iso: Iso-level value
            seg_counter: Segment counter (inout, single uint32)
            segments_out: Output segments array
        """
        # Param 0: width_of dist
        width = np.int32(dist.shape[0])

        # Param 1: height_of dist
        height = np.int32(dist.shape[1])

        # Param 2: in float dist
        dist = dist.astype(np.float32, copy=False)
        buf_dist = self.dmgr.write(dist, wait=True)

        # Param 3: float iso
        iso_f32 = np.float32(iso)

        # Param 4: inout uint32 seg_counter
        seg_counter = seg_counter.astype(np.uint32, copy=False)  # No copy needed
        buf_seg_counter = self.dmgr.write(seg_counter, wait=True)

        # Param 5: out float segments_out
        segments_out = segments_out.astype(np.float32, copy=False)
        buf_segments_out = self.dmgr.ensure(segments_out)

        # Execute kernel
        global_size = (dist.size,) if self.autoflatten else dist.shape
        kernel = self._kernels['marching_squares']
        kernel(
            self.queue,
            global_size,
            None,
            width,
            height,
            buf_dist,
            iso_f32,
            buf_seg_counter,
            buf_segments_out
        )

        self.dmgr.read(seg_counter)  # Read back modified counter
        self.dmgr.read(segments_out)
