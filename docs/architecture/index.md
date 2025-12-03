# Architecture

Understanding yapocis internals.

This section describes the key architectural components of yapocis:

- **Buffer Management**: How GPU buffers are tracked, written, and read
- **Interface Specification**: The IDL syntax for defining kernel interfaces
- **Kernel Code**: Backend-specific kernel implementations (OpenCL, CUDA, Metal, Numba)
- **Kernel Program**: The KernelManager architecture that dispatches calls
- **Bound Kernels**: The high-level library pattern
- **Extensions**: Using YAPOCIS_PROJECT for custom kernel collections

## Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Application                             │
├─────────────────────────────────────────────────────────────────┤
│                    Library Layer (yapocis.lib)                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ Graphics4│ │ Gaussian │ │Anisodiff │ │ ... 25+ modules  │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────────┬─────────┘   │
│       └────────────┴────────────┴────────────────┘              │
│                           │                                      │
│                    BoundKernels base class                       │
├─────────────────────────────────────────────────────────────────┤
│                  Core Layer (yapocis.core)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ load_program │  │InterfaceSpec │  │   KernelManager      │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────────┘  │
│         │                 │                    │                 │
│         ▼                 ▼                    ▼                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Interface Parser (PyParsing)                 │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                   Backend Layer                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │  OpenCL  │ │   CUDA   │ │  Metal   │ │      Numba       │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────────┬─────────┘   │
│       │            │            │                 │              │
│       ▼            ▼            ▼                 ▼              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │     Mako Templates (.mako) / Python Kernels (.py)         │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                   Hardware Layer                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ CPU/GPU  │ │  NVIDIA  │ │  Apple   │ │       CPU        │   │
│  │ (OpenCL) │ │   GPU    │ │   GPU    │ │    (Numba JIT)   │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **User code** calls library methods (e.g., `Gaussian().gauss_image(img)`)
2. **BoundKernels** delegates to the **KernelManager** via `load_program()`
3. **Interface parser** reads `.interface` files defining kernel signatures
4. **Template engine** (Mako) generates backend-specific kernel code
5. **Backend** compiles and executes kernels on the appropriate device
6. **Buffer manager** handles CPU↔GPU data transfers

```{toctree}
:maxdepth: 1

buffer-management
interface-spec
kernel-code
kernel-program
bound-kernels
extensions
```
