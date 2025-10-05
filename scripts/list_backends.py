#!/usr/bin/env python3
"""List all available backends and their devices."""

import sys
from typing import Dict, Any

from yapocis.core.manager import get_backends, get_backend


def get_backend_info() -> Dict[str, Any]:
    """Get information about all available backends and their devices."""
    backend_info = {}

    for backend in get_backends():
        backend_info[backend] = {"available": True, "devices": [], "error": None}

        try:
            if backend == "ocl":
                from yapocis.ocl.device import get_available_devices

                devices = get_available_devices(aliases=False)
                backend_info[backend]["devices"] = list(devices.keys())

            elif backend == "metal":
                from yapocis.metallic.device import get_available_devices

                devices = get_available_devices(aliases=False)
                backend_info[backend]["devices"] = list(devices.keys())

            elif backend == "cuda":
                try:
                    import cupy as cp

                    n_devices = cp.cuda.runtime.getDeviceCount()
                    backend_info[backend]["devices"] = [
                        f"CUDA Device {i}: {cp.cuda.Device(i).attributes['Name'].decode('utf-8')}"
                        for i in range(n_devices)
                    ]
                except Exception as e:
                    backend_info[backend]["error"] = str(e)
                    backend_info[backend]["available"] = False

            elif backend == "numba":
                # Numba runs on CPU, no separate device enumeration
                import platform

                backend_info[backend]["devices"] = [
                    f"CPU: {platform.processor() or platform.machine()}"
                ]

        except Exception as e:
            backend_info[backend]["error"] = str(e)
            backend_info[backend]["available"] = False

    return backend_info


def print_backend_info(backend_info: Dict[str, Any], verbose: bool = False):
    """Print backend and device information in a formatted way."""
    print("=" * 80)
    print("Available Backends and Devices")
    print("=" * 80)
    print()
    current_backend = get_backend()
    for backend, info in backend_info.items():
        status = "✓" if info["available"] else "✗"
        if backend == current_backend:
            status += "*"
        print(f"{status} {backend.upper()}")

        if info["available"] and info["devices"]:
            for i, device in enumerate(info["devices"]):
                print(f"  [{i}] {device}")
        elif info["error"]:
            print(f"  Error: {info['error']}")
        else:
            print(f"  No devices found")
        print()

    if verbose:
        print("Environment Variables:")
        print("  YAPOCIS_BACKEND - Select backend (ocl, metal, cuda, numba)")
        print("  YAPOCIS_ENGINE  - Select device by name or index")
        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="List all available Yapocis backends and devices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all backends and devices
  python scripts/list_backends.py

  # Show with environment variable help
  python scripts/list_backends.py -v

  # JSON output
  python scripts/list_backends.py --json
        """,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show additional information about environment variables",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    try:
        backend_info = get_backend_info()

        if args.json:
            import json

            print(json.dumps(backend_info, indent=2))
        else:
            print_backend_info(backend_info, verbose=args.verbose)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
