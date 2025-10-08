#!/usr/bin/env python3
"""list all available backends and their devices."""

import sys
from typing import Any

from yapocis.core import (
    get_backend,
    get_backend_infos,
    get_backend_device,
)

def print_backend_info(info: dict[str, Any], verbose: bool = False):
    """Print backend and device information in a formatted way."""
    print("=" * 80)
    print("Available Backends and Devices")
    print("=" * 80)
    print()
    current_backend = get_backend()
    current_device_index, current_device_name = get_backend_device()

    for backend, info in info.items():
        status = "✓" if info["available"] else "✗"
        is_current_backend = backend == current_backend
        backend_marker = " ←" if is_current_backend else ""
        print(f"{status} {backend.upper()}{backend_marker}")

        if info["available"] and info["devices"]:
            for i, device in enumerate(info["devices"]):
                # Mark selected device for current backend
                device_marker = ""
                if is_current_backend:
                    if current_device_index == i or current_device_name == device:
                        device_marker = " ←"
                print(f"  [{i}] {device}{device_marker}")
        elif info["error"]:
            print(f"  Error: {info['error']}")
        else:
            print(f"  No devices found")
        print()

    print("Legend:")
    print("  ✓  = Backend available")
    print("  ✗  = Backend not available")
    print("  ← = Currently selected (backend or device)")
    print()

    if verbose:
        print("Environment Variables:")
        print("  YAPOCIS_BACKEND - Select backend (ocl, metal, cuda, numba)")
        print("  YAPOCIS_DEVICE  - Select device by name or index")
        print()

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="list all available Yapocis backends and devices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # list all backends and devices
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
        backend_infos = get_backend_infos()

        if args.json:
            import json

            print(json.dumps(backend_infos, indent=2))
        else:
            print_backend_info(backend_infos, verbose=args.verbose)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
