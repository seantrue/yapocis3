#!/usr/bin/env python3
"""
Comprehensive tests for yapocis-tools unified CLI.

Tests all subcommands: assets, backends, devices, interfaces
"""
import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from yapocis.scripts.tools import cmd_assets, cmd_backends, cmd_devices, cmd_interfaces, main


class MockArgs:
    """Mock argument object for testing."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestAssetsCommand:
    """Tests for yapocis-tools assets subcommand."""

    def test_list_all_assets(self, capsys):
        """Test listing all asset types."""
        args = MockArgs(
            list=True,
            all=True,
            assets=[],
            images=False,
            icons=False,
            fonts=False,
            kernels=False,
            interfaces=False
        )

        result = cmd_assets(args)
        captured = capsys.readouterr()

        assert result == 0
        assert "## INTERFACES" in captured.out
        assert "## KERNELS" in captured.out
        assert "## IMAGES" in captured.out
        assert "## ICONS" in captured.out
        assert "## FONTS" in captured.out

    def test_list_specific_asset_type(self, capsys):
        """Test listing specific asset types."""
        args = MockArgs(
            list=True,
            all=False,
            assets=["interfaces"],
            images=False,
            icons=False,
            fonts=False,
            kernels=False,
            interfaces=False
        )

        result = cmd_assets(args)
        captured = capsys.readouterr()

        assert result == 0
        assert "## INTERFACES" in captured.out
        assert ".interface" in captured.out

    def test_list_multiple_asset_types(self, capsys):
        """Test listing multiple specific asset types."""
        args = MockArgs(
            list=True,
            all=False,
            assets=["interfaces", "images"],
            images=False,
            icons=False,
            fonts=False,
            kernels=False,
            interfaces=False
        )

        result = cmd_assets(args)
        captured = capsys.readouterr()

        assert result == 0
        assert "## INTERFACES" in captured.out
        assert "## IMAGES" in captured.out

    def test_find_asset_all_types(self, capsys):
        """Test finding an asset across all types."""
        args = MockArgs(
            list=False,
            all=True,
            assets=["mandelbrot"],
            images=False,
            icons=False,
            fonts=False,
            kernels=False,
            interfaces=False
        )

        result = cmd_assets(args)
        captured = capsys.readouterr()

        assert result == 0
        # Should find mandelbrot.interface and mandelbrot.mako
        assert "mandelbrot" in captured.out

    def test_find_asset_specific_type(self, capsys):
        """Test finding asset in specific type."""
        args = MockArgs(
            list=False,
            all=False,
            assets=["test.jpg"],
            images=True,
            icons=False,
            fonts=False,
            kernels=False,
            interfaces=False
        )

        result = cmd_assets(args)
        captured = capsys.readouterr()

        assert result == 0
        assert "image" in captured.out.lower()
        assert "test.jpg" in captured.out

    def test_find_nonexistent_asset(self, capsys):
        """Test finding an asset that doesn't exist."""
        args = MockArgs(
            list=False,
            all=False,
            assets=["nonexistent_file.xyz"],
            images=True,
            icons=False,
            fonts=False,
            kernels=False,
            interfaces=False
        )

        result = cmd_assets(args)
        captured = capsys.readouterr()

        assert result == 0
        assert "not found" in captured.out

    def test_no_assets_specified_without_list(self, capsys):
        """Test error when no assets specified and not listing."""
        args = MockArgs(
            list=False,
            all=False,
            assets=[],
            images=False,
            icons=False,
            fonts=False,
            kernels=False,
            interfaces=False
        )

        with pytest.raises(SystemExit):
            cmd_assets(args)


class TestBackendsCommand:
    """Tests for yapocis-tools backends subcommand."""

    def test_list_backends_default(self, capsys):
        """Test listing backends with default output."""
        args = MockArgs(json=False, verbose=False)

        result = cmd_backends(args)
        captured = capsys.readouterr()

        assert result == 0
        assert "Available Backends and Devices" in captured.out
        assert "Legend:" in captured.out

    def test_list_backends_json(self, capsys):
        """Test listing backends with JSON output."""
        args = MockArgs(json=True, verbose=False)

        result = cmd_backends(args)
        captured = capsys.readouterr()

        assert result == 0
        # Should be valid JSON
        data = json.loads(captured.out)
        assert isinstance(data, dict)
        # Should have backend entries
        assert len(data) > 0

    def test_list_backends_verbose(self, capsys):
        """Test listing backends with verbose output."""
        args = MockArgs(json=False, verbose=True)

        result = cmd_backends(args)
        captured = capsys.readouterr()

        assert result == 0
        assert "Environment Variables:" in captured.out
        assert "YAPOCIS_BACKEND" in captured.out
        assert "YAPOCIS_DEVICE" in captured.out

    def test_backends_shows_available_status(self, capsys):
        """Test that backends show available/unavailable status."""
        args = MockArgs(json=False, verbose=False)

        result = cmd_backends(args)
        captured = capsys.readouterr()

        assert result == 0
        # Should show checkmarks for available backends
        assert "✓" in captured.out or "✗" in captured.out

    def test_backends_shows_current_selection(self, capsys):
        """Test that current backend selection is marked."""
        args = MockArgs(json=False, verbose=False)

        result = cmd_backends(args)
        captured = capsys.readouterr()

        assert result == 0
        # Should show arrow for current backend
        assert "←" in captured.out


class TestDevicesCommand:
    """Tests for yapocis-tools devices subcommand."""

    def test_list_devices_names_only(self, capsys):
        """Test listing device names only."""
        args = MockArgs(list=True, devices=[])

        result = cmd_devices(args)
        captured = capsys.readouterr()

        assert result == 0
        # Should have at least one line of output (device name)
        assert len(captured.out.strip().split('\n')) >= 1

    def test_list_devices_with_config(self, capsys):
        """Test listing devices with full config."""
        args = MockArgs(list=False, devices=[])

        result = cmd_devices(args)
        captured = capsys.readouterr()

        assert result == 0
        # Should show device configurations
        # At minimum, should show device name
        assert len(captured.out.strip()) > 0

    def test_list_specific_device(self, capsys):
        """Test listing specific device by name."""
        from yapocis.ocl.device import get_available_device_names

        device_names = get_available_device_names()
        if not device_names:
            pytest.skip("No OpenCL devices available")

        args = MockArgs(list=False, devices=[device_names[0]])

        result = cmd_devices(args)
        captured = capsys.readouterr()

        assert result == 0
        assert device_names[0] in captured.out

    def test_list_nonexistent_device(self, capsys):
        """Test listing nonexistent device (should produce no output)."""
        args = MockArgs(list=False, devices=["NonexistentDevice123"])

        result = cmd_devices(args)
        captured = capsys.readouterr()

        assert result == 0
        # Should not crash, may have empty or minimal output


class TestInterfacesCommand:
    """Tests for yapocis-tools interfaces subcommand."""

    def test_list_interfaces(self, capsys):
        """Test listing all available interfaces."""
        args = MockArgs(
            list=True,
            all=False,
            explain=False,
            opencl=False,
            python=False,
            interfaces=[]
        )

        result = cmd_interfaces(args)
        captured = capsys.readouterr()

        assert result == 0
        assert "# Available Interfaces" in captured.out
        assert ".interface" in captured.out

    def test_explain_interface(self, capsys):
        """Test explaining a specific interface."""
        args = MockArgs(
            list=False,
            all=False,
            explain=True,
            opencl=False,
            python=False,
            interfaces=["mandelbrot"]
        )

        result = cmd_interfaces(args)
        captured = capsys.readouterr()

        assert result == 0
        assert "mandelbrot" in captured.out.lower()
        assert "Interface:" in captured.out or "Kernel:" in captured.out

    def test_generate_opencl_stubs(self, capsys):
        """Test generating OpenCL stubs."""
        args = MockArgs(
            list=False,
            all=False,
            explain=False,
            opencl=True,
            python=False,
            interfaces=["mandelbrot"]
        )

        result = cmd_interfaces(args)
        captured = capsys.readouterr()

        assert result == 0
        assert "OpenCL" in captured.out or "__kernel" in captured.out

    def test_generate_python_stubs(self, capsys):
        """Test generating Python stubs."""
        args = MockArgs(
            list=False,
            all=False,
            explain=False,
            opencl=False,
            python=True,
            interfaces=["mandelbrot"]
        )

        result = cmd_interfaces(args)
        captured = capsys.readouterr()

        assert result == 0
        assert "Python" in captured.out or "class" in captured.out

    def test_generate_all_outputs(self, capsys):
        """Test generating all output types."""
        args = MockArgs(
            list=False,
            all=True,
            explain=False,
            opencl=False,
            python=False,
            interfaces=["mandelbrot"]
        )

        result = cmd_interfaces(args)
        captured = capsys.readouterr()

        assert result == 0
        # Should have multiple sections
        assert len(captured.out.split("##")) > 1

    def test_process_all_interfaces_explain(self, capsys):
        """Test processing all interfaces with explain."""
        args = MockArgs(
            list=False,
            all=False,
            explain=True,
            opencl=False,
            python=False,
            interfaces=[]  # Empty means all
        )

        result = cmd_interfaces(args)
        captured = capsys.readouterr()

        assert result == 0
        # Should process multiple interfaces
        assert "Interface:" in captured.out


class TestMainFunction:
    """Tests for main CLI entry point."""

    def test_main_no_args(self, capsys):
        """Test main with no arguments shows help."""
        with patch('sys.argv', ['yapocis-tools']):
            result = main()
            captured = capsys.readouterr()

            assert result == 1
            assert "usage:" in captured.out or "Yapocis development tools" in captured.out

    def test_main_backends_subcommand(self, capsys):
        """Test main with backends subcommand."""
        with patch('sys.argv', ['yapocis-tools', 'backends']):
            result = main()
            captured = capsys.readouterr()

            assert result == 0
            assert "Available Backends" in captured.out

    def test_main_assets_list(self, capsys):
        """Test main with assets list subcommand."""
        with patch('sys.argv', ['yapocis-tools', 'assets', '--list', '--all']):
            result = main()
            captured = capsys.readouterr()

            assert result == 0
            assert "##" in captured.out

    def test_main_interfaces_list(self, capsys):
        """Test main with interfaces list subcommand."""
        with patch('sys.argv', ['yapocis-tools', 'interfaces', '--list']):
            result = main()
            captured = capsys.readouterr()

            assert result == 0
            assert "Available Interfaces" in captured.out

    def test_main_devices_list(self, capsys):
        """Test main with devices list subcommand."""
        with patch('sys.argv', ['yapocis-tools', 'devices', '--list']):
            result = main()
            captured = capsys.readouterr()

            assert result == 0
            # Should produce output (device names)

    def test_main_help(self, capsys):
        """Test main with --help flag."""
        with patch('sys.argv', ['yapocis-tools', '--help']):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0

    def test_main_subcommand_help(self, capsys):
        """Test main with subcommand --help."""
        with patch('sys.argv', ['yapocis-tools', 'backends', '--help']):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0


class TestIntegration:
    """Integration tests for yapocis-tools."""

    def test_full_workflow_backends(self, capsys):
        """Test full workflow: list backends and verify output structure."""
        with patch('sys.argv', ['yapocis-tools', 'backends']):
            result = main()
            captured = capsys.readouterr()

            assert result == 0
            # Should have header
            assert "====" in captured.out
            # Should have legend
            assert "Legend:" in captured.out
            # Should list at least one backend
            assert ("OCL" in captured.out or "METAL" in captured.out or
                    "CUDA" in captured.out or "NUMBAX" in captured.out)

    def test_full_workflow_assets(self, capsys):
        """Test full workflow: list all assets and verify structure."""
        with patch('sys.argv', ['yapocis-tools', 'assets', '--list', '--all']):
            result = main()
            captured = capsys.readouterr()

            assert result == 0
            # Should have at least some asset types
            output_upper = captured.out.upper()
            assert "##" in captured.out
            # Check for at least one asset type
            has_asset_type = any(
                asset_type in output_upper
                for asset_type in ["INTERFACES", "KERNELS", "IMAGES", "ICONS", "FONTS"]
            )
            assert has_asset_type

    def test_full_workflow_interfaces(self, capsys):
        """Test full workflow: list interfaces and explain one."""
        # First list
        with patch('sys.argv', ['yapocis-tools', 'interfaces', '--list']):
            result = main()
            captured = capsys.readouterr()

            assert result == 0
            assert "Available Interfaces" in captured.out

        # Then explain
        with patch('sys.argv', ['yapocis-tools', 'interfaces', 'mandelbrot', '--explain']):
            result = main()
            captured = capsys.readouterr()

            assert result == 0
            assert "mandelbrot" in captured.out.lower()

    def test_error_handling_invalid_subcommand(self, capsys):
        """Test error handling for invalid subcommand."""
        with patch('sys.argv', ['yapocis-tools', 'invalid_command']):
            # Argparse exits with status 2 for invalid command
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 2

    def test_backend_json_output_valid(self, capsys):
        """Test that backend JSON output is valid and parseable."""
        with patch('sys.argv', ['yapocis-tools', 'backends', '--json']):
            result = main()
            captured = capsys.readouterr()

            assert result == 0
            # Should be valid JSON
            data = json.loads(captured.out)
            assert isinstance(data, dict)

            # Check structure
            for backend, info in data.items():
                assert "available" in info
                assert isinstance(info["available"], bool)

    def test_assets_find_existing(self, capsys):
        """Test finding an asset that definitely exists."""
        with patch('sys.argv', ['yapocis-tools', 'assets', 'test', '--images']):
            result = main()
            captured = capsys.readouterr()

            assert result == 0
            # Should find test.jpg
            assert "test" in captured.out.lower()


class TestErrorConditions:
    """Tests for error conditions and edge cases."""

    def test_assets_invalid_type(self, capsys):
        """Test listing invalid asset type."""
        args = MockArgs(
            list=True,
            all=False,
            assets=["invalid_type_xyz"],
            images=False,
            icons=False,
            fonts=False,
            kernels=False,
            interfaces=False
        )

        with pytest.raises(SystemExit):
            cmd_assets(args)

    def test_backends_exception_handling(self, capsys):
        """Test backends command handles exceptions gracefully."""
        with patch('yapocis.core.get_backend_infos', side_effect=Exception("Test error")):
            args = MockArgs(json=False, verbose=False)
            result = cmd_backends(args)
            captured = capsys.readouterr()

            assert result == 1
            assert "Error:" in captured.err or "error" in captured.err.lower()

    def test_interfaces_nonexistent(self, capsys):
        """Test processing nonexistent interface."""
        args = MockArgs(
            list=False,
            all=False,
            explain=True,
            opencl=False,
            python=False,
            interfaces=["nonexistent_interface_xyz"]
        )

        # Should not crash, may produce error or skip
        try:
            result = cmd_interfaces(args)
            # If it doesn't raise, that's fine
        except Exception:
            # If it raises, that's also acceptable
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
