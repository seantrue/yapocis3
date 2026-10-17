#!/usr/bin/env python3
#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.
"""
Extract and validate code examples from documentation.

This tool finds all Python code blocks in markdown documentation files,
extracts them, and runs them to verify they execute without errors.

Usage:
    yapocis-doctest [docs_path] [options]

Examples:
    # Test all docs in current directory
    yapocis-doctest docs/

    # Test specific file
    yapocis-doctest docs/reference/lib/simplex.md

    # Dry run - show what would be tested
    yapocis-doctest docs/ --dry-run

    # Verbose output
    yapocis-doctest docs/ -v

    # Continue on errors
    yapocis-doctest docs/ --continue-on-error
"""
from __future__ import annotations

import argparse
import ast
import io
import os
import re
import sys
import traceback
from collections.abc import Iterator
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CodeBlock:
    """Represents an extracted code block from documentation."""

    source: str
    file_path: Path
    line_number: int
    language: str = "python"
    is_continuation: bool = False  # True if block continues from previous
    no_exec: bool = False  # True if marked with {.no-exec}

    @property
    def location(self) -> str:
        return f"{self.file_path}:{self.line_number}"


@dataclass
class TestResult:
    """Result of running a code block."""

    block: CodeBlock
    success: bool
    error: str | None = None
    output: str | None = None
    skipped: bool = False
    skip_reason: str | None = None

    @property
    def status(self) -> str:
        """Get result status as string."""
        if self.skipped:
            return "skipped"
        elif self.success:
            return "passed"
        else:
            return "failed"

    @property
    def reason(self) -> str:
        """Get the reason for failure or skip."""
        if self.skipped:
            return self.skip_reason or ""
        elif not self.success:
            return self.error or ""
        return ""


@dataclass
class DocTestStats:
    """Statistics for doc testing run."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    no_exec: int = 0  # Blocks marked with {.no-exec}
    errors: list[TestResult] = field(default_factory=list)
    all_results: list[TestResult] = field(default_factory=list)


def extract_code_blocks(file_path: Path) -> Iterator[CodeBlock]:
    """
    Extract Python code blocks from a markdown file.

    Handles fenced code blocks with ```python or ```py markers.
    Only extracts blocks explicitly marked as Python or with no language specified.
    Detects {.no-exec} attribute to mark blocks that should not be executed.
    """
    content = file_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    # Pattern for any fenced code block opening, capturing language and attributes
    # Matches: ```python, ```python {.no-exec}, ```py {.class}, etc.
    open_fence_pattern = re.compile(r"^```(\w*)?\s*(\{[^}]*\})?\s*$", re.IGNORECASE)
    end_fence_pattern = re.compile(r"^```\s*$")

    in_code_block = False
    code_lines: list[str] = []
    block_start_line = 0
    language = ""
    is_python_block = False
    is_no_exec = False

    for line_num, line in enumerate(lines, start=1):
        if not in_code_block:
            match = open_fence_pattern.match(line)
            if match:
                in_code_block = True
                code_lines = []
                block_start_line = line_num
                lang = match.group(1) or ""
                attrs = match.group(2) or ""
                language = lang.lower()
                # Only process blocks explicitly marked as Python
                # (not unmarked blocks - those are often example output)
                is_python_block = language in ("python", "py", "python3")
                # Check for {.no-exec} attribute
                is_no_exec = ".no-exec" in attrs
        else:
            if end_fence_pattern.match(line):
                in_code_block = False
                if is_python_block and code_lines:
                    source = "\n".join(code_lines)
                    # Skip empty blocks
                    if source.strip():
                        yield CodeBlock(
                            source=source,
                            file_path=file_path,
                            line_number=block_start_line,
                            language=language if language else "python",
                            no_exec=is_no_exec,
                        )
                is_python_block = False
                is_no_exec = False
            elif is_python_block:
                code_lines.append(line)


def contains_graphics_chars_outside_strings(source: str) -> bool:
    """
    Check if source contains graphics/box-drawing characters outside of quoted strings.

    These characters suggest the block is ASCII art or a diagram, not Python code.
    """
    # Box-drawing and other graphics characters that indicate non-code content
    graphics_chars = set(
        "┌┐└┘├┤┬┴┼─│═║╔╗╚╝╠╣╦╩╬"  # Box drawing
        "▀▄█▌▐░▒▓"  # Block elements
        "◀▶▲▼◆◇○●"  # Geometric shapes
        "←→↑↓↔↕"  # Arrows
        "★☆♠♣♥♦"  # Misc symbols
    )

    # Simple state machine to track if we're inside a string
    in_string = False
    string_char = None
    escape_next = False
    i = 0

    while i < len(source):
        char = source[i]

        if escape_next:
            escape_next = False
            i += 1
            continue

        if char == "\\" and in_string:
            escape_next = True
            i += 1
            continue

        # Check for triple quotes
        if not in_string and i + 2 < len(source):
            triple = source[i : i + 3]
            if triple in ('"""', "'''"):
                in_string = True
                string_char = triple
                i += 3
                continue

        # Check for end of triple-quoted string
        if in_string and len(string_char) == 3 and i + 2 < len(source):
            if source[i : i + 3] == string_char:
                in_string = False
                string_char = None
                i += 3
                continue

        # Check for single/double quotes (single char strings)
        if not in_string and char in ('"', "'"):
            in_string = True
            string_char = char
            i += 1
            continue

        if in_string and len(string_char) == 1 and char == string_char:
            in_string = False
            string_char = None
            i += 1
            continue

        # Check for graphics characters outside strings
        if not in_string and char in graphics_chars:
            return True

        i += 1

    return False


def should_skip_block(block: CodeBlock) -> tuple[bool, str | None]:
    """
    Determine if a code block should be skipped.

    Returns (should_skip, reason).
    """
    source = block.source.strip()

    # Check for {.no-exec} marker (tracked separately)
    if block.no_exec:
        return True, "marked {.no-exec}"

    # Check for explicit skip markers
    if (
        "# notest" in source
        or "# doctest: skip" in source
        or "# skip" in source.split("\n")[0]
    ):
        return True, "explicit skip marker"

    # Check for empty blocks first
    if not source or source.isspace():
        return True, "empty block"

    # Check for graphics/box-drawing characters outside strings (ASCII art diagrams)
    if contains_graphics_chars_outside_strings(source):
        return True, "contains graphics/diagram characters"

    # Skip blocks that are clearly not runnable
    # These patterns must match the ENTIRE block to skip
    skip_patterns_full = [
        (r"^\.\.\.\s*$", "ellipsis only"),
        (r"^#[^\n]*$", "comment only"),
    ]

    for pattern, reason in skip_patterns_full:
        if re.match(pattern, source):
            return True, reason

    # These patterns skip if found anywhere in the block
    skip_patterns_partial = [
        (r"^interface\s+\w+\s*\{", "IDL definition"),
        (r"^kernel\s+\w+\(", "kernel signature"),
        (r"^extension\s+\w+", "extension definition"),
        (r"^YAPOCIS_", "environment variable example"),
        (r"^\$\s+", "shell command"),
        (r"^>>>", "doctest format"),
    ]

    for pattern, reason in skip_patterns_partial:
        if re.search(pattern, source, re.MULTILINE):
            return True, reason

    # Class definition fragments (signature + INTERFACE only)
    if re.match(r"^class\s+\w+.*:\s*\n\s+INTERFACE\s*=", source):
        return True, "class definition fragment"

    # Method signature documentation (variable = obj.method( with params on multiple lines)
    # Only skip if it's a short block showing just the method signature
    lines = source.split("\n")
    if len(lines) <= 12 and re.match(r"^\w+\s*=\s*\w+\.\w+\s*\($", lines[0].strip()):
        # Check if most lines are just parameter documentation
        param_lines = sum(
            1 for line in lines[1:] if re.match(r"^\s+\w+.*[,)]?\s*(#.*)?$", line)
        )
        if param_lines >= len(lines) - 2:  # Most lines are params
            return True, "method signature documentation"

    # Function signature documentation (function_name( with params and return type)
    # Patterns like: load_program(\n    param: type,\n    ...\n) -> ReturnType
    if len(lines) <= 15 and re.match(r"^\w+\s*\($", lines[0].strip()):
        # Check if it ends with ) -> Type or just )
        last_line = lines[-1].strip()
        if re.match(r"^\)\s*(->.*)?$", last_line):
            # Check if middle lines are parameter documentation
            # Match: param: type, param=default, *args, **kwargs
            param_lines = sum(
                1
                for line in lines[1:-1]
                if re.match(
                    r"^\s*(\*{0,2}\w+\s*[:=].*|[\*]{1,2}\w+)[,)]?\s*(#.*)?$",
                    line.strip(),
                )
            )
            if param_lines >= len(lines) - 3:  # Most lines are params
                return True, "function signature documentation"

    # Single-line function signature (e.g., "get_backends() -> list[str]")
    if len(lines) == 1:
        line = lines[0].strip()
        # Match: function_name(...) -> return_type  or  function_name()
        if re.match(r"^\w+\s*\([^)]*\)\s*(->.*)?$", line):
            return True, "function signature documentation"

    # Decorator list documentation (multiple @decorator lines without function)
    # Used to show available decorators/markers
    if all(
        re.match(r"^@\w+", line.strip())
        or line.strip().startswith("#")
        or not line.strip()
        for line in lines
        if line.strip()
    ):
        decorator_lines = sum(1 for line in lines if line.strip().startswith("@"))
        if decorator_lines >= 2:
            return True, "decorator/marker documentation"

    # Skip if it starts with continuation markers
    if source.startswith("..."):
        return True, "continuation marker"

    # Try to parse - if it fails, check if it's an intentional fragment
    try:
        ast.parse(source)
    except SyntaxError:
        # Check if it's an intentional fragment with ellipsis placeholders
        if "..." in source:
            return True, "contains ellipsis placeholder"
        # Could be incomplete - check for common patterns
        if source.strip().endswith(",") or source.strip().endswith("\\"):
            return True, "incomplete statement"
        # Check for parameter list fragments (indented lines ending with comma)
        if re.search(r"^\s+\w+.*,$", source, re.MULTILINE):
            # Could be a multi-line method call - those are OK to try
            pass

    return False, None


def create_test_environment() -> dict:
    """
    Create a clean environment for running code blocks.

    Pre-imports common modules that examples might need.
    Ensures access to current Python context including environment variables.
    """
    import builtins

    env = {
        "__name__": "__main__",
        "__doc__": None,
        "__builtins__": builtins,  # Ensure all builtins are available
    }

    # Pre-import os and sys for environment access
    env["os"] = os
    env["sys"] = sys

    # Pre-import numpy since almost all examples use it
    try:
        import numpy as np

        env["np"] = np
        env["numpy"] = np
    except ImportError:
        pass

    return env


def add_mock_functions(env: dict) -> None:
    """Add mock functions and common variables for undefined helpers in examples."""
    import numpy as np

    def mock_load_image(path: str) -> np.ndarray:
        """Mock image loader - returns random test data."""
        return np.random.rand(256, 256).astype(np.float32)

    def mock_load_audio(path: str) -> np.ndarray:
        """Mock audio loader - returns random test data."""
        return np.random.rand(8000).astype(np.float32)

    def mock_save_image(path: str, data: np.ndarray) -> None:
        """Mock image saver - does nothing."""
        pass

    def mock_save_frame(path: str, data: np.ndarray) -> None:
        """Mock frame saver - does nothing."""
        pass

    def mock_display(data) -> None:
        """Mock display - does nothing."""
        pass

    def mock_render(data) -> None:
        """Mock render - does nothing."""
        pass

    def mock_render_cloud(data) -> None:
        """Mock cloud renderer - does nothing."""
        pass

    env.update(
        {
            "load_image": mock_load_image,
            "load_audio": mock_load_audio,
            "save_image": mock_save_image,
            "save_frame": mock_save_frame,
            "display": mock_display,
            "render": mock_render,
            "render_cloud": mock_render_cloud,
        }
    )

    # Add common example variables that documentation often uses without defining
    env.update(
        {
            # Image variables
            "image": np.random.rand(256, 256).astype(np.float32),
            "canvas": np.zeros((512, 512, 4), dtype=np.float32),
            "rgb": np.random.rand(256, 256, 3).astype(np.float32),
            "rgba_image": np.random.rand(256, 256, 4).astype(np.float32),
            "noisy_image": np.random.rand(256, 256).astype(np.float32),
            "original_rgb": np.random.rand(256, 256, 3).astype(np.float32),
            # Signal variables
            "signal": np.random.rand(1024).astype(np.float32),
            "signal1": np.random.rand(1024).astype(np.float32),
            "signal2": np.random.rand(1024).astype(np.float32),
            "audio": np.random.rand(8000).astype(np.float32),
            "noise": np.random.rand(1024).astype(np.float32),
            # FFT variables
            "fft_result": np.random.rand(256).astype(np.complex64),
            # Generic parameters
            "coords": [(100, 100, 50), (200, 200, 30)],
            "locations": [(100, 100), (200, 200)],
            "threshold": 0.1,
            "width": 512,
            "height": 512,
            "depth": 64,
            "reach": 3,
            "iteration": 0,
            "max_iterations": 10,
            "data": np.random.rand(256, 256).astype(np.float32),
            # Complex coordinates for Mandelbrot
            "complex_coords": (
                np.random.rand(256, 256) + 1j * np.random.rand(256, 256)
            ).astype(np.complex64),
        }
    )


def save_yapocis_env() -> dict[str, str | None]:
    """Save all YAPOCIS_* environment variables."""
    saved = {}
    for key in list(os.environ.keys()):
        if key.startswith("YAPOCIS_"):
            saved[key] = os.environ.get(key)
    return saved


def restore_yapocis_env(saved: dict[str, str | None]) -> None:
    """Restore YAPOCIS_* environment variables to saved state."""
    # Remove any YAPOCIS_* vars that were set during execution
    for key in list(os.environ.keys()):
        if key.startswith("YAPOCIS_"):
            if key not in saved:
                del os.environ[key]

    # Restore original values
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def run_code_block(
    block: CodeBlock,
    env: dict | None = None,
    timeout: float = 30.0,
    compile_only: bool = False,
) -> TestResult:
    """
    Execute a code block and return the result.

    Args:
        block: The code block to execute
        env: Environment dict (will be modified)
        timeout: Maximum execution time in seconds
        compile_only: If True, only compile/parse, don't execute

    Returns:
        TestResult with success/failure info
    """
    # Check if should skip
    should_skip, reason = should_skip_block(block)
    if should_skip:
        return TestResult(block=block, success=True, skipped=True, skip_reason=reason)

    if env is None:
        env = create_test_environment()
        add_mock_functions(env)

    # Save environment variables to prevent pollution between blocks
    saved_env = save_yapocis_env()

    # Capture stdout/stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        # Compile first to catch syntax errors with better location info
        try:
            code = compile(block.source, block.location, "exec")
        except SyntaxError as e:
            return TestResult(
                block=block,
                success=False,
                error=f"SyntaxError at line {e.lineno}: {e.msg}",
            )

        # If compile_only, we're done - just return success
        if compile_only:
            return TestResult(block=block, success=True, output=None)

        # Execute with captured output
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, env)

        output = stdout_capture.getvalue()
        return TestResult(block=block, success=True, output=output if output else None)

    except Exception as e:
        # Get formatted traceback
        tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
        # Filter to show relevant parts
        error_msg = f"{type(e).__name__}: {e}"

        return TestResult(block=block, success=False, error=error_msg)

    finally:
        # Restore environment variables to prevent pollution
        restore_yapocis_env(saved_env)


def find_doc_files(path: Path, recursive: bool = True) -> Iterator[Path]:
    """Find all markdown files in the given path."""
    if path.is_file():
        if path.suffix.lower() in (".md", ".markdown", ".rst"):
            yield path
    elif path.is_dir():
        pattern = "**/*.md" if recursive else "*.md"
        yield from path.glob(pattern)
        if recursive:
            yield from path.glob("**/*.markdown")


def run_tests(
    docs_path: Path,
    verbose: bool = False,
    dry_run: bool = False,
    continue_on_error: bool = False,
    file_filter: str | None = None,
    isolated: bool = False,
    compile_only: bool = False,
    progress: bool = False,
) -> DocTestStats:
    """
    Run all code examples from documentation files.

    Args:
        docs_path: Path to docs directory or specific file
        verbose: Print detailed output
        dry_run: Don't run, just show what would be tested
        continue_on_error: Keep running after failures
        file_filter: Only test files matching this pattern
        isolated: Run each block in isolation (no shared state)
        compile_only: Only check syntax/compilation, don't execute
        progress: Show each test before execution

    Returns:
        DocTestStats with results
    """
    stats = DocTestStats()

    # Find files to process
    files = list(find_doc_files(docs_path))
    if file_filter:
        files = [f for f in files if file_filter in str(f)]

    if not files:
        print(f"No documentation files found in {docs_path}")
        return stats

    print(f"Found {len(files)} documentation files")

    # Create shared environment for related blocks
    env = create_test_environment()
    add_mock_functions(env)

    current_file: Path | None = None

    for doc_file in sorted(files):
        blocks = list(extract_code_blocks(doc_file))
        if not blocks:
            continue

        if verbose or dry_run:
            print(f"\n{'='*60}")
            print(f"File: {doc_file}")
            print(f"Found {len(blocks)} code blocks")
            print("=" * 60)

        # Create environment for this file (shared across blocks unless isolated)
        file_env = create_test_environment()
        add_mock_functions(file_env)

        for i, block in enumerate(blocks):
            stats.total += 1

            # In isolated mode, reset environment for each block
            if isolated:
                file_env = create_test_environment()
                add_mock_functions(file_env)

            if dry_run:
                print(f"\n[Block {i+1}] {block.location}")
                print("-" * 40)
                # Show first few lines
                lines = block.source.split("\n")
                preview = "\n".join(lines[:5])
                if len(lines) > 5:
                    preview += f"\n... ({len(lines) - 5} more lines)"
                print(preview)

                should_skip, reason = should_skip_block(block)
                if should_skip:
                    print(f"  [SKIP: {reason}]")
                continue

            # Show progress if requested
            if progress:
                print(f"  RUN  {block.location}", end="", flush=True)

            # Run the block
            result = run_code_block(block, file_env, compile_only=compile_only)
            stats.all_results.append(result)

            # Clear progress line if shown
            if progress:
                # Move cursor back and clear line
                print(f"\r{' ' * 60}\r", end="", flush=True)

            if result.skipped:
                # Track {.no-exec} blocks separately
                if block.no_exec:
                    stats.no_exec += 1
                else:
                    stats.skipped += 1
                if verbose:
                    print(f"  SKIP {block.location}: {result.skip_reason}")
            elif result.success:
                stats.passed += 1
                if verbose:
                    print(f"  PASS {block.location}")
                    if result.output:
                        print(f"       Output: {result.output[:100]}")
            else:
                stats.failed += 1
                stats.errors.append(result)
                print(f"  FAIL {block.location}")
                print(f"       {result.error}")

                if not continue_on_error:
                    print(
                        "\nStopping on first error. Use --continue-on-error to run all."
                    )
                    return stats

    return stats


def write_csv_results(stats: DocTestStats, output_path: Path) -> None:
    """
    Write test results to a CSV file.

    CSV format:
        file,line,status,reason

    Args:
        stats: Test statistics with all results
        output_path: Path to write CSV file
    """
    import csv

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "line", "status", "reason"])

        for result in stats.all_results:
            writer.writerow(
                [
                    str(result.block.file_path),
                    result.block.line_number,
                    result.status,
                    result.reason,
                ]
            )

    print(f"Results written to: {output_path}")


def setup_project_environment() -> Path | None:
    """
    Set up the environment to run from the project root directory.

    This ensures:
    1. Working directory is the project root
    2. src/ directory is in sys.path so yapocis is importable
    3. Environment variables are set to prevent interactive prompts

    Returns:
        Project root path if found, None otherwise
    """
    # Set environment variables to prevent interactive device selection prompts
    # PYOPENCL_CTX: Auto-select first OpenCL context (prevents cl.create_some_context() prompt)
    if "PYOPENCL_CTX" not in os.environ:
        os.environ["PYOPENCL_CTX"] = "0"

    # YAPOCIS_DEVICE: Auto-select first device for yapocis
    if "YAPOCIS_DEVICE" not in os.environ:
        os.environ["YAPOCIS_DEVICE"] = "0"

    # Try to find project root by looking for pyproject.toml or src/yapocis
    cwd = Path.cwd()

    # Check current directory and parents
    for check_dir in [cwd] + list(cwd.parents):
        pyproject = check_dir / "pyproject.toml"
        src_yapocis = check_dir / "src" / "yapocis"

        if pyproject.exists() and src_yapocis.exists():
            # Found project root
            project_root = check_dir
            src_dir = project_root / "src"

            # Add src to sys.path if not already there
            src_str = str(src_dir)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)

            # Change to project root
            os.chdir(project_root)

            return project_root

    return None


def main():
    """Main entry point for yapocis-doctest."""
    parser = argparse.ArgumentParser(
        description="Extract and validate code examples from documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  yapocis-doctest docs/                    # Test all docs
  yapocis-doctest docs/library/            # Test lib reference only
  yapocis-doctest docs/quickstart/index.md # Test specific file
  yapocis-doctest docs/ --dry-run          # Show blocks without running
  yapocis-doctest docs/ -v                 # Verbose output
  yapocis-doctest docs/ --continue-on-error  # Run all despite failures
  yapocis-doctest docs/ --filter simplex   # Only files with 'simplex'
  yapocis-doctest docs/ --compile-only     # Only check syntax
  yapocis-doctest docs/ --csv results.csv  # Output results to CSV
  yapocis-doctest docs/ --progress         # Show test being executed

Note: This script should be run from the project root directory or a subdirectory.
      It will automatically find the project root and set up the environment.
        """,
    )

    parser.add_argument(
        "docs_path",
        nargs="?",
        default="docs",
        help="Path to docs directory or specific markdown file (default: docs)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print detailed output"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show code blocks without running them",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running after failures",
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="Only test files containing this string in path",
    )
    parser.add_argument(
        "--isolated",
        action="store_true",
        help="Run each block in isolation (no shared state between blocks)",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only check syntax/compilation, don't execute code",
    )
    parser.add_argument(
        "--csv",
        type=str,
        metavar="FILE",
        help="Write results to CSV file (columns: file,line,status,reason)",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show each test before execution",
    )

    args = parser.parse_args()

    # Set up project environment (find project root, add src to path)
    project_root = setup_project_environment()
    if project_root:
        print(f"Project root: {project_root}")
        print(f"Working directory: {Path.cwd()}")
    else:
        print("Warning: Could not find project root (pyproject.toml + src/yapocis)")
        print("         yapocis imports may not work correctly")

    # Verify yapocis is importable
    try:
        import yapocis

        print(f"yapocis package: {yapocis.__file__}")
    except ImportError as e:
        print(f"Warning: Cannot import yapocis: {e}")
        print("         Code examples that import yapocis will fail")

    # Show relevant environment variables
    yapocis_vars = {k: v for k, v in os.environ.items() if k.startswith("YAPOCIS_")}
    if yapocis_vars:
        print(f"YAPOCIS environment: {yapocis_vars}")

    print()

    docs_path = Path(args.docs_path)
    if not docs_path.exists():
        print(f"Error: Path does not exist: {docs_path}")
        sys.exit(1)

    mode = "compile-only" if args.compile_only else "execute"
    print(f"Testing documentation examples in: {docs_path} (mode: {mode})")
    print()

    stats = run_tests(
        docs_path,
        verbose=args.verbose,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
        file_filter=args.filter,
        isolated=args.isolated,
        compile_only=args.compile_only,
        progress=args.progress,
    )

    # Print summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total blocks:  {stats.total}")
    print(f"  Passed:        {stats.passed}")
    print(f"  Failed:        {stats.failed}")
    print(f"  Skipped:       {stats.skipped}")
    print(f"  No-exec:       {stats.no_exec}  (marked {{.no-exec}})")

    if stats.errors:
        print()
        print("Failed examples:")
        for result in stats.errors:
            print(f"  - {result.block.location}")
            print(f"    {result.error}")

    # Write CSV if requested
    if args.csv:
        write_csv_results(stats, Path(args.csv))

    # Exit with error code if there were failures
    sys.exit(1 if stats.failed > 0 else 0)


if __name__ == "__main__":
    main()
