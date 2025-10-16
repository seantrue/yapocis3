# !/usr/bin/env python
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.
"""\
Monitor a directory for new images.
Not efficient.
"""

import os
import sys
from argparse import ArgumentParser

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
backends = ["metal", "ocl", "numba", "cuda", "ref", "test"]
from datetime import datetime
import re

import pygame

# Import watchdog for file system monitoring
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("Warning: watchdog not available, using manual cache invalidation")

from yapocis.fonts import font as find_font
from yapocis.utils.basic import VIEWING_QUEUE

# Import validation framework
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pathlib import Path
from yapocis.utils.validation import (
    promote_to_validation,
    load_validation_reference,
    compare_with_validation,
)

queue = VIEWING_QUEUE
RAW_QUEUE = os.path.join(VIEWING_QUEUE, "raw")
VALIDATION_PATH = os.path.join(VIEWING_QUEUE, "validation")
os.makedirs(RAW_QUEUE, exist_ok=True)
os.makedirs(VALIDATION_PATH, exist_ok=True)

# Keyboard help mapping
KEYBOARD_HELP = {
    "q / Esc":   "Quit",
    "a":         "Previous image",
    "d":         "Next image",
    "w":         "First image (beginning)",
    "e":         "Last image (end)",
    "s":         "Toggle tailing mode (auto-advance on new files)",
    "Backspace": "Delete current image (or validation if viewing validation)",
    "F12":       "Promote to validation reference (agnostic)",
    "Shift+F12": "Force promote to validation (agnostic overwrite)",
    "Ctrl+F12":  "Promote to backend-specific validation",
    "F11":       "Compare with validation reference",
    "F10":       "Show validation info for current image",
    "F9":        "Toggle between test image and validation reference",
    "h":         "Show this help",
}


def print_help(prog: str = None) -> None:
    """Print usage and keyboard-to-action mapping."""
    prog = prog or os.path.basename(sys.argv[0])
    usage = (
        f"Usage: {prog} [queue_dir] [filter_string1] [filter_string2] ... [options]\n\n"
    )
    usage += "Arguments:\n"
    usage += (
        "  queue_dir        Directory to watch for images (default: VIEWING_QUEUE)\n"
    )
    usage += (
        "  filter_strings   Show only images whose names contain ALL of these strings\n"
    )
    usage += "                   (case-insensitive, e.g., 'graphics text metal')\n\n"
    usage += "Options:\n"
    usage += "  --validation-failed    Show only images that fail validation\n"
    usage += "  --no-validation        Show only images without validation data\n"
    usage += "  --help, -h             Show this help message\n\n"
    details = ["Keyboard shortcuts:"]
    for k, v in KEYBOARD_HELP.items():
        details.append(f"  {k:16s} -> {v}")
    print(usage + "\n".join(details))
    sys.exit(0)


def parse_image_name(image_path):
    """
    Parse an image path to extract test name and backend.

    Strips out [ref]/[test] qualifiers and backend qualifiers like [ocl].

    Args:
        image_path: Path to the image file

    Returns:
        tuple: (test_name, backend) or (None, None) if parsing fails
    """
    if not image_path:
        return None, None

    filename = os.path.basename(image_path)

    # Remove file extension
    name = os.path.splitext(filename)[0]

    # Extract backend from the last [backend] pattern (actual backend like ocl, metal, numba)
    backend_match = re.search(r"\[(\w+)\]$", name)
    backend = None
    if backend_match:
        potential_backend = backend_match.group(1)
        # Only treat as backend if it's a known backend (not ref/test)
        if potential_backend.lower() in ["ocl", "metal", "numba", "cuda", "ref"]:
            backend = potential_backend

    # Remove all qualifiers: [ref], [test], and backend qualifiers like [ocl]
    test_name = re.sub(r"\s*\[(?:ref|test|\w+)\]", "", name)
    test_name = test_name.strip()

    return test_name, backend


def get_npz_path(image_path, target_dir=None):
    """Get corresponding .npz file path for an image."""
    if not image_path:
        return None

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    npz_name = f"{base_name}.npz"

    if target_dir:
        return os.path.join(target_dir, npz_name)
    else:
        # Default to raw directory
        return os.path.join(RAW_QUEUE, npz_name)


status_message = status_color = status_time = None


def show_status_message(font, message, color=(255, 255, 0)):
    """Show a temporary status message (will be displayed by main loop)."""
    global status_message, status_color, status_timer
    status_message = message
    status_color = color
    status_timer = pygame.time.get_ticks()
    print(f"Status: {message}")  # Also print to console


# Global variables for status display
status_message = None
status_color = (255, 255, 0)
status_timer = 0
STATUS_DISPLAY_TIME = 3000  # 3 seconds in milliseconds

# Global variable for validation view toggle
showing_validation_reference = False


def get_images(filter_strings=None, force_refresh=False):
    """
    Get list of images, optionally filtered by name strings and validation status.
    Uses caching to improve performance when validation filters are active.

    Args:
        filter_strings: list of strings. Only show images whose names contain ALL of these strings.
                       If None or empty, show all images.
        force_refresh: If True, ignore cache and rebuild the image list.

    Returns:
        list of (creation_time, path) tuples sorted by creation time
    """
    global cached_images, images_cache_dirty, last_filter_strings, last_validation_filters
    global show_validation_failed, show_no_validation

    # Determine current filter state
    current_filter_strings = filter_strings or []
    current_validation_filters = (show_validation_failed, show_no_validation)

    # Check if we need to refresh the cache
    filters_changed = (
            current_filter_strings != last_filter_strings
            or current_validation_filters != last_validation_filters
    )

    if (
            not force_refresh
            and not images_cache_dirty
            and not filters_changed
            and cached_images
    ):
        # Return cached results if valid
        return cached_images

    print(
        "Refreshing image list..."
        if filters_changed or images_cache_dirty
        else "Building image list..."
    )

    # Build fresh image list
    images = [
        os.path.join(queue, name)
        for name in os.listdir(queue)
        if name.endswith(".png") or name.endswith(".pgm")
    ]

    # Apply name-based filtering if filter strings are provided
    if current_filter_strings:
        filtered_images = []
        for path in images:
            filename = os.path.basename(path)
            # Check if filename contains ALL filter strings (case-insensitive)
            if all(
                    filter_str.lower() in filename.lower()
                    for filter_str in current_filter_strings
            ):
                filtered_images.append(path)
        images = filtered_images

    # Apply validation-based filtering (expensive operation)
    if show_validation_failed or show_no_validation:
        validation_filtered = []
        total_images = len(images)
        for i, path in enumerate(images):
            # Show progress for large directories
            if i % 20 == 0 and total_images > 50:
                print(f"  Checking validation: {i}/{total_images}")

            validation_text, validation_color = get_validation_status(path)

            # Check validation failed filter
            if show_validation_failed:
                # Include if validation exists but fails (red color indicates failure)
                if validation_color == (255, 0, 0) and validation_text.startswith("[✗"):
                    validation_filtered.append(path)

            # Check no validation filter
            if show_no_validation:
                # Include if no validation reference exists (orange color)
                if (
                        validation_color == (255, 165, 0)
                        and validation_text == "[no validation]"
                ):
                    validation_filtered.append(path)

        images = validation_filtered

    # Sort by creation time
    images = [(os.stat(path).st_ctime, path) for path in images]
    images.sort()

    # Update cache
    cached_images = images
    images_cache_dirty = False
    last_filter_strings = current_filter_strings.copy()
    last_validation_filters = current_validation_filters

    print(f"Found {len(images)} images matching filters")
    return images


def invalidate_image_cache():
    """Mark the image cache as dirty to force refresh on next access."""
    global images_cache_dirty
    images_cache_dirty = True


most_recent = None
image_index = -1
images = []
# Global filter strings for image name filtering
filter_strings = []

# Global validation filter flags
show_validation_failed = False
show_no_validation = False

# Global cached image list and dirty flag for performance
cached_images = []
images_cache_dirty = True
last_filter_strings = []
last_validation_filters = (False, False)

# Watchdog file system observer
file_observer = None
# Flag to signal when tailing mode should advance to next image
tailing_advance_needed = False


class ImageWatcherFileHandler(FileSystemEventHandler):
    """Handle file system events for automatic cache invalidation."""

    def __init__(self, image_dir, validation_dir):
        super().__init__()
        self.image_dir = image_dir
        self.validation_dir = validation_dir

    def on_created(self, event):
        """Handle file creation events."""
        global tailing_advance_needed
        if not event.is_directory:
            if self._is_image_file(event.src_path) or self._is_validation_file(
                    event.src_path
            ):
                print(f"File created: {os.path.basename(event.src_path)}")
                invalidate_image_cache()
                # Signal that tailing mode should advance
                if self._is_image_file(event.src_path):
                    tailing_advance_needed = True
                    # Post a pygame event to trigger step_image in the main loop
                    try:
                        pygame.event.post(pygame.event.Event(pygame.USEREVENT))
                    except pygame.error:
                        # pygame might not be initialized yet
                        pass

    def on_deleted(self, event):
        """Handle file deletion events."""
        if not event.is_directory:
            if self._is_image_file(event.src_path) or self._is_validation_file(
                    event.src_path
            ):
                print(f"File deleted: {os.path.basename(event.src_path)}")
                invalidate_image_cache()

    def on_moved(self, event):
        """Handle file move/rename events."""
        global tailing_advance_needed
        if not event.is_directory:
            if (
                    self._is_image_file(event.src_path)
                    or self._is_image_file(event.dest_path)
                    or self._is_validation_file(event.src_path)
                    or self._is_validation_file(event.dest_path)
            ):
                print(
                    f"File moved: {os.path.basename(event.src_path)} -> {os.path.basename(event.dest_path)}"
                )
                invalidate_image_cache()
                # Signal that tailing mode should advance if moving TO an image file
                if self._is_image_file(event.dest_path):
                    tailing_advance_needed = True
                    # Post a pygame event to trigger step_image in the main loop
                    try:
                        pygame.event.post(pygame.event.Event(pygame.USEREVENT))
                    except pygame.error:
                        # pygame might not be initialized yet
                        pass

    def _is_image_file(self, path):
        """Check if file is an image file."""
        return path.endswith(".png") or path.endswith(".pgm")

    def _is_validation_file(self, path):
        """Check if file is a validation file."""
        return path.endswith(".npz") and self.validation_dir in path


def setup_file_monitoring():
    """set up file system monitoring using watchdog."""
    global file_observer

    if not WATCHDOG_AVAILABLE:
        print("Watchdog not available, file monitoring disabled")
        return

    try:
        # set up file system observer
        file_observer = Observer()
        event_handler = ImageWatcherFileHandler(queue, VALIDATION_PATH)

        # Watch the image directory
        file_observer.schedule(event_handler, queue, recursive=False)

        # Watch the validation directory if it exists
        if os.path.exists(VALIDATION_PATH):
            file_observer.schedule(event_handler, VALIDATION_PATH, recursive=False)

        file_observer.start()
        print(f"File monitoring started for {queue} and {VALIDATION_PATH}")

    except Exception as e:
        print(f"Failed to start file monitoring: {e}")
        file_observer = None


def stop_file_monitoring():
    """Stop file system monitoring."""
    global file_observer

    if file_observer:
        file_observer.stop()
        file_observer.join()
        file_observer = None
        print("File monitoring stopped")


def image_path(i=None):
    global image_index, images
    i = image_index if i is None else i

    # Get cached image list (this will only rebuild if cache is dirty or filters changed)
    current_images = get_images(filter_strings)
    images[:] = current_images  # Update global images list for compatibility

    if len(images) == 0:
        return None, None

    # Clamp index to avoid wrap-around
    if i < 0:
        i = 0
    elif i >= len(images):
        i = len(images) - 1

    created, path = images[i]
    return created, path


def get_validation_status(image_path):
    """
    Get validation status for an image.

    Returns:
        tuple: (status_text, status_color) where status_text describes validation state
    """
    test_name, backend = parse_image_name(image_path)
    if not test_name:
        return "", (255, 255, 255)

    # Check if validation reference exists
    reference = load_validation_reference(test_name, backend)
    if reference is None:
        return "[no validation]", (255, 165, 0)  # Orange

    # Check if current image matches validation
    npz_path = get_npz_path(image_path)
    if not os.path.exists(npz_path):
        return "[validation exists, no .npz]", (255, 255, 0)  # Yellow

    try:
        import numpy as np

        current_data = np.load(npz_path)
        current_array = (
            current_data["arr_0"]
            if "arr_0" in current_data
            else current_data[list(current_data.keys())[0]]
        )

        matches, info = compare_with_validation(test_name, current_array, backend)

        if matches:
            return "[✓ validation match]", (0, 255, 0)  # Green
        else:
            status = info["status"]
            if status == "numerical_mismatch":
                max_diff = info.get("max_abs_diff", 0)
                return f"[✗ differs by {max_diff:.2e}]", (255, 0, 0)  # Red
            else:
                return f"[✗ {status}]", (255, 0, 0)  # Red

    except Exception as e:
        return f"[validation error: {str(e)[:20]}]", (255, 0, 0)  # Red


def fetch_image(i):
    global image_index
    global most_recent
    created, path = image_path(i)
    if created is None or path is None:
        return None, None
    try:
        created = str(datetime.fromtimestamp(created)).split(".")[0]
        image = pygame.image.load(path)
    except pygame.error:
        return None, None
    if max(image.get_size()) > WIDTH:
        image = pygame.transform.smoothscale(image, (WIDTH - 50, HEIGHT - 50))
    image_name, _ = os.path.splitext(os.path.basename(path))

    # Resolve the actual in-bounds index for this path
    try:
        resolved_index = next(idx for idx, (_, p) in enumerate(images) if p == path)
    except StopIteration:
        # Fallback: clamp the requested index if the path isn't found for any reason
        resolved_index = (
            max(0, min(i if i is not None else 0, len(images) - 1)) if images else -1
        )

    # Get validation status
    validation_text, validation_color = get_validation_status(path)

    # Update globals using the resolved index so we never store an out-of-bounds value
    image_index = resolved_index if resolved_index >= 0 else image_index
    most_recent = path

    # Include validation status in the label (use resolved index)
    display_index = (resolved_index + 1) if resolved_index >= 0 else 0
    base_label = f"{display_index}/{len(images)}@{created}  {image_name}"
    if validation_text:
        full_label = f"{base_label}  {validation_text}"
        return image, (full_label, validation_color)
    else:
        return image, (base_label, (255, 255, 255))


def next_image():
    global showing_validation_reference
    showing_validation_reference = False  # Reset to test image when navigating
    return fetch_image(image_index + 1)


def previous_image():
    global showing_validation_reference
    showing_validation_reference = False  # Reset to test image when navigating
    return fetch_image(image_index - 1)


def last_image():
    global showing_validation_reference, images
    showing_validation_reference = False  # Reset to test image when navigating

    # Ensure we compute the current last index without relying on negative indexing
    current_images = get_images(filter_strings)
    images[:] = current_images
    if not images:
        return None, None
    return fetch_image(len(images) - 1)


def first_image():
    global showing_validation_reference
    showing_validation_reference = False  # Reset to test image when navigating
    return fetch_image(0)


def step_image():
    global showing_validation_reference, images, image_index, tailing_advance_needed
    showing_validation_reference = False  # Reset to test image when auto-advancing

    # Only proceed if watchdog detected a new file
    if not tailing_advance_needed:
        return fetch_image(image_index)

    # Clear the flag
    tailing_advance_needed = False

    # Force refresh to get the new images (cache was invalidated by watchdog)
    current_images = get_images(filter_strings, force_refresh=True)
    old_length = len(images)
    images[:] = current_images

    if not images:
        # Nothing to show
        return None, None

    # If new images arrived, advance to the next one
    if len(images) > old_length:
        # New images detected - move to the next image after our current position
        return fetch_image(image_index + 1)

    # If there are more images ahead in the current list, move forward by one
    if image_index < len(images) - 1:
        return fetch_image(image_index + 1)

    # Stay on the current image (shouldn't normally reach here since we only call when new files arrive)
    return fetch_image(image_index)


def delete_image(i=None):
    global most_recent
    if i is None:
        path = most_recent
    else:
        _, path = image_path(i)
        most_recent = path
    if path and os.path.exists(path):
        os.unlink(path)
    # Cache invalidation now handled automatically by watchdog


def promote_current_image(font, force_overwrite=False, backend_specific=False):
    """
    Promote current image to validation reference using the new framework.

    Args:
        font: Pygame font for status messages
        force_overwrite: If True, overwrite existing validation references
        backend_specific: If True, create backend-specific reference (default: agnostic)
    """
    global most_recent
    if not most_recent:
        show_status_message(font, "No current image", (255, 0, 0))
        return

    test_name, backend = parse_image_name(most_recent)
    if not test_name:
        show_status_message(font, "Cannot parse image name", (255, 0, 0))
        return

    # Get the npz file path directly from the image path (same base name, different extension)
    npz_path = get_npz_path(most_recent)
    if not npz_path or not os.path.exists(npz_path):
        show_status_message(font, "No .npz file found", (255, 0, 0))
        return

    try:
        # Check if validation reference already exists
        existing_ref = load_validation_reference(test_name, backend)
        if existing_ref is not None and not force_overwrite:
            ref_type = "backend-specific" if backend_specific else "agnostic"
            show_status_message(
                font, f"Validation exists for {test_name} ({ref_type})", (255, 165, 0)
            )
            return

        # Try to promote using the validation framework, passing the actual source file
        success = promote_to_validation(
            test_name,
            backend,
            source_file=Path(npz_path),
            backend_specific=backend_specific,
        )
        if success:
            status = (
                "promoted (overwritten)" if existing_ref is not None else "promoted"
            )
            ref_type = f"[{backend}]" if backend_specific else "(agnostic)"
            show_status_message(font, f"{test_name} {ref_type} {status}", (0, 255, 0))
        else:
            show_status_message(font, f"Failed to promote {test_name}", (255, 0, 0))

    except Exception as e:
        show_status_message(font, f"Error promoting: {str(e)[:50]}", (255, 0, 0))


def compare_with_validation_reference(font):
    """Compare current image with its validation reference."""
    global most_recent
    if not most_recent:
        show_status_message(font, "No current image", (255, 0, 0))
        return

    test_name, backend = parse_image_name(most_recent)
    if not test_name:
        show_status_message(font, "Cannot parse image name", (255, 0, 0))
        return

    # Get the raw .npz file
    npz_path = get_npz_path(most_recent)
    if not os.path.exists(npz_path):
        show_status_message(font, "No .npz file found", (255, 0, 0))
        return

    try:
        # Load current result
        import numpy as np

        current_data = np.load(npz_path)
        current_array = (
            current_data["arr_0"]
            if "arr_0" in current_data
            else current_data[list(current_data.keys())[0]]
        )

        # Compare with validation reference
        matches, info = compare_with_validation(test_name, current_array, backend)

        if matches:
            show_status_message(font, f"✓ Matches validation reference", (0, 255, 0))
        else:
            status = info["status"]
            if status == "no_reference":
                show_status_message(
                    font, "No validation reference found", (255, 165, 0)
                )
            elif status == "numerical_mismatch":
                max_diff = info.get("max_abs_diff", 0)
                show_status_message(font, f"✗ Differs by {max_diff:.2e}", (255, 0, 0))
            else:
                show_status_message(font, f"✗ {status}", (255, 0, 0))

    except Exception as e:
        show_status_message(font, f"Compare error: {str(e)[:40]}", (255, 0, 0))


def show_validation_info(font):
    """Show validation information for current image."""
    global most_recent
    if not most_recent:
        show_status_message(font, "No current image", (255, 0, 0))
        return

    test_name, backend = parse_image_name(most_recent)
    if not test_name:
        show_status_message(font, "Cannot parse image name", (255, 0, 0))
        return

    try:
        # Check if validation reference exists
        reference = load_validation_reference(test_name, backend)
        if reference is not None:
            shape_str = "x".join(map(str, reference.shape))
            show_status_message(
                font, f"Validation: {shape_str} {reference.dtype}", (0, 255, 255)
            )
        else:
            show_status_message(font, f"No validation reference", (255, 165, 0))

    except Exception as e:
        show_status_message(font, f"Info error: {str(e)[:40]}", (255, 0, 0))


def view_validation_reference(font):
    """Load and display the validation reference as an image."""
    global most_recent, image_index, images
    if not most_recent:
        show_status_message(font, "No current image", (255, 0, 0))
        return None, None

    test_name, backend = parse_image_name(most_recent)
    if not test_name:
        show_status_message(font, "Cannot parse image name", (255, 0, 0))
        return None, None

    try:
        # Load validation reference
        import numpy as np

        reference = load_validation_reference(test_name, backend)
        if reference is None:
            show_status_message(font, "No validation reference found", (255, 165, 0))
            return None, None

        # Convert numpyx array to pygame surface
        # Handle different array shapes and formats
        if len(reference.shape) == 2:
            # 2D array - display as grayscale (use as single channel, replicate to RGB)
            ref_rgb = np.stack([reference, reference, reference], axis=-1)
        elif len(reference.shape) == 3:
            if reference.shape[2] == 1:
                # Single channel - convert to RGB
                ref_rgb = np.repeat(reference, 3, axis=2)
            elif reference.shape[2] == 2:
                # 2 channels - use channel 0 as grayscale
                channel_0 = reference[:, :, 0]
                ref_rgb = np.stack([channel_0, channel_0, channel_0], axis=-1)
            elif reference.shape[2] == 3:
                # RGB
                ref_rgb = reference
            elif reference.shape[2] == 4:
                # RGBA - use RGB channels only for display
                ref_rgb = reference[:, :, :3]
            else:
                show_status_message(
                    font, f"Unsupported array shape: {reference.shape}", (255, 0, 0)
                )
                return None, None
        else:
            show_status_message(
                font, f"Unsupported array shape: {reference.shape}", (255, 0, 0)
            )
            return None, None

        # Normalize to 0-255 range if needed
        if ref_rgb.dtype == np.float32 or ref_rgb.dtype == np.float64:
            if ref_rgb.max() <= 1.0:
                # Values in 0-1 range
                ref_rgb = (ref_rgb * 255).astype(np.uint8)
            else:
                # Values already in 0-255+ range
                ref_rgb = np.clip(ref_rgb, 0, 255).astype(np.uint8)
        else:
            ref_rgb = ref_rgb.astype(np.uint8)

        # Create pygame surface from numpyx array
        validation_surface = pygame.surfarray.make_surface(
            ref_rgb.swapaxes(0, 1)
        )  # pygame expects (width, height)

        # Scale if too large
        if max(validation_surface.get_size()) > WIDTH:
            validation_surface = pygame.transform.smoothscale(
                validation_surface, (WIDTH - 50, HEIGHT - 50)
            )

        ref_type = (
            "agnostic"
            if load_validation_reference(test_name, None) is not None
            else f"[{backend}]"
        )
        label = f"VALIDATION REF ({ref_type}): {test_name}"
        show_status_message(
            font, f"Showing validation reference ({ref_type})", (0, 255, 255)
        )

        return validation_surface, label

    except Exception as e:
        show_status_message(
            font, f"Error loading validation: {str(e)[:40]}", (255, 0, 0)
        )
        return None, None


def toggle_validation_view(font):
    """Toggle between showing test image and validation reference."""
    global showing_validation_reference

    if not showing_validation_reference:
        # Switch to validation reference
        validation_image, validation_label = view_validation_reference(font)
        if validation_image is not None:
            showing_validation_reference = True
            return validation_image, (validation_label, (0, 255, 255))
        else:
            # Failed to load validation, stay with test image
            showing_validation_reference = False
            return fetch_current_test_image()
    else:
        # Switch back to test image
        showing_validation_reference = False
        show_status_message(font, "Showing test image", (0, 255, 0))
        return fetch_current_test_image()


def fetch_current_test_image():
    """Fetch current test image without triggering image list refresh."""
    global most_recent, image_index
    if most_recent and image_index >= 0:
        # Use the already-loaded image information to avoid circular calls
        try:
            created = os.stat(most_recent).st_ctime
            created = str(datetime.fromtimestamp(created)).split(".")[0]
            image = pygame.image.load(most_recent)

            if max(image.get_size()) > WIDTH:
                image = pygame.transform.smoothscale(image, (WIDTH - 50, HEIGHT - 50))

            image_name, _ = os.path.splitext(os.path.basename(most_recent))

            # Get validation status for current image
            validation_text, validation_color = get_validation_status(most_recent)

            # Include validation status in the label
            base_label = f"{image_index + 1}/{len(images)}@{created}  {image_name}"
            if validation_text:
                full_label = f"{base_label}  {validation_text}"
                return image, (full_label, validation_color)
            else:
                return image, (base_label, (255, 255, 255))

        except (pygame.error, OSError):
            # Fall back to regular fetch if there's an error
            return fetch_image(image_index)
    else:
        # Fall back to regular fetch if no current image
        return fetch_image(image_index)


def delete_validation_reference(font):
    """Delete the validation reference for the current image."""
    global most_recent, showing_validation_reference
    if not most_recent:
        show_status_message(font, "No current image", (255, 0, 0))
        return False

    test_name, backend = parse_image_name(most_recent)
    if not test_name:
        show_status_message(font, "Cannot parse image name", (255, 0, 0))
        return False

    try:
        import os
        from pathlib import Path

        # Check for both agnostic and backend-specific validation references
        validation_dir = Path(VALIDATION_PATH)
        agnostic_file = validation_dir / f"{test_name}.npz"
        backend_specific_file = (
            validation_dir / f"{test_name} [{backend}].npz" if backend else None
        )

        deleted_files = []

        # Delete agnostic reference if it exists
        if agnostic_file.exists():
            agnostic_file.unlink()
            deleted_files.append("agnostic")

        # Delete backend-specific reference if it exists
        if backend_specific_file and backend_specific_file.exists():
            backend_specific_file.unlink()
            deleted_files.append(f"[{backend}]")

        if deleted_files:
            deleted_str = " & ".join(deleted_files)
            show_status_message(
                font, f"Deleted validation ({deleted_str})", (255, 255, 0)
            )

            # If we were showing validation reference, switch back to test image
            if showing_validation_reference:
                showing_validation_reference = False
                return True  # Indicate we need to refresh the display

            return False  # No display refresh needed
        else:
            show_status_message(font, "No validation reference found", (255, 165, 0))
            return False

    except Exception as e:
        show_status_message(
            font, f"Error deleting validation: {str(e)[:30]}", (255, 0, 0)
        )
        return False


def show_help_overlay(font):
    """Show keyboard help overlay."""
    show_status_message(font, "Help: Check console for full shortcuts", (0, 255, 255))
    print("\n" + "=" * 50)
    print("IMAGEWATCHER KEYBOARD SHORTCUTS")
    print("=" * 50)
    for key, desc in KEYBOARD_HELP.items():
        print(f"  {key:16s} -> {desc}")
    print("=" * 50 + "\n")


WIDTH = 1024
HEIGHT = 800


def main():
    global WIDTH, HEIGHT, status_message, queue, RAW_QUEUE, VALIDATION_PATH
    global filter_strings, show_validation_failed, show_no_validation, tailing_mode

    # Parse command-line arguments
    parser = ArgumentParser(description="Monitor a directory for new images")
    parser.add_argument(
        "--queue",
        default=VIEWING_QUEUE,
        help=f"Queue directory to watch (default: {VIEWING_QUEUE})",
    )
    parser.add_argument(
        "--validation-failed",
        action="store_true",
        help="Show only images that failed validation",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Show only images without validation references",
    )
    parser.add_argument(
        "filters", nargs="*", help="Filter images by name (substring match)"
    )
    args = parser.parse_args()

    # Update global variables based on arguments
    queue = args.queue
    RAW_QUEUE = os.path.join(queue, "raw")
    VALIDATION_PATH = os.path.join(queue, "validation")
    os.makedirs(RAW_QUEUE, exist_ok=True)
    os.makedirs(VALIDATION_PATH, exist_ok=True)

    filter_strings = args.filters
    show_validation_failed = args.validation_failed
    show_no_validation = args.no_validation

    # Print initial status
    status_parts = []
    if filter_strings:
        status_parts.append(f"name filters: {', '.join(filter_strings)}")

    if show_validation_failed:
        status_parts.append("validation: failed only")

    if show_no_validation:
        status_parts.append("validation: none only")

    if status_parts:
        print(f"Filtering images by: {' | '.join(status_parts)}")

    print(f"Watching directory: {queue}")

    # set up file system monitoring
    setup_file_monitoring()

    # Track tailing mode state
    tailing_mode = False

    pygame.init()
    font_path = find_font("default")
    font = pygame.font.Font(font_path, 22)

    background_color = (100, 100, 100)
    display_surface = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

    # Update window caption to show active filters
    caption = "Image Watcher"
    filter_parts = []

    if filter_strings:
        filter_parts.append(f"name: {', '.join(filter_strings)}")

    if show_validation_failed:
        filter_parts.append("validation: failed only")

    if show_no_validation:
        filter_parts.append("validation: none only")

    if filter_parts:
        caption += f" (filtering: {' | '.join(filter_parts)})"

    pygame.display.set_caption(caption)

    pygame.key.set_repeat(150)
    image, label_info = next_image()
    while True:
        if image is not None:
            display_surface.fill(background_color)
            w, h = image.get_size()
            display_surface.blit(
                image, (25 + WIDTH // 2 - w // 2, 25 + HEIGHT // 2 - h // 2)
            )
            if label_info is not None:
                # Handle both old string format and new tuple format for compatibility
                if isinstance(label_info, tuple):
                    label_text, label_color = label_info
                else:
                    label_text = label_info
                    label_color = (255, 255, 255)

                label = font.render(label_text, True, label_color, (0, 0, 0))
                display_surface.blit(label, (0, 0))

            # Display status message if present and within time limit
            if status_message is not None:
                current_time = pygame.time.get_ticks()
                if current_time - status_timer < STATUS_DISPLAY_TIME:
                    status_surface = font.render(
                        status_message, True, status_color, (0, 0, 0)
                    )
                    # Position status message below the main label
                    y_pos = 25 if label_info is not None else 0
                    display_surface.blit(status_surface, (0, y_pos))
                else:
                    # Clear expired status message
                    status_message = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.VIDEORESIZE:
                # Resize window and image to match
                dim = min(event.w, event.h)
                WIDTH = HEIGHT = dim
                display_surface = pygame.display.set_mode(
                    (dim + 50, dim + 50), pygame.RESIZABLE
                )
                image, label_info = fetch_image(image_index)
            if event.type == pygame.USEREVENT:
                # Only respond to USEREVENT if in tailing mode
                if tailing_mode:
                    image, label_info = step_image()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    return
                if event.key == pygame.K_F12:
                    # Check for Shift modifier for force overwrite
                    force_overwrite = bool(event.mod & pygame.KMOD_SHIFT)
                    # Check for Ctrl modifier for backend-specific
                    backend_specific = bool(event.mod & pygame.KMOD_CTRL)
                    promote_current_image(font, force_overwrite, backend_specific)
                    image, label_info = next_image()
                elif event.key == pygame.K_F11:
                    # Compare with validation reference
                    compare_with_validation_reference(font)
                elif event.key == pygame.K_F10:
                    # Show validation info
                    show_validation_info(font)
                elif event.key == pygame.K_F9:
                    # Toggle between test image and validation reference
                    image, label_info = toggle_validation_view(font)
                elif event.key == pygame.K_h:
                    # Show help overlay
                    show_help_overlay(font)
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                elif event.key == pygame.K_a:
                    # Backup - disable tailing when manually navigating
                    tailing_mode = False
                    image, label_info = previous_image()
                elif event.key == pygame.K_d:
                    # Forward - disable tailing when manually navigating
                    tailing_mode = False
                    image, label_info = next_image()
                elif event.key == pygame.K_e:
                    # End - disable tailing when manually navigating
                    tailing_mode = False
                    image, label_info = last_image()
                elif event.key == pygame.K_w:
                    # Beginning - disable tailing when manually navigating
                    tailing_mode = False
                    image, label_info = first_image()
                elif event.key == pygame.K_s:
                    # Toggle tailing mode
                    tailing_mode = not tailing_mode
                    if tailing_mode:
                        show_status_message(font, "Tailing enabled - waiting for new files", (0, 255, 0))
                        print("Tailing mode enabled - will auto-advance when new files are detected")
                    else:
                        show_status_message(font, "Tailing disabled", (255, 255, 0))
                        print("Tailing mode disabled")
                elif event.key == pygame.K_BACKSPACE:
                    # Delete validation reference if showing validation, otherwise delete test image
                    if showing_validation_reference:
                        # Delete validation reference and refresh display
                        if delete_validation_reference(font):
                            # Validation was deleted and we were showing it, refresh to test image
                            image, label_info = fetch_image(image_index)
                    else:
                        # Delete the current test image
                        delete_image()
                        image, label_info = next_image()
        pygame.display.update()


if __name__ == "__main__":
    try:
        main()
    except (pygame.error, KeyboardInterrupt):
        pass
    finally:
        stop_file_monitoring()
