#!/bin/bash
# Quick demo launcher for difference visualization

echo "=========================================="
echo "Difference Visualization Demo"
echo "=========================================="
echo ""

# Check if test data exists
if [ ! -f "queue/validation/graphics_test.npz" ]; then
    echo "Creating test data..."
    python claude-debug/demo_diff_view.py
    echo ""
fi

echo "Test data ready!"
echo ""
echo "Launching imagewatcher..."
echo ""
echo "Instructions:"
echo "  - Press F8 to toggle difference view"
echo "  - Press a/d to navigate between images"
echo "  - Press h for help"
echo "  - Press q to quit"
echo ""
echo "Look for these test images:"
echo "  - graphics_test [test]"
echo "  - gradient_test [test]"
echo "  - color_squares [test]"
echo "  - float_noise [test]"
echo "  - alpha_gradient [test]"
echo ""
read -p "Press Enter to launch imagewatcher..."

yapocis-imagewatcher
