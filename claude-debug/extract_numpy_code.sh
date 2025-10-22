#!/bin/bash
# For each numpyx kernel, extract only the NumPy (else) branch of if HAVE_NUMBA blocks

cd /Users/seant/workspace/yapocis3

# List of kernel files to process (excluding mandelbrot which is already done)
kernels="colormath hsi median3x3 zcs demo anisodiff graphics graphics4 truetrace convolve convolvesep operators"

for kernel in $kernels; do
    file="yapocis/numpy/kernels/${kernel}.py"
    if [ -f "$file" ]; then
        echo "Processing $kernel..."
        # Backup
        cp "$file" "$file.bak"
    fi
done

echo "Backups created. Now manually clean each file or use Python script."
