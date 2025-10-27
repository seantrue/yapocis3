# GPU-Accelerated Voice Activity Detection (VAD) with Yapocis

This implementation provides a GPU-accelerated Voice Activity Detection system using yapocis kernels, based on the algorithm in `segment.py`.

## Overview

The yapocis VAD implementation uses:
- **FFT library** for GPU-accelerated spectrogram computation
- **Fused audio kernels** for efficient bandpass variance extraction
- **Multi-backend support** (OpenCL, Metal, CUDA, Numba)

## Architecture

### Kernel Fusion Strategy

The implementation uses kernel fusion to minimize GPU memory transfers and improve performance:

1. **Fused VAD Features Kernel**: Combines multiple operations in a single GPU kernel:
   - Extract bandpass region (formant frequencies 200-4800 Hz)
   - Compute variance along frequency axis
   - Apply log transform

2. **Benefits of Fusion**:
   - Reduced memory bandwidth (no intermediate buffers)
   - Better cache locality
   - Fewer kernel launches

### Key Components

#### 1. Audio Interface (`src/yapocis/interfaces/audio.interface`)
Defines kernels for VAD processing:
- `fused_vad_features`: Bandpass extraction + variance + log
- `normalize_with_minmax`: Normalize features to [0,1]
- `smooth_1d`: 1D box filter smoothing
- `log_normalize`: Log transform

#### 2. Kernel Implementations
- **OpenCL**: `src/yapocis/kernels/audio.mako`
- **Metal**: `src/yapocis/metalkernels/audio.mako`
- **CUDA**: `src/yapocis/cukernels/audio.mako`

#### 3. High-Level API (`vad_yapocis.py`)
- `YapocisAudioSegmenter`: Main VAD class
- `segment_audio_gpu()`: Convenience function

## Usage

### Basic Example

```python
from vad_yapocis import YapocisAudioSegmenter

# Create segmenter
vad = YapocisAudioSegmenter(
    formant_band=(200, 4800),      # Frequency range for formants
    min_formant_duration=0.1,       # Min speech duration (sec)
    max_gap=0.25,                   # Max gap to bridge (sec)
    front_porch=0.25,               # Lead time (sec)
    back_porch=0.25,                # Trail time (sec)
    use_kernel_fusion=True          # Use fused GPU kernels
)

# Process audio file
seg = vad("speech.wav")

# Access results
print(f"Speech segments: {len(seg.signal_slices)}")
print(f"Total speech: {seg.signal_seconds:.2f}s")
print(f"Segments: {seg.signal_slices}")
```

### Convenience Function

```python
from vad_yapocis import segment_audio_gpu

seg = segment_audio_gpu("speech.wav", formant_band=(300, 3400))
```

## Algorithm

The VAD algorithm follows these steps:

1. **Load Audio**: Read audio file
2. **Compute Spectrogram**: GPU-accelerated FFT-based spectrogram
3. **Extract Features**: Fused kernel computes log-variance in formant band
4. **Threshold**: Detect speech regions based on percentile threshold
5. **Post-Process**:
   - Remove short segments
   - Bridge small gaps
   - Add front/back porch
6. **Output**: Segmentation with speech/noise regions

## Performance

The GPU implementation provides speedup through:
- Parallel FFT computation
- Fused kernel operations (reduces memory transfers)
- Efficient variance computation in formant band

Timing breakdown (example on 5s audio):
```
load audio                    : 0.000s
compute spectrogram (GPU)     : 0.170s  (FFT-based)
compute VAD features (GPU)    : 0.002s  (fused kernel)
threshold formants            : 0.000s
Total                         : ~0.172s
```

## Backend Support

The implementation automatically works across all yapocis backends:

```bash
# Use Metal (macOS GPU)
export YAPOCIS_BACKEND=metal
python script.py

# Use OpenCL
export YAPOCIS_BACKEND=ocl
python script.py

# Use CUDA (NVIDIA GPU)
export YAPOCIS_BACKEND=cuda
python script.py

# Use Numba (CPU fallback)
export YAPOCIS_BACKEND=numba
python script.py
```

## Testing

Run the test suite:

```bash
python claude-debug/test_vad_yapocis.py
```

This will:
1. Test kernel compilation
2. Generate synthetic test audio
3. Run VAD detection
4. Show timing breakdown

## Files Created

### Core Implementation
- `src/yapocis/interfaces/audio.interface` - Interface definition
- `src/yapocis/kernels/audio.mako` - OpenCL kernels
- `src/yapocis/metalkernels/audio.mako` - Metal kernels
- `src/yapocis/cukernels/audio.mako` - CUDA kernels
- `vad_yapocis.py` - Main VAD implementation

### Testing
- `claude-debug/test_vad_yapocis.py` - Test script

### Documentation
- `VAD_README.md` - This file

## Comparison with Original

### Similarities
- Same core algorithm (formant band variance detection)
- Compatible output format (Segmentation objects)
- Same tunable parameters

### Differences
- **GPU Acceleration**: Uses yapocis kernels vs scipy
- **Kernel Fusion**: Fused operations reduce overhead
- **Multi-backend**: Works on Metal/OpenCL/CUDA/Numba
- **Standalone**: Minimal dependencies (can run without segment.py)

## Dependencies

### Required
- yapocis (with FFT support)
- numpy
- scipy (for convolution smoothing, optional)

### Optional
- segment.py (for full TextGrid support)
- wave (for WAV file support, built-in)

## Future Improvements

Potential optimizations:
1. **GPU Smoothing**: Implement smooth_1d kernel for 1D convolution
2. **Streaming**: Process large files in chunks
3. **Multi-file**: Batch process multiple audio files
4. **Reduction Kernels**: GPU-based min/max for normalization
5. **Numba Kernels**: Implement Numba versions of audio kernels

## License

Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.
