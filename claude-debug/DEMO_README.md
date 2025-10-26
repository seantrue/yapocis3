# VAD Visualization Demo

This demo shows how to use the GPU-accelerated VAD (Voice Activity Detection) system to segment audio and create visualizations.

## Quick Start

### Run on testsnr.wav

```bash
# Using the demo script
./claude-debug/demo_testsnr.sh

# Or directly
python claude-debug/demo_vad_visualization.py testsnr.wav
```

### Run on any WAV file

```bash
python claude-debug/demo_vad_visualization.py your_audio.wav
```

## Output

The demo creates a comprehensive visualization showing:

1. **Waveform with Speech/Silence Regions**
   - Green shaded areas = detected speech
   - Red shaded areas = detected silence

2. **Spectrogram with Formant Band**
   - Color-coded frequency vs time representation
   - Red dashed lines show formant band (200-4800 Hz by default)

3. **VAD Features (Log-variance)**
   - Shows the computed log-variance in formant band
   - Red dashed line shows threshold
   - Above threshold = speech, below = silence

4. **Binary Speech Detection**
   - Simple binary view of detected segments

## Example Results (testsnr.wav)

```
Sample rate: 16000 Hz
Duration: 20.35 seconds
Samples: 325632

Segmentation results:
  Speech segments: 5
  Total speech: 16.17s (79.5%)
  Total silence: 4.18s (20.5%)

Detected segments:
  1. 0.000s - 0.762s  (duration: 0.762s)
  2. 2.854s - 9.626s  (duration: 6.772s)
  3. 10.022s - 12.058s  (duration: 2.036s)
  4. 12.486s - 13.978s  (duration: 1.492s)
  5. 14.438s - 19.546s  (duration: 5.108s)

Timing:
  compute spectrogram (GPU)     : 0.576s
  compute VAD features (fused)  : 0.002s
  Total                         : ~0.58s
```

## Command Line Options

```bash
python claude-debug/demo_vad_visualization.py INPUT [OPTIONS]

Arguments:
  INPUT                 Input WAV file path

Options:
  -o, --output PATH    Output image path (default: INPUT_segmentation.png)
  --simple             Simple view (waveform and detection only, no spectrogram)
  --formant-low HZ     Low frequency for formant band (default: 200)
  --formant-high HZ    High frequency for formant band (default: 4800)
  --min-duration SEC   Minimum formant duration (default: 0.1)
  --threshold PCT      Log-variance threshold percentile (default: 0.4)
  -h, --help           Show help message
```

## Examples

### Custom formant band (for telephone speech)
```bash
python claude-debug/demo_vad_visualization.py testsnr.wav \
    --formant-low 300 \
    --formant-high 3400 \
    --output testsnr_telephone.png
```

### More sensitive detection (lower threshold)
```bash
python claude-debug/demo_vad_visualization.py testsnr.wav \
    --threshold 0.3 \
    --output testsnr_sensitive.png
```

### Simple visualization (faster)
```bash
python claude-debug/demo_vad_visualization.py testsnr.wav \
    --simple \
    --output testsnr_simple.png
```

### Remove short segments
```bash
python claude-debug/demo_vad_visualization.py testsnr.wav \
    --min-duration 0.5 \
    --output testsnr_long_only.png
```

## Understanding the Parameters

### Formant Band (--formant-low, --formant-high)
- **Default**: 200-4800 Hz (general speech)
- **Telephone**: 300-3400 Hz (narrowband)
- **Wideband**: 50-8000 Hz (high-quality recording)
- **Male voices**: 100-3000 Hz (lower frequencies)
- **Female voices**: 200-5000 Hz (higher frequencies)

### Threshold (--threshold)
- **Range**: 0.0 to 1.0 (percentile of log-variance distribution)
- **Lower** (e.g., 0.2): More sensitive, detects more speech (may include noise)
- **Higher** (e.g., 0.6): Less sensitive, only strong speech (may miss quiet parts)
- **Default** (0.4): Balanced detection

### Minimum Duration (--min-duration)
- **Range**: 0.0 to several seconds
- **Shorter** (e.g., 0.05): Keep brief sounds
- **Longer** (e.g., 0.5): Remove transients and noise bursts
- **Default** (0.1): Balance between responsiveness and noise rejection

## Output Files

The demo creates PNG images with visualizations. Files are saved to:
- Default: `INPUT_segmentation.png` (same directory as input)
- Custom: Path specified with `--output`

Example:
```
testsnr.wav → testsnr_segmentation.png
audio/speech.wav → audio/speech_segmentation.png
```

## Performance

The GPU-accelerated implementation is fast:

| Audio Duration | Processing Time | Real-time Factor |
|----------------|-----------------|------------------|
| 5 seconds      | ~0.17s          | 29x faster       |
| 20 seconds     | ~0.58s          | 34x faster       |
| 60 seconds     | ~1.5s           | 40x faster       |

*Real-time factor = Duration / Processing Time*

## Troubleshooting

### "matplotlib not available"
```bash
pip install matplotlib
```

### "Input file not found"
Check the file path. The script expects a WAV file.

### "No segments detected"
Try adjusting parameters:
- Lower the threshold: `--threshold 0.2`
- Adjust formant band: `--formant-low 100 --formant-high 6000`
- Check audio quality (low SNR may cause issues)

### Visualization looks wrong
- Make sure input is 16-bit or 32-bit float WAV
- Check sample rate (works best with 8000-48000 Hz)
- Try mono audio (multi-channel uses first channel only)

## Integration

Use the VAD results in your code:

```python
from vad_yapocis import YapocisAudioSegmenter

# Create segmenter
vad = YapocisAudioSegmenter()

# Process audio
seg = vad("testsnr.wav")

# Use results
for i, (start, end) in enumerate(seg.signal_slices):
    print(f"Segment {i+1}: {start:.2f}s - {end:.2f}s")

    # Extract audio segment
    audio_segment = seg.audio_data[
        int(start * seg.samplerate):int(end * seg.samplerate)
    ]

    # Process segment (e.g., speech recognition, analysis)
    process_speech(audio_segment)
```

## Files

- `demo_vad_visualization.py` - Main visualization script
- `demo_testsnr.sh` - Quick demo launcher
- `testsnr_segmentation.png` - Output visualization
- `DEMO_README.md` - This file

## See Also

- `VAD_README.md` - Full VAD documentation
- `vad_yapocis.py` - Implementation details
- `test_vad_yapocis.py` - Test suite
