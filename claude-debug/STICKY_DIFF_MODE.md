# Sticky Difference View Mode

## What Changed

The difference view mode is now **sticky** - it persists when navigating between images.

## Previous Behavior

- Press F8 to view differences for current image
- Press 'a' or 'd' to navigate → **difference view disabled**, back to normal view
- Have to press F8 again for each image

## New Behavior

- Press F8 to enable difference view mode
- Navigate with 'a', 'd', 'w', 'e' → **difference view stays active**
- Difference visualization automatically applied to each new image
- Press F8 again to toggle back to normal view

## Benefits

1. **Efficient batch comparison** - Review multiple test results without re-toggling
2. **Workflow continuity** - Stay in analysis mode while navigating
3. **Faster debugging** - Quickly scan through failed tests in diff mode

## Usage Example

```bash
# Start imagewatcher
yapocis-imagewatcher

# Enable difference mode
Press F8

# Navigate through images - diff view persists!
Press 'd' → Next image in diff view
Press 'd' → Next image in diff view
Press 'd' → Next image in diff view

# Toggle back to normal view when done
Press F8
```

## Technical Details

### Functions Modified

**Navigation functions** (`next_image`, `previous_image`, `first_image`, `last_image`)
- Now check `showing_difference_view` flag
- If active, re-apply difference visualization to new image
- If inactive, show normal view

**Tailing function** (`step_image`)
- Preserves view mode when auto-advancing
- Applies difference view to newly detected files

**View functions** (`create_difference_visualization`, `view_validation_reference`)
- Made `font` parameter optional (defaults to `None`)
- Status messages only shown when font provided
- Allows silent re-application during navigation

### View Mode States

Three mutually exclusive states:
1. **Normal view** - Show test image only
2. **Validation view** (F9) - Show validation reference only
3. **Difference view** (F8) - Show 3-panel comparison

Toggling any view automatically disables the others.

### State Persistence

The view mode flags are preserved across:
- Forward navigation ('d')
- Backward navigation ('a')
- Jump to first ('w')
- Jump to last ('e')
- Tailing mode advancement ('s')

## Examples

### Reviewing All Graphics Tests

```bash
yapocis-imagewatcher --queue queue/rendered graphics

# Enable difference mode
F8

# Scan through all graphics tests
# (Difference view persists automatically)
d d d d d ...
```

### Comparing Failed Validations

```bash
yapocis-imagewatcher --validation-failed

# Enable difference mode
F8

# Review each failure
# (Each shows current | validation | difference)
d d d ...
```

## Code Changes

**yapocis/scripts/imagewatcher.py:**
- Lines 514-572: Updated navigation functions to preserve view mode
- Lines 575-615: Updated `step_image` to preserve view mode
- Lines 762-858: Made `view_validation_reference` font optional
- Lines 936-1175: Made `create_difference_visualization` font optional

Total changes: ~100 lines modified
