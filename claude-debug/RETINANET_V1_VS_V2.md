# RetinaNet V1 vs V2 Differences

## Summary

RetinaNet V1 and V2 have different architectures, particularly in the detection heads.

## Architecture Differences

### V1 (retinanet_resnet50_fpn)
- Backbone: ResNet50 with BN after each conv (needs folding)
- FPN: Conv layers WITH bias (no BN)
- **Classification Head**: Conv layers WITH bias (no BN)
- **Regression Head**: Conv layers WITH bias (no BN)

### V2 (retinanet_resnet50_fpn_v2)
- Backbone: ResNet50 with BN after each conv (needs folding)
- FPN: Conv layers WITH bias (no BN)
- **Classification Head**: Conv layers WITHOUT bias, FOLLOWED BY BN (needs folding!)
- **Regression Head**: Conv layers WITHOUT bias, FOLLOWED BY BN (needs folding!)

## Key Naming Differences

### V1 Head Structure
```
head.classification_head.conv.{i}.0.weight  # Conv layer (HAS bias)
head.classification_head.conv.{i}.0.bias    # Conv bias
```

### V2 Head Structure
```
head.classification_head.conv.{i}.0.weight      # Conv layer (NO bias)
head.classification_head.conv.{i}.1.weight      # BN weight (gamma)
head.classification_head.conv.{i}.1.bias        # BN bias (beta)
head.classification_head.conv.{i}.1.running_mean
head.classification_head.conv.{i}.1.running_var
```

## Impact on Conversion

V1: Heads can be copied directly (conv weights + bias)
V2: **Heads need BN folding** (conv + BN → folded conv weights + bias)

This is why v2 conversion created only 134 tensors vs v1's 142 tensors - v2 skipped the BN layers in heads.

## Solution

Update conversion script to detect and fold BN in v2 heads:
1. Check if `conv.{i}.1.weight` exists (indicates BN)
2. If yes: fold BN into conv
3. If no: copy conv weights + bias directly
