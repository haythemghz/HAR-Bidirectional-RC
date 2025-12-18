# Fixes Applied for Accuracy Stagnation Issue

## Problem Identified
The training was showing accuracy stagnation at very low values (~1.75% train, 2.00% test) despite loss decreasing, indicating the model wasn't learning effectively.

## Root Causes and Fixes

### 1. **Learning Rate Issues**
**Problem**: Learning rate (0.001) might be too conservative for the readout layer
**Fix**: 
- Increased initial learning rate to 2x (0.002)
- Added `ReduceLROnPlateau` scheduler that reduces LR when accuracy plateaus
- Added weight decay (1e-5) for better regularization

### 2. **Weight Initialization**
**Problem**: Xavier initialization might not be optimal for ReLU-based activations
**Fix**:
- Changed to Kaiming (He) initialization for hidden layers (better for ReLU)
- Reduced output layer initialization (gain=0.1) to prevent large initial outputs
- Improved KAF initialization (smaller scale: 0.1 for alpha, 0.5 for centers)

### 3. **Batch Normalization with Small Batches**
**Problem**: BatchNorm can cause issues when batch size is small (16 samples)
**Fix**:
- Added conditional BatchNorm: only applies when batch size > 1
- Prevents numerical instability with small batches

### 4. **Gradient Clipping**
**Problem**: Gradient clipping was too aggressive (max_norm=1.0)
**Fix**:
- Increased max_norm to 5.0 to allow larger gradient updates
- Still prevents gradient explosion while allowing learning

### 5. **Learning Rate Scheduling**
**Problem**: No adaptive learning rate adjustment
**Fix**:
- Added `ReduceLROnPlateau` scheduler
- Reduces LR by factor of 0.5 when test accuracy doesn't improve for 5 epochs
- Minimum LR set to 1e-6

## Files Modified

1. **`train_with_logging.py`**:
   - Added learning rate scheduler
   - Increased initial learning rate
   - Improved gradient clipping
   - Added LR logging

2. **`readout.py`**:
   - Changed weight initialization (Xavier → Kaiming)
   - Fixed BatchNorm for small batches
   - Improved KAF initialization

3. **`model.py`**:
   - Fixed feature dimension calculation (more explicit)

## Expected Improvements

With these fixes, you should see:
- **Faster convergence**: Higher initial LR helps model learn faster
- **Better accuracy**: Improved initialization and BatchNorm fixes
- **Adaptive learning**: LR scheduler adjusts based on performance
- **More stable training**: Better gradient handling

## Next Steps

**Important**: The current training run is using the old code. To apply these fixes:

1. **Stop current training** (if still running)
2. **Restart training** with the updated code:
   ```bash
   python train_with_logging.py \
       --data_path "..\nturgbd_skeletons_s001_to_s017\nturgb+d_skeletons" \
       --max_samples 500 \
       --epochs 50 \
       --batch_size 16 \
       --reservoir_size 500 \
       --learning_rate 0.001
   ```

The fixes will automatically apply with the updated code.

## Monitoring

After restarting, you should see:
- Learning rate displayed in logs
- LR reductions when accuracy plateaus
- Faster accuracy improvement
- Better final accuracy

Use `python show_summary.py` to monitor progress.











