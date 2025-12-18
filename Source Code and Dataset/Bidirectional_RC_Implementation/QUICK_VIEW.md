# Quick View Guide - See Metrics Every Epoch

## Training Status

Training is running in the background with all fixes applied.

## How to View Results

### Option 1: View All Epochs (Recommended)
```bash
python display_epochs.py
```
Shows a formatted table with all epochs and their metrics.

### Option 2: View Latest Epoch Only
```bash
python show_latest_epoch.py
```
Shows the most recent epoch in detail.

### Option 3: Continuous Monitoring
```bash
python watch_training.py
```
Continuously monitors and displays new epochs as they appear (runs until stopped).

## What You'll See

### Every Epoch Display:
```
================================================================================
EPOCH X/50 RESULTS
================================================================================
Progress: X%

Performance Metrics:
  Train Accuracy: XX.XX%
  Test Accuracy:  XX.XX%
  Train Loss:     X.XXXX
  Learning Rate:  X.XXXXXX

Best So Far:
  Best Test Accuracy: XX.XX% (Epoch X)

Class-wise Metrics:
  Macro Precision: X.XXXX
  Macro Recall:    X.XXXX
  Macro F1-Score:  X.XXXX
================================================================================
```

## Current Training

**Log File**: `training_log_20251217_214640.txt`

**Status**: Training is running (may still be in setup phase)

**Fixes Applied**:
- ✅ Representation learning will be TRAINED (gradients will flow)
- ✅ Improved learning rate with scheduling
- ✅ Better weight initialization
- ✅ Fixed BatchNorm for small batches

## Expected Results

With the fixes, you should see:
- **Accuracy improving** from ~2% to 15-40%+ within first 10-20 epochs
- **Loss decreasing** more rapidly
- **Learning rate adapting** based on performance

## Quick Commands

```bash
# Check current status
python show_latest_epoch.py

# View all epochs
python display_epochs.py

# Monitor continuously
python watch_training.py
```










