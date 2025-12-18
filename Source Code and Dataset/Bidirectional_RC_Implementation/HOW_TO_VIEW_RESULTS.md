# How to View Training Results

## Problem
Training runs in the background, so console output isn't visible in real-time.

## Solution: Use These Commands

### 1. View Latest Epoch Results (Recommended)
```bash
python show_latest_epoch.py
```
Shows the most recent epoch in a formatted display.

### 2. View Summary (Every 5 Epochs)
```bash
python show_summary.py
```
Shows a summary table with progress every 5 epochs.

### 3. View Full Log File
```bash
# Windows PowerShell
Get-Content results\training_log_*.txt -Tail 50

# Or open the file directly
notepad results\training_log_*.txt
```

### 4. Live Monitoring (Optional)
```bash
python live_monitor.py
```
Continuously monitors and displays new epochs as they appear.

## Quick Status Check

Run this to see current status:
```bash
python show_latest_epoch.py
```

## What You'll See

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

**Log File**: `training_log_20251217_212932.txt`

**Status**: Training is running in background
- Check progress anytime with `python show_latest_epoch.py`
- Results are saved to the log file
- Final results will be in JSON format in `results/` directory










