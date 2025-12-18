"""
Live monitor to display training results in real-time
Shows formatted results as they appear in the log file
"""

import os
import re
import time
from pathlib import Path
from datetime import datetime

def get_latest_log():
    """Get the latest training log file."""
    results_dir = Path("results")
    if not results_dir.exists():
        return None
    log_files = list(results_dir.glob("training_log_*.txt"))
    if not log_files:
        return None
    return max(log_files, key=lambda p: p.stat().st_mtime)

def extract_epoch_info(log_content):
    """Extract epoch information from log."""
    # Pattern to match epoch results
    epoch_pattern = r'Epoch (\d+)/(\d+).*?Train Loss: ([\d.]+).*?Train Accuracy: ([\d.]+)%.*?Test Accuracy: ([\d.]+)%.*?Learning Rate: ([\d.]+)'
    
    epochs = []
    for match in re.finditer(epoch_pattern, log_content, re.DOTALL):
        epoch = int(match.group(1))
        total = int(match.group(2))
        loss = float(match.group(3))
        train_acc = float(match.group(4))
        test_acc = float(match.group(5))
        lr = float(match.group(6))
        epochs.append((epoch, total, loss, train_acc, test_acc, lr))
    
    # Extract best performance
    best_match = re.search(r'Best Test Accuracy: ([\d.]+)% \(Epoch (\d+)\)', log_content)
    best_acc = float(best_match.group(1)) if best_match else None
    best_epoch = int(best_match.group(2)) if best_match else None
    
    # Extract class metrics for latest epoch
    class_metrics = None
    if epochs:
        # Find metrics for latest epoch
        latest_epoch = epochs[-1][0]
        metrics_pattern = rf'Epoch {latest_epoch}/\d+.*?Macro Precision: ([\d.]+).*?Macro Recall: ([\d.]+).*?Macro F1-Score: ([\d.]+)'
        metrics_match = re.search(metrics_pattern, log_content, re.DOTALL)
        if metrics_match:
            class_metrics = {
                'precision': float(metrics_match.group(1)),
                'recall': float(metrics_match.group(2)),
                'f1': float(metrics_match.group(3))
            }
    
    return epochs, best_acc, best_epoch, class_metrics

def display_epoch_results(epoch, total, loss, train_acc, test_acc, lr, best_acc, best_epoch, class_metrics, elapsed_time, remaining_time):
    """Display formatted epoch results."""
    print("\n" + "=" * 80)
    print(f"EPOCH {epoch}/{total} RESULTS")
    print("=" * 80)
    print(f"Progress: {100 * epoch / total:.1f}% | Elapsed: {elapsed_time/60:.2f} min | Remaining: ~{remaining_time/60:.2f} min")
    print("")
    print(f"Performance Metrics:")
    print(f"  Train Accuracy: {train_acc:.2f}%")
    print(f"  Test Accuracy:  {test_acc:.2f}%")
    print(f"  Train Loss:     {loss:.4f}")
    print(f"  Learning Rate:  {lr:.6f}")
    print("")
    if best_acc is not None:
        print(f"Best So Far:")
        print(f"  Best Test Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
        print("")
    if class_metrics:
        print(f"Class-wise Metrics:")
        print(f"  Macro Precision: {class_metrics['precision']:.4f}")
        print(f"  Macro Recall:    {class_metrics['recall']:.4f}")
        print(f"  Macro F1-Score:  {class_metrics['f1']:.4f}")
        print("")
    print("=" * 80)

def monitor_live(interval=5):
    """Monitor training log and display results in real-time."""
    print("=" * 80)
    print("LIVE TRAINING MONITOR")
    print("=" * 80)
    print("Monitoring training progress...")
    print("Press Ctrl+C to stop monitoring\n")
    
    log_file = get_latest_log()
    if not log_file:
        print("No training log file found. Waiting for training to start...")
        while not log_file:
            time.sleep(2)
            log_file = get_latest_log()
    
    print(f"Monitoring: {log_file.name}\n")
    
    last_epoch = 0
    start_time = time.time()
    
    try:
        while True:
            # Read log file
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                print(f"Error reading log: {e}")
                time.sleep(interval)
                continue
            
            # Extract epoch information
            epochs, best_acc, best_epoch, class_metrics = extract_epoch_info(content)
            
            # Display new epochs
            if epochs and epochs[-1][0] > last_epoch:
                for epoch_info in epochs:
                    epoch, total, loss, train_acc, test_acc, lr = epoch_info
                    if epoch > last_epoch:
                        # Calculate times
                        elapsed = time.time() - start_time
                        avg_epoch_time = elapsed / epoch if epoch > 0 else 0
                        remaining = avg_epoch_time * (total - epoch)
                        
                        # Display results
                        display_epoch_results(
                            epoch, total, loss, train_acc, test_acc, lr,
                            best_acc, best_epoch, class_metrics,
                            elapsed, remaining
                        )
                        
                        last_epoch = epoch
            
            # Check if training is complete
            if "Training completed" in content or "RESULTS SUMMARY" in content:
                print("\n" + "=" * 80)
                print("TRAINING COMPLETED!")
                print("=" * 80)
                if epochs:
                    epoch, total, loss, train_acc, test_acc, lr = epochs[-1]
                    elapsed = time.time() - start_time
                    display_epoch_results(
                        epoch, total, loss, train_acc, test_acc, lr,
                        best_acc, best_epoch, class_metrics,
                        elapsed, 0
                    )
                break
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        if epochs:
            epoch, total, loss, train_acc, test_acc, lr = epochs[-1]
            elapsed = time.time() - start_time
            display_epoch_results(
                epoch, total, loss, train_acc, test_acc, lr,
                best_acc, best_epoch, class_metrics,
                elapsed, 0
            )

if __name__ == '__main__':
    monitor_live(interval=3)  # Check every 3 seconds










