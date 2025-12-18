"""
Real-time training monitor - displays metrics every epoch as they appear
"""

import time
import re
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

def extract_latest_epoch_info(log_content):
    """Extract the latest epoch information."""
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
    
    if not epochs:
        return None
    
    latest = epochs[-1]
    
    # Find best performance
    best_match = re.search(r'Best Test Accuracy: ([\d.]+)% \(Epoch (\d+)\)', log_content)
    best_acc = float(best_match.group(1)) if best_match else None
    best_epoch = int(best_match.group(2)) if best_match else None
    
    # Find class metrics for latest epoch
    latest_epoch_num = latest[0]
    metrics_pattern = rf'Epoch {latest_epoch_num}/\d+.*?Macro Precision: ([\d.]+).*?Macro Recall: ([\d.]+).*?Macro F1-Score: ([\d.]+)'
    metrics_match = re.search(metrics_pattern, log_content, re.DOTALL)
    
    class_metrics = None
    if metrics_match:
        class_metrics = {
            'precision': float(metrics_match.group(1)),
            'recall': float(metrics_match.group(2)),
            'f1': float(metrics_match.group(3))
        }
    
    return {
        'epoch': latest[0],
        'total': latest[1],
        'loss': latest[2],
        'train_acc': latest[3],
        'test_acc': latest[4],
        'lr': latest[5],
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'class_metrics': class_metrics,
        'all_epochs': epochs
    }

def display_epoch(epoch_info):
    """Display formatted epoch information."""
    if not epoch_info:
        return
    
    epoch, total, loss, train_acc, test_acc, lr = (
        epoch_info['epoch'], epoch_info['total'], epoch_info['loss'],
        epoch_info['train_acc'], epoch_info['test_acc'], epoch_info['lr']
    )
    
    print("\n" + "=" * 80)
    print(f"EPOCH {epoch}/{total} RESULTS")
    print("=" * 80)
    print(f"Progress: {100 * epoch / total:.1f}%")
    print("")
    print(f"Performance Metrics:")
    print(f"  Train Accuracy: {train_acc:.2f}%")
    print(f"  Test Accuracy:  {test_acc:.2f}%")
    print(f"  Train Loss:     {loss:.4f}")
    print(f"  Learning Rate:  {lr:.6f}")
    print("")
    if epoch_info['best_acc']:
        print(f"Best So Far:")
        print(f"  Best Test Accuracy: {epoch_info['best_acc']:.2f}% (Epoch {epoch_info['best_epoch']})")
        print("")
    if epoch_info['class_metrics']:
        print(f"Class-wise Metrics:")
        print(f"  Macro Precision: {epoch_info['class_metrics']['precision']:.4f}")
        print(f"  Macro Recall:    {epoch_info['class_metrics']['recall']:.4f}")
        print(f"  Macro F1-Score:  {epoch_info['class_metrics']['f1']:.4f}")
        print("")
    print("=" * 80)

def monitor_continuously():
    """Continuously monitor and display new epochs."""
    print("=" * 80)
    print("REAL-TIME TRAINING MONITOR")
    print("=" * 80)
    print("Monitoring training progress...")
    print("New epochs will be displayed automatically")
    print("Press Ctrl+C to stop\n")
    
    log_file = get_latest_log()
    if not log_file:
        print("Waiting for training log to be created...")
        while not log_file:
            time.sleep(2)
            log_file = get_latest_log()
    
    print(f"Monitoring: {log_file.name}\n")
    
    last_epoch = 0
    last_size = 0
    
    try:
        while True:
            # Read log file
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    current_size = len(content)
            except Exception as e:
                print(f"Error reading log: {e}")
                time.sleep(3)
                continue
            
            # Check if file has been updated
            if current_size == last_size:
                time.sleep(3)
                continue
            
            last_size = current_size
            
            # Extract epoch information
            epoch_info = extract_latest_epoch_info(content)
            
            # Display new epochs
            if epoch_info and epoch_info['epoch'] > last_epoch:
                # Display all new epochs
                for ep in epoch_info['all_epochs']:
                    if ep[0] > last_epoch:
                        # Create epoch info dict for this epoch
                        ep_info = {
                            'epoch': ep[0],
                            'total': ep[1],
                            'loss': ep[2],
                            'train_acc': ep[3],
                            'test_acc': ep[4],
                            'lr': ep[5],
                            'best_acc': epoch_info['best_acc'],
                            'best_epoch': epoch_info['best_epoch'],
                            'class_metrics': epoch_info['class_metrics'] if ep[0] == epoch_info['epoch'] else None
                        }
                        display_epoch(ep_info)
                        last_epoch = ep[0]
            
            # Check if training is complete
            if "Training completed" in content or "RESULTS SUMMARY" in content:
                print("\n" + "=" * 80)
                print("TRAINING COMPLETED!")
                print("=" * 80)
                if epoch_info:
                    display_epoch(epoch_info)
                break
            
            time.sleep(3)  # Check every 3 seconds
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        if epoch_info:
            display_epoch(epoch_info)

if __name__ == '__main__':
    monitor_continuously()










