"""
Display all epoch results in a formatted table
Shows progress every epoch
"""

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

def extract_all_epochs(log_content):
    """Extract all epoch information."""
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
    
    # Find best performance
    best_match = re.search(r'Best Test Accuracy: ([\d.]+)% \(Epoch (\d+)\)', log_content)
    best_acc = float(best_match.group(1)) if best_match else None
    best_epoch = int(best_match.group(2)) if best_match else None
    
    return epochs, best_acc, best_epoch

def display_all_epochs(epochs, best_acc, best_epoch):
    """Display all epochs in a formatted table."""
    if not epochs:
        print("No epoch data found yet. Training may still be in setup phase.")
        return
    
    latest = epochs[-1]
    total = latest[1]
    
    print("\n" + "=" * 100)
    print(f"TRAINING PROGRESS - Epoch {latest[0]}/{total}")
    print("=" * 100)
    print(f"{'Epoch':<8} {'Train Acc':<12} {'Test Acc':<12} {'Loss':<12} {'LR':<15} {'Status':<15}")
    print("-" * 100)
    
    for epoch, total_ep, loss, train_acc, test_acc, lr in epochs:
        status = ""
        if best_epoch and epoch == best_epoch:
            status = "★ BEST"
        elif epoch == latest[0]:
            status = "→ Latest"
        
        print(f"{epoch:<8} {train_acc:<12.2f} {test_acc:<12.2f} {loss:<12.4f} {lr:<15.6f} {status:<15}")
    
    print("-" * 100)
    print("")
    print(f"Latest Results:")
    print(f"  Train Accuracy: {latest[3]:.2f}%")
    print(f"  Test Accuracy:  {latest[4]:.2f}%")
    print(f"  Train Loss:     {latest[2]:.4f}")
    print(f"  Learning Rate:  {latest[5]:.6f}")
    if best_acc:
        print(f"")
        print(f"Best Performance:")
        print(f"  Best Test Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
    print("=" * 100 + "\n")

if __name__ == '__main__':
    log_file = get_latest_log()
    if log_file:
        print(f"Reading: {log_file.name}")
        print(f"Last updated: {datetime.fromtimestamp(log_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        epochs, best_acc, best_epoch = extract_all_epochs(content)
        display_all_epochs(epochs, best_acc, best_epoch)
    else:
        print("No training log file found.")










