"""
Display the latest epoch results in a formatted way
Run this script to see the most recent training results
"""

import re
from pathlib import Path

def get_latest_log():
    """Get the latest training log file."""
    results_dir = Path("results")
    if not results_dir.exists():
        return None
    log_files = list(results_dir.glob("training_log_*.txt"))
    if not log_files:
        return None
    return max(log_files, key=lambda p: p.stat().st_mtime)

def extract_latest_epoch(log_content):
    """Extract the latest epoch information."""
    # Find all epochs
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
        'class_metrics': class_metrics
    }

def display_results(data):
    """Display formatted results."""
    if not data:
        print("No epoch data found yet.")
        return
    
    print("\n" + "=" * 80)
    print(f"EPOCH {data['epoch']}/{data['total']} RESULTS")
    print("=" * 80)
    print(f"Progress: {100 * data['epoch'] / data['total']:.1f}%")
    print("")
    print(f"Performance Metrics:")
    print(f"  Train Accuracy: {data['train_acc']:.2f}%")
    print(f"  Test Accuracy:  {data['test_acc']:.2f}%")
    print(f"  Train Loss:     {data['loss']:.4f}")
    print(f"  Learning Rate:  {data['lr']:.6f}")
    print("")
    if data['best_acc']:
        print(f"Best So Far:")
        print(f"  Best Test Accuracy: {data['best_acc']:.2f}% (Epoch {data['best_epoch']})")
        print("")
    if data['class_metrics']:
        print(f"Class-wise Metrics:")
        print(f"  Macro Precision: {data['class_metrics']['precision']:.4f}")
        print(f"  Macro Recall:    {data['class_metrics']['recall']:.4f}")
        print(f"  Macro F1-Score:  {data['class_metrics']['f1']:.4f}")
        print("")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    log_file = get_latest_log()
    if log_file:
        print(f"Reading: {log_file.name}")
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        data = extract_latest_epoch(content)
        display_results(data)
    else:
        print("No training log file found.")










