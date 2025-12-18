"""
Quick script to show training summary every 5 epochs
Run this script to see current training progress
"""

import os
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

def extract_summaries(log_file):
    """Extract epoch summaries from log file."""
    if not log_file or not log_file.exists():
        return None
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Find all epochs
    epoch_pattern = r'Epoch (\d+)/(\d+).*?Train Accuracy: ([\d.]+)%.*?Test Accuracy: ([\d.]+)%'
    epochs = []
    
    for match in re.finditer(epoch_pattern, content, re.DOTALL):
        epoch = int(match.group(1))
        total = int(match.group(2))
        train_acc = float(match.group(3))
        test_acc = float(match.group(4))
        epochs.append((epoch, total, train_acc, test_acc))
    
    # Find best performance
    best_match = re.search(r'Best Test Accuracy: ([\d.]+)% \(Epoch (\d+)\)', content)
    best_acc = float(best_match.group(1)) if best_match else None
    best_epoch = int(best_match.group(2)) if best_match else None
    
    # Find latest loss
    loss_matches = list(re.finditer(r'Train Loss: ([\d.]+)', content))
    latest_loss = float(loss_matches[-1].group(1)) if loss_matches else None
    
    return {
        'epochs': epochs,
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'latest_loss': latest_loss,
        'total_epochs': epochs[0][1] if epochs else None
    }

def print_summary(data):
    """Print formatted summary."""
    if not data or not data['epochs']:
        print("\n" + "=" * 80)
        print("TRAINING STATUS")
        print("=" * 80)
        print("Training is still in setup phase (data loading, reservoir processing, etc.)")
        print("Please wait for training to begin...")
        print("=" * 80 + "\n")
        return
    
    epochs = data['epochs']
    latest = epochs[-1]
    total = data['total_epochs']
    
    print("\n" + "=" * 80)
    print("TRAINING PROGRESS SUMMARY")
    print("=" * 80)
    print(f"Current Status: Epoch {latest[0]}/{total}")
    print(f"Progress: {100 * latest[0] / total:.1f}%")
    print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    print("Latest Metrics:")
    print(f"  Train Accuracy: {latest[2]:.2f}%")
    print(f"  Test Accuracy: {latest[3]:.2f}%")
    if data['latest_loss']:
        print(f"  Train Loss: {data['latest_loss']:.4f}")
    print("")
    
    if data['best_acc']:
        print(f"Best Performance So Far:")
        print(f"  Best Test Accuracy: {data['best_acc']:.2f}% (Epoch {data['best_epoch']})")
        print("")
    
    # Show summary every 5 epochs
    print("Summary Every 5 Epochs:")
    print(f"{'Epoch':<8} {'Train Acc':<12} {'Test Acc':<12} {'Improvement':<12}")
    print("-" * 44)
    
    prev_test_acc = None
    for epoch, total, train_acc, test_acc in epochs:
        if epoch % 5 == 0 or epoch == 1:
            improvement = ""
            if prev_test_acc is not None:
                diff = test_acc - prev_test_acc
                improvement = f"{diff:+.2f}%" if diff != 0 else "0.00%"
            print(f"{epoch:<8} {train_acc:<12.2f} {test_acc:<12.2f} {improvement:<12}")
            prev_test_acc = test_acc
    
    # Show latest if not a multiple of 5
    if latest[0] % 5 != 0:
        diff = latest[3] - prev_test_acc if prev_test_acc else 0
        improvement = f"{diff:+.2f}%" if diff != 0 else "0.00%"
        print(f"{latest[0]:<8} {latest[2]:<12.2f} {latest[3]:<12.2f} {improvement:<12} (latest)")
    
    print("=" * 80 + "\n")

if __name__ == '__main__':
    log_file = get_latest_log()
    if log_file:
        print(f"Reading log file: {log_file.name}")
        data = extract_summaries(log_file)
        print_summary(data)
    else:
        print("No training log file found. Training may not have started yet.")











