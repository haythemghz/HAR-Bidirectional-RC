"""
Monitor training progress and display summary every 5 epochs
"""

import os
import re
import time
from pathlib import Path

def find_latest_log():
    """Find the latest training log file."""
    results_dir = Path("results")
    if not results_dir.exists():
        return None
    
    log_files = list(results_dir.glob("training_log_*.txt"))
    if not log_files:
        return None
    
    return max(log_files, key=lambda p: p.stat().st_mtime)

def extract_epoch_summary(log_content):
    """Extract summary information from log content."""
    # Pattern to match epoch information
    epoch_pattern = r'Epoch (\d+)/(\d+).*?Test Accuracy: ([\d.]+)%'
    
    epochs = []
    for match in re.finditer(epoch_pattern, log_content, re.DOTALL):
        epoch_num = int(match.group(1))
        total_epochs = int(match.group(2))
        test_acc = float(match.group(3))
        epochs.append((epoch_num, total_epochs, test_acc))
    
    return epochs

def extract_best_performance(log_content):
    """Extract best performance metrics."""
    best_pattern = r'Best Test Accuracy: ([\d.]+)% \(Epoch (\d+)\)'
    match = re.search(best_pattern, log_content)
    if match:
        return float(match.group(1)), int(match.group(2))
    return None, None

def extract_training_metrics(log_content):
    """Extract training metrics from log."""
    metrics = {}
    
    # Extract train accuracy
    train_acc_pattern = r'Train Accuracy: ([\d.]+)%'
    train_accs = [float(m.group(1)) for m in re.finditer(train_acc_pattern, log_content)]
    if train_accs:
        metrics['latest_train_acc'] = train_accs[-1]
    
    # Extract test accuracy
    test_acc_pattern = r'Test Accuracy: ([\d.]+)%'
    test_accs = [float(m.group(1)) for m in re.finditer(test_acc_pattern, log_content)]
    if test_accs:
        metrics['latest_test_acc'] = test_accs[-1]
    
    # Extract loss
    loss_pattern = r'Train Loss: ([\d.]+)'
    losses = [float(m.group(1)) for m in re.finditer(loss_pattern, log_content)]
    if losses:
        metrics['latest_loss'] = losses[-1]
    
    return metrics

def display_summary(epochs, best_acc, best_epoch, metrics, total_epochs):
    """Display formatted summary."""
    if not epochs:
        print("No epoch data found yet. Training may still be in early phases.")
        return
    
    latest_epoch, _, latest_test_acc = epochs[-1]
    
    print("\n" + "=" * 80)
    print("TRAINING PROGRESS SUMMARY")
    print("=" * 80)
    print(f"Current Epoch: {latest_epoch}/{total_epochs}")
    print(f"Progress: {100 * latest_epoch / total_epochs:.1f}%")
    print("")
    
    if metrics:
        print("Latest Metrics:")
        if 'latest_train_acc' in metrics:
            print(f"  Train Accuracy: {metrics['latest_train_acc']:.2f}%")
        if 'latest_test_acc' in metrics:
            print(f"  Test Accuracy: {metrics['latest_test_acc']:.2f}%")
        if 'latest_loss' in metrics:
            print(f"  Train Loss: {metrics['latest_loss']:.4f}")
        print("")
    
    if best_acc is not None:
        print(f"Best Performance:")
        print(f"  Best Test Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
        print("")
    
    # Show every 5th epoch
    print("Epoch Progress (every 5 epochs):")
    print(f"{'Epoch':<10} {'Test Acc (%)':<15}")
    print("-" * 25)
    for epoch_num, total, test_acc in epochs:
        if epoch_num % 5 == 0 or epoch_num == 1:
            print(f"{epoch_num:<10} {test_acc:<15.2f}")
    
    # Show latest if not a multiple of 5
    if latest_epoch % 5 != 0:
        print(f"{latest_epoch:<10} {latest_test_acc:<15.2f} (latest)")
    
    print("=" * 80 + "\n")

def monitor_training(interval=10):
    """Monitor training log file and display updates."""
    print("Monitoring training progress...")
    print("Press Ctrl+C to stop monitoring\n")
    
    last_size = 0
    total_epochs = 50  # Default, will be updated from log
    
    try:
        while True:
            log_file = find_latest_log()
            
            if log_file is None:
                print("Waiting for log file to be created...")
                time.sleep(5)
                continue
            
            # Read log file
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    current_size = f.tell()
            except Exception as e:
                print(f"Error reading log file: {e}")
                time.sleep(interval)
                continue
            
            # Check if file has been updated
            if current_size == last_size:
                time.sleep(interval)
                continue
            
            last_size = current_size
            
            # Extract total epochs from log
            total_match = re.search(r'Epoch \d+/(\d+)', content)
            if total_match:
                total_epochs = int(total_match.group(1))
            
            # Extract information
            epochs = extract_epoch_summary(content)
            best_acc, best_epoch = extract_best_performance(content)
            metrics = extract_training_metrics(content)
            
            # Display summary
            if epochs:
                display_summary(epochs, best_acc, best_epoch, metrics, total_epochs)
            
            # Check if training is complete
            if "Training completed" in content or "RESULTS SUMMARY" in content:
                print("\n" + "=" * 80)
                print("TRAINING COMPLETED!")
                print("=" * 80)
                display_summary(epochs, best_acc, best_epoch, metrics, total_epochs)
                break
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        log_file = find_latest_log()
        if log_file:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            epochs = extract_epoch_summary(content)
            best_acc, best_epoch = extract_best_performance(content)
            metrics = extract_training_metrics(content)
            total_match = re.search(r'Epoch \d+/(\d+)', content)
            total_epochs = int(total_match.group(1)) if total_match else 50
            display_summary(epochs, best_acc, best_epoch, metrics, total_epochs)

if __name__ == '__main__':
    import sys
    
    # Check if we should just show current status
    if len(sys.argv) > 1 and sys.argv[1] == '--once':
        log_file = find_latest_log()
        if log_file:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            epochs = extract_epoch_summary(content)
            best_acc, best_epoch = extract_best_performance(content)
            metrics = extract_training_metrics(content)
            total_match = re.search(r'Epoch \d+/(\d+)', content)
            total_epochs = int(total_match.group(1)) if total_match else 50
            display_summary(epochs, best_acc, best_epoch, metrics, total_epochs)
        else:
            print("No log file found.")
    else:
        # Continuous monitoring
        monitor_training(interval=10)











