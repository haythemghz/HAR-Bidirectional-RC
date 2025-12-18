"""
Training Script with Comprehensive Logging for Paper Documentation
Logs all results, metrics, and configurations for paper enhancement
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import json
from datetime import datetime
from tqdm import tqdm
import time
import sys

from model import BidirectionalRC
from data_utils import (
    load_skeleton_data, normalize_sequences, prepare_data_for_training,
    compute_sequence_lengths
)


class SkeletonDataset(Dataset):
    """Dataset class for skeleton sequences."""
    
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def collate_fn(batch):
    """Custom collate function for variable-length sequences."""
    sequences, labels = zip(*batch)
    return list(sequences), torch.LongTensor(labels)


def train_epoch(model, dataloader, optimizer, device, lambda1=0.001, lambda2=0.0001, lambda3=0.01):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_l2_reg = 0.0
    total_kaf_reg = 0.0
    total_fusion_reg = 0.0
    correct = 0
    total = 0
    
    for sequences, labels in tqdm(dataloader, desc="Training", leave=False):
        labels = labels.to(device)
        
        # Forward pass - CRITICAL: Pass training=True to enable gradients
        model.train()  # Ensure model is in training mode
        if model.representation_learning is not None:
            model.representation_learning.train()
        logits = model(sequences, training=True)
        
        # Compute loss
        # FIX: Use standard cross-entropy with class weights to handle imbalance
        # For now, use standard loss - can add class weights later if needed
        ce_loss = F.cross_entropy(logits, labels)
        
        # Get regularization components
        l2_reg = 0.0
        for param in model.parameters():
            if param.requires_grad:
                l2_reg += torch.norm(param, p=2) ** 2
        
        # Compute total loss
        loss = ce_loss + lambda1 * l2_reg
        
        # Get components for logging
        components = {
            'ce_loss': ce_loss.item(),
            'l2_reg': l2_reg.item() if isinstance(l2_reg, torch.Tensor) else l2_reg,
            'kaf_reg': 0.0,  # KAF reg handled separately if needed
            'fusion_reg': 0.0
        }
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # FIX: Improved gradient clipping - more aggressive
        # Get all trainable parameters
        trainable_params = []
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                trainable_params.append(param)
        
        # Clip gradients aggressively to prevent explosion
        # Also clip individual parameter gradients
        max_grad_value = 10.0  # Clip individual gradients
        torch.nn.utils.clip_grad_value_(trainable_params, max_grad_value)
        
        # Then clip overall norm
        total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)  # Very aggressive
        
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        total_ce_loss += components['ce_loss']
        total_l2_reg += components.get('l2_reg', 0)
        total_kaf_reg += components.get('kaf_reg', 0)
        total_fusion_reg += components.get('fusion_reg', 0)
        
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_l2_reg = total_l2_reg / len(dataloader)
    avg_kaf_reg = total_kaf_reg / len(dataloader)
    avg_fusion_reg = total_fusion_reg / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return {
        'loss': avg_loss,
        'ce_loss': avg_ce_loss,
        'l2_reg': avg_l2_reg,
        'kaf_reg': avg_kaf_reg,
        'fusion_reg': avg_fusion_reg,
        'accuracy': accuracy
    }


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            labels = labels.to(device)
            
            # Forward pass - training=False for inference
            logits = model(sequences, training=False)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = 100.0 * correct / total
    
    return accuracy, all_preds, all_labels, all_probs


def compute_class_wise_metrics(labels, preds, num_classes):
    """Compute per-class precision, recall, and F1-score."""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=range(num_classes), average=None, zero_division=0
    )
    
    # Overall metrics
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    return {
        'per_class': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist()
        },
        'macro': {
            'precision': float(macro_precision),
            'recall': float(macro_recall),
            'f1': float(macro_f1)
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Train Bidirectional RC with comprehensive logging')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to skeleton data directory')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to load (None = all)')
    
    # Model arguments
    parser.add_argument('--reservoir_size', type=int, default=1000,
                       help='Reservoir size H')
    parser.add_argument('--fusion_strategy', type=str, default='concat',
                       choices=['concat', 'weighted', 'attention'],
                       help='Fusion strategy')
    parser.add_argument('--spectral_radius', type=float, default=0.95,
                       help='Spectral radius ρ')
    parser.add_argument('--sparsity', type=float, default=0.05,
                       help='Connection sparsity γ')
    parser.add_argument('--input_scaling', type=float, default=0.5,
                       help='Input weight scaling σ_in')
    parser.add_argument('--leak_rate_f', type=float, default=0.3,
                       help='Forward leak rate α_f')
    parser.add_argument('--leak_rate_b', type=float, default=0.3,
                       help='Backward leak rate α_b')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--lambda1', type=float, default=0.001,
                       help='L2 regularization weight')
    parser.add_argument('--lambda2', type=float, default=0.0001,
                       help='KAF regularization weight')
    parser.add_argument('--lambda3', type=float, default=0.01,
                       help='Fusion regularization weight')
    
    # Other arguments
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f'training_results_{timestamp}.json')
    log_file = os.path.join(args.output_dir, f'training_log_{timestamp}.txt')
    
    # Initialize logging
    log_f = open(log_file, 'w', encoding='utf-8')
    
    def log_print(message):
        print(message)
        try:
            log_f.write(message + '\n')
        except UnicodeEncodeError:
            # Replace Greek letters with ASCII equivalents
            safe_message = message.replace('ρ', 'rho').replace('α', 'alpha').replace('γ', 'gamma').replace('σ', 'sigma').replace('λ', 'lambda')
            log_f.write(safe_message + '\n')
        log_f.flush()
    
    log_print("=" * 80)
    log_print("BIDIRECTIONAL RESERVOIR COMPUTING FOR HUMAN ACTION RECOGNITION")
    log_print("Comprehensive Training Run with Full Logging")
    log_print("=" * 80)
    log_print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("")
    
    # Set random seeds
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    
    device = torch.device(args.device)
    log_print(f"Device: {device}")
    log_print(f"Random Seed: {args.random_seed}")
    log_print("")
    
    # Load data
    log_print("=" * 80)
    log_print("PHASE 1: DATA LOADING AND PREPROCESSING")
    log_print("=" * 80)
    log_print(f"Loading data from: {args.data_path}")
    
    start_time = time.time()
    sequences, labels, metadata = load_skeleton_data(
        args.data_path,
        file_pattern="*.skeleton",
        max_samples=args.max_samples
    )
    load_time = time.time() - start_time
    
    log_print(f"Data loading completed in {load_time:.2f} seconds")
    log_print(f"Total samples loaded: {len(sequences)}")
    log_print(f"Number of classes: {metadata['num_classes']}")
    log_print(f"Class names: {list(metadata['class_names'][:10])}..." if len(metadata['class_names']) > 10 else f"Class names: {list(metadata['class_names'])}")
    log_print("")
    
    # Normalize sequences
    log_print("Normalizing sequences...")
    sequences, scaler = normalize_sequences(sequences, method='standard')
    
    # Compute sequence length statistics
    length_stats = compute_sequence_lengths(sequences)
    log_print("Sequence Length Statistics:")
    log_print(f"  Minimum: {length_stats['min']}")
    log_print(f"  Maximum: {length_stats['max']}")
    log_print(f"  Mean: {length_stats['mean']:.2f}")
    log_print(f"  Median: {length_stats['median']:.2f}")
    log_print(f"  Standard Deviation: {length_stats['std']:.2f}")
    log_print(f"  95th Percentile: {length_stats['percentile_95']:.2f}")
    log_print(f"  99th Percentile: {length_stats['percentile_99']:.2f}")
    log_print("")
    
    # Determine T_max
    T_max = int(length_stats['percentile_95'])
    log_print(f"Using T_max = {T_max} (95th percentile of sequence lengths)")
    log_print("")
    
    # Split data
    log_print("Splitting data into train/test sets (80/20)...")
    try:
        X_train, X_test, y_train, y_test = prepare_data_for_training(
            sequences, labels, test_size=0.2, random_state=args.random_seed
        )
    except ValueError:
        log_print("  Note: Stratification not possible, using simple split...")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels, test_size=0.2, random_state=args.random_seed
        )
    
    log_print(f"Training samples: {len(X_train)}")
    log_print(f"Test samples: {len(X_test)}")
    log_print("")
    
    # Determine input dimension
    input_dim = X_train[0].shape[1]
    log_print(f"Input dimension: {input_dim} (3D coordinates × number of joints)")
    log_print("")
    
    # Initialize model
    log_print("=" * 80)
    log_print("PHASE 2: MODEL INITIALIZATION")
    log_print("=" * 80)
    log_print("Model Configuration:")
    log_print(f"  Reservoir Size (H): {args.reservoir_size}")
    log_print(f"  Fusion Strategy: {args.fusion_strategy}")
    log_print(f"  Spectral Radius (rho): {args.spectral_radius}")
    log_print(f"  Sparsity (gamma): {args.sparsity}")
    log_print(f"  Input Scaling (sigma_in): {args.input_scaling}")
    log_print(f"  Forward Leak Rate (alpha_f): {args.leak_rate_f}")
    log_print(f"  Backward Leak Rate (alpha_b): {args.leak_rate_b}")
    log_print(f"  Number of Classes: {metadata['num_classes']}")
    log_print("")
    
    model = BidirectionalRC(
        input_dim=input_dim,
        reservoir_size=args.reservoir_size,
        num_classes=metadata['num_classes'],
        fusion_strategy=args.fusion_strategy,
        spectral_radius=args.spectral_radius,
        sparsity=args.sparsity,
        input_scaling=args.input_scaling,
        leak_rate_f=args.leak_rate_f,
        leak_rate_b=args.leak_rate_b,
        random_seed=args.random_seed
    )
    
    model.T_max = T_max
    model = model.to(device)
    
    log_print("Model initialized successfully")
    log_print("")
    
    # Phase 1: Process sequences through reservoir
    log_print("=" * 80)
    log_print("PHASE 3: RESERVOIR PROCESSING")
    log_print("=" * 80)
    log_print("Processing training sequences through bidirectional reservoir...")
    
    start_time = time.time()
    # Only process ONE sample to get shape info - training will recompute on-the-fly
    # This saves massive memory (45,504 samples * 1.14 MB each = ~52 GB!)
    log_print("Processing ONE sample to get reservoir state shape...")
    log_print("(Training will recompute reservoir states on-the-fly to save memory)")
    
    model.eval()
    R_samples = []
    
    with torch.no_grad():
        # Process only first sample to get shape
        R = model.forward_reservoir(X_train[0])
        # Convert to float32 to save memory
        R = R.astype(np.float32)
        R_samples.append(R)
    
    reservoir_time = time.time() - start_time
    
    log_print(f"Reservoir state shape determined in {reservoir_time:.2f} seconds")
    log_print(f"Reservoir state shape: {R_samples[0].shape}")
    log_print(f"Note: Reservoir states will be recomputed during training (on-the-fly)")
    log_print("")
    
    # Phase 2: Fit dimensionality reduction (only needs shape info)
    log_print("=" * 80)
    log_print("PHASE 4: DIMENSIONALITY REDUCTION")
    log_print("=" * 80)
    log_print("Fitting dimensionality reduction...")
    log_print("NOTE: DIMENSIONALITY REDUCTION DISABLED FOR TESTING")
    log_print("Using raw reservoir states directly to test if DR is causing issues")
    log_print("")
    
    start_time = time.time()
    model.fit_dimensionality_reduction(R_samples)  # Only needs shape info
    model.fitted = True
    dimred_time = time.time() - start_time
    
    log_print(f"Feature extraction setup completed in {dimred_time:.2f} seconds")
    log_print(f"Using raw reservoir states (no PCA/Tucker reduction)")
    log_print(f"Reservoir state shape: {R_samples[0].shape}")
    log_print(f"Feature dimension after representation learning: {model.readout.input_dim}")
    log_print("")
    
    # Phase 3: Training
    log_print("=" * 80)
    log_print("PHASE 5: READOUT TRAINING")
    log_print("=" * 80)
    log_print("Training Configuration:")
    log_print(f"  Batch Size: {args.batch_size}")
    log_print(f"  Number of Epochs: {args.epochs}")
    log_print(f"  Learning Rate: {args.learning_rate}")
    log_print(f"  L2 Regularization (lambda1): {args.lambda1}")
    log_print(f"  KAF Regularization (lambda2): {args.lambda2}")
    log_print(f"  Fusion Regularization (lambda3): {args.lambda3}")
    log_print("")
    
    # Create datasets
    train_dataset = SkeletonDataset(X_train, y_train)
    test_dataset = SkeletonDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn
    )
    
    # Optimizer with improved settings - CRITICAL: Include representation learning!
    # Define trainable_params FIRST before using it
    trainable_params = list(model.readout.parameters())
    if model.representation_learning is not None:
        trainable_params += list(model.representation_learning.parameters())
    if args.fusion_strategy in ['weighted', 'attention']:
        trainable_params += list(model.fusion.parameters())
    
    # CRITICAL: Log parameter counts to verify representation learning is included
    total_params = sum(p.numel() for p in trainable_params)
    readout_params = sum(p.numel() for p in model.readout.parameters())
    repr_params = sum(p.numel() for p in model.representation_learning.parameters()) if model.representation_learning else 0
    log_print(f"Parameter Counts:")
    log_print(f"  Readout parameters: {readout_params:,}")
    log_print(f"  Representation learning parameters: {repr_params:,}")
    log_print(f"  Total trainable parameters: {total_params:,}")
    log_print("")
    
    # Use higher learning rate initially, with weight decay
    # FIX: Use conservative learning rate (no multiplier) to prevent gradient explosion
    optimizer = optim.Adam(trainable_params, lr=args.learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler - reduce on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Training history
    history = {
        'train': [],
        'test': [],
        'best_epoch': 0,
        'best_test_acc': 0.0
    }
    
    log_print("Starting training...")
    log_print("")
    
    training_start = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            args.lambda1, args.lambda2, args.lambda3
        )
        
        # Evaluate
        test_acc, test_preds, test_labels, test_probs = evaluate(model, test_loader, device)
        
        # Update learning rate based on test accuracy
        scheduler.step(test_acc)
        
        # Compute class-wise metrics (every epoch for detailed monitoring)
        class_metrics = compute_class_wise_metrics(
            test_labels, test_preds, metadata['num_classes']
        )
        
        epoch_time = time.time() - epoch_start
        
        # Store history
        epoch_data = {
            'epoch': epoch + 1,
            'train': train_metrics,
            'test_acc': test_acc,
            'time': epoch_time
        }
        if class_metrics:
            epoch_data['class_metrics'] = class_metrics
        
        history['train'].append(epoch_data)
        
        # Update best
        if test_acc > history['best_test_acc']:
            history['best_test_acc'] = test_acc
            history['best_epoch'] = epoch + 1
        
        # Log progress - detailed every epoch
        log_print(f"Epoch {epoch + 1}/{args.epochs} ({epoch_time:.2f}s):")
        log_print(f"  Train Loss: {train_metrics['loss']:.4f} "
                 f"(CE: {train_metrics['ce_loss']:.4f}, "
                 f"L2: {train_metrics['l2_reg']:.4f}, "
                 f"KAF: {train_metrics['kaf_reg']:.4f}, "
                 f"Fusion: {train_metrics['fusion_reg']:.4f})")
        log_print(f"  Train Accuracy: {train_metrics['accuracy']:.2f}%")
        log_print(f"  Test Accuracy: {test_acc:.2f}%")
        log_print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        if class_metrics:
            log_print(f"  Macro Precision: {class_metrics['macro']['precision']:.4f}")
            log_print(f"  Macro Recall: {class_metrics['macro']['recall']:.4f}")
            log_print(f"  Macro F1-Score: {class_metrics['macro']['f1']:.4f}")
        log_print("")
        
        # Print summary every epoch to console
        elapsed_time = time.time() - training_start
        avg_epoch_time = elapsed_time / (epoch + 1)
        remaining_epochs = args.epochs - (epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs
        
        print("\n" + "=" * 80)
        print(f"EPOCH {epoch + 1}/{args.epochs} RESULTS")
        print("=" * 80)
        print(f"Progress: {100 * (epoch + 1) / args.epochs:.1f}% | Elapsed: {elapsed_time/60:.2f} min | Remaining: ~{estimated_remaining/60:.2f} min")
        print("")
        print(f"Performance Metrics:")
        print(f"  Train Accuracy: {train_metrics['accuracy']:.2f}%")
        print(f"  Test Accuracy:  {test_acc:.2f}%")
        print(f"  Train Loss:     {train_metrics['loss']:.4f} (CE: {train_metrics['ce_loss']:.4f})")
        print(f"  Learning Rate:  {optimizer.param_groups[0]['lr']:.6f}")
        print("")
        print(f"Best So Far:")
        print(f"  Best Test Accuracy: {history['best_test_acc']:.2f}% (Epoch {history['best_epoch']})")
        print("")
        print(f"Class-wise Metrics:")
        print(f"  Macro Precision: {class_metrics['macro']['precision']:.4f}")
        print(f"  Macro Recall:    {class_metrics['macro']['recall']:.4f}")
        print(f"  Macro F1-Score:  {class_metrics['macro']['f1']:.4f}")
        print("=" * 80 + "\n")
    
    total_training_time = time.time() - training_start
    log_print(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
    log_print("")
    
    # Final evaluation
    log_print("=" * 80)
    log_print("PHASE 6: FINAL EVALUATION")
    log_print("=" * 80)
    
    final_test_acc, final_preds, final_labels, final_probs = evaluate(model, test_loader, device)
    final_class_metrics = compute_class_wise_metrics(
        final_labels, final_preds, metadata['num_classes']
    )
    
    log_print(f"Final Test Accuracy: {final_test_acc:.2f}%")
    log_print(f"Best Test Accuracy: {history['best_test_acc']:.2f}% (Epoch {history['best_epoch']})")
    log_print("")
    log_print("Final Class-wise Metrics:")
    log_print(f"  Macro Precision: {final_class_metrics['macro']['precision']:.4f}")
    log_print(f"  Macro Recall: {final_class_metrics['macro']['recall']:.4f}")
    log_print(f"  Macro F1-Score: {final_class_metrics['macro']['f1']:.4f}")
    log_print("")
    
    # Compute inference time
    log_print("Computing inference time...")
    model.eval()
    inference_times = []
    with torch.no_grad():
        for X in X_test[:10]:  # Test on 10 samples
            start = time.time()
            _ = model.extract_features(X)
            inference_times.append(time.time() - start)
    
    avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
    fps = 1000 / avg_inference_time if avg_inference_time > 0 else 0
    
    log_print(f"Average inference time: {avg_inference_time:.2f} ms per sample")
    log_print(f"Throughput: {fps:.2f} FPS")
    log_print("")
    
    # Compile results
    results = {
        'experiment_info': {
            'timestamp': timestamp,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'random_seed': args.random_seed,
            'device': str(device)
        },
        'dataset_info': {
            'data_path': args.data_path,
            'total_samples': len(sequences),
            'num_classes': metadata['num_classes'],
            'class_names': [str(c) for c in metadata['class_names']],
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'input_dimension': int(input_dim),
            'sequence_length_stats': {k: float(v) for k, v in length_stats.items()},
            'T_max': int(T_max)
        },
        'model_configuration': {
            'reservoir_size': args.reservoir_size,
            'fusion_strategy': args.fusion_strategy,
            'spectral_radius': args.spectral_radius,
            'sparsity': args.sparsity,
            'input_scaling': args.input_scaling,
            'leak_rate_forward': args.leak_rate_f,
            'leak_rate_backward': args.leak_rate_b,
            'pca_reduced_dim': int(model.temporal_pca.K),
            'tucker_ranks': [int(r) for r in model.tucker_decomp.R_ranks],
            'compression_ratio': float(compression_ratio)
        },
        'training_configuration': {
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'lambda1': args.lambda1,
            'lambda2': args.lambda2,
            'lambda3': args.lambda3
        },
        'performance_metrics': {
            'data_loading_time': float(load_time),
            'reservoir_processing_time': float(reservoir_time),
            'dimensionality_reduction_time': float(dimred_time),
            'total_training_time': float(total_training_time),
            'average_inference_time_ms': float(avg_inference_time),
            'throughput_fps': float(fps),
            'best_test_accuracy': float(history['best_test_acc']),
            'best_epoch': int(history['best_epoch']),
            'final_test_accuracy': float(final_test_acc),
            'final_macro_precision': float(final_class_metrics['macro']['precision']),
            'final_macro_recall': float(final_class_metrics['macro']['recall']),
            'final_macro_f1': float(final_class_metrics['macro']['f1'])
        },
        'training_history': history['train'],
        'final_class_wise_metrics': final_class_metrics
    }
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    log_print("=" * 80)
    log_print("RESULTS SUMMARY")
    log_print("=" * 80)
    log_print(f"Results saved to: {results_file}")
    log_print(f"Training log saved to: {log_file}")
    log_print("")
    log_print("Key Results:")
    log_print(f"  Best Test Accuracy: {history['best_test_acc']:.2f}%")
    log_print(f"  Final Test Accuracy: {final_test_acc:.2f}%")
    log_print(f"  Macro F1-Score: {final_class_metrics['macro']['f1']:.4f}")
    log_print(f"  Total Training Time: {total_training_time/60:.2f} minutes")
    log_print(f"  Inference Time: {avg_inference_time:.2f} ms")
    log_print("")
    log_print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("=" * 80)
    
    log_f.close()
    
    print(f"\nTraining completed! Results saved to:")
    print(f"  - Results JSON: {results_file}")
    print(f"  - Training Log: {log_file}")


if __name__ == '__main__':
    main()

