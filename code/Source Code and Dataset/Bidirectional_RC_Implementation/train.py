"""
Training Script for Bidirectional Reservoir Computing
Implements Algorithm 1 from the paper
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
from tqdm import tqdm
import json
from datetime import datetime

from model import BidirectionalRC
from data_utils import (
    load_skeleton_data, normalize_sequences, prepare_data_for_training,
    compute_sequence_lengths, create_dummy_data
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
    correct = 0
    total = 0
    
    for sequences, labels in tqdm(dataloader, desc="Training"):
        labels = labels.to(device)
        
        # Forward pass
        logits = model(sequences)
        
        # Compute loss
        loss, components = model.compute_total_loss(
            logits, labels, lambda1, lambda2, lambda3
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        total_ce_loss += components['ce_loss']
        
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, avg_ce_loss, accuracy


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc="Evaluating"):
            labels = labels.to(device)
            
            logits = model(sequences)
            preds = torch.argmax(logits, dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100.0 * correct / total
    
    return accuracy, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='Train Bidirectional RC for HAR')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to skeleton data directory')
    parser.add_argument('--use_dummy_data', action='store_true',
                       help='Use dummy data for testing')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of dummy samples')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of action classes')
    
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
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    if args.use_dummy_data:
        print("Using dummy data for testing...")
        sequences, labels, metadata = create_dummy_data(
            num_samples=args.num_samples,
            num_classes=args.num_classes,
            random_seed=args.random_seed
        )
    else:
        if args.data_path is None:
            raise ValueError("Must provide --data_path or use --use_dummy_data")
        
        print(f"Loading data from {args.data_path}...")
        sequences, labels, metadata = load_skeleton_data(args.data_path)
    
    print(f"Loaded {len(sequences)} samples, {metadata['num_classes']} classes")
    
    # Normalize sequences
    sequences, scaler = normalize_sequences(sequences, method='standard')
    
    # Compute sequence length statistics
    length_stats = compute_sequence_lengths(sequences)
    print(f"Sequence length stats: {length_stats}")
    
    # Determine T_max (95th percentile)
    T_max = int(length_stats['percentile_95'])
    print(f"Using T_max = {T_max}")
    
    # Split data
    X_train, X_test, y_train, y_test = prepare_data_for_training(
        sequences, labels, test_size=0.2, random_state=args.random_seed
    )
    
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Determine input dimension
    input_dim = X_train[0].shape[1]
    print(f"Input dimension: {input_dim}")
    
    # Initialize model
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
    
    print("Model initialized")
    print(f"Fusion strategy: {args.fusion_strategy}")
    
    # Phase 1: Process all training samples through reservoir
    print("\nPhase 1: Processing sequences through reservoir...")
    R_samples = []
    
    model.eval()
    with torch.no_grad():
        for X in tqdm(X_train, desc="Reservoir processing"):
            R = model.forward_reservoir(X)
            R_samples.append(R)
    
    print(f"Generated {len(R_samples)} reservoir state sequences")
    
    # Phase 2: Fit dimensionality reduction
    print("\nPhase 2: Fitting dimensionality reduction...")
    model.fit_dimensionality_reduction(R_samples)
    model.fitted = True
    print("Dimensionality reduction fitted")
    
    # Phase 3: Training readout
    print("\nPhase 3: Training readout...")
    
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
    
    # Optimizer
    optimizer = optim.Adam(
        list(model.readout.parameters()) + 
        (list(model.fusion.parameters()) if args.fusion_strategy in ['weighted', 'attention'] else []),
        lr=args.learning_rate
    )
    
    # Training loop
    best_acc = 0.0
    history = {
        'train_loss': [], 'train_ce_loss': [], 'train_acc': [],
        'test_acc': []
    }
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss, train_ce_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device,
            args.lambda1, args.lambda2, args.lambda3
        )
        
        # Evaluate
        test_acc, _, _ = evaluate(model, test_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train CE Loss: {train_ce_loss:.4f}")
        print(f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_ce_loss'].append(train_ce_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'args': vars(args),
                'metadata': metadata
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Saved best model with test accuracy: {test_acc:.2f}%")
    
    print(f"\nTraining completed!")
    print(f"Best test accuracy: {best_acc:.2f}%")
    
    # Save training history
    history_path = os.path.join(args.save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")


if __name__ == '__main__':
    main()











