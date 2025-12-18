"""
Test script for Bidirectional RC on NTU RGB+D dataset (small subset)
"""

import numpy as np
import torch
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import BidirectionalRC
from data_utils import (
    load_skeleton_data, normalize_sequences, prepare_data_for_training,
    compute_sequence_lengths
)
from train import SkeletonDataset, collate_fn
from torch.utils.data import DataLoader

def main():
    print("=" * 60)
    print("Testing Bidirectional RC on NTU RGB+D Dataset (Small Subset)")
    print("=" * 60)
    
    # Path to skeleton data
    data_path = os.path.join("..", "nturgbd_skeletons_s001_to_s017", "nturgb+d_skeletons")
    
    if not os.path.exists(data_path):
        print(f"Error: Data path not found: {data_path}")
        print("Please check the path to your skeleton data.")
        return
    
    # Load a small subset of data (50 samples for quick testing)
    print(f"\nLoading skeleton data from: {data_path}")
    print("Loading first 50 samples for testing...")
    
    sequences, labels, metadata = load_skeleton_data(
        data_path, 
        file_pattern="*.skeleton",
        max_samples=50  # Limit to 50 samples for quick test
    )
    
    if len(sequences) == 0:
        print("Error: No sequences loaded. Check data path and file format.")
        return
    
    print(f"\nLoaded {len(sequences)} samples")
    print(f"Number of classes: {metadata['num_classes']}")
    print(f"Class names: {metadata['class_names'][:10]}...")  # Show first 10
    
    # Normalize sequences
    print("\nNormalizing sequences...")
    sequences, scaler = normalize_sequences(sequences, method='standard')
    
    # Compute sequence length statistics
    length_stats = compute_sequence_lengths(sequences)
    print(f"\nSequence length statistics:")
    print(f"  Min: {length_stats['min']}")
    print(f"  Max: {length_stats['max']}")
    print(f"  Mean: {length_stats['mean']:.1f}")
    print(f"  Median: {length_stats['median']:.1f}")
    print(f"  95th percentile: {length_stats['percentile_95']:.1f}")
    
    # Determine T_max (95th percentile)
    T_max = int(length_stats['percentile_95'])
    print(f"\nUsing T_max = {T_max}")
    
    # Split data (80/20)
    print("\nSplitting data into train/test sets...")
    try:
        X_train, X_test, y_train, y_test = prepare_data_for_training(
            sequences, labels, test_size=0.2, random_state=42
        )
    except ValueError as e:
        # If stratification fails (not enough samples per class), use simple split
        print("  Note: Stratification not possible, using simple split...")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels, test_size=0.2, random_state=42
        )
    
    print(f"Train: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Determine input dimension
    input_dim = X_train[0].shape[1]
    print(f"\nInput dimension: {input_dim} (should be 75 for 25 joints * 3D)")
    
    # Initialize model with smaller reservoir for testing
    print("\n" + "=" * 60)
    print("Initializing Bidirectional RC Model")
    print("=" * 60)
    
    model = BidirectionalRC(
        input_dim=input_dim,
        reservoir_size=500,  # Smaller for quick testing
        num_classes=metadata['num_classes'],
        fusion_strategy='concat',
        spectral_radius=0.95,
        sparsity=0.05,
        input_scaling=0.5,
        leak_rate_f=0.3,
        leak_rate_b=0.3,
        random_seed=42
    )
    
    model.T_max = T_max
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Phase 1: Process sequences through reservoir
    print("\n" + "=" * 60)
    print("Phase 1: Processing sequences through reservoir")
    print("=" * 60)
    
    R_samples = []
    model.eval()
    
    with torch.no_grad():
        for i, X in enumerate(X_train):
            if (i + 1) % 5 == 0:
                print(f"  Processing sample {i+1}/{len(X_train)}...")
            R = model.forward_reservoir(X)
            R_samples.append(R)
    
    print(f"\nGenerated {len(R_samples)} reservoir state sequences")
    print(f"Reservoir state shape: {R_samples[0].shape}")
    
    # Phase 2: Fit dimensionality reduction
    print("\n" + "=" * 60)
    print("Phase 2: Fitting dimensionality reduction")
    print("=" * 60)
    
    model.fit_dimensionality_reduction(R_samples)
    model.fitted = True
    
    print("Dimensionality reduction fitted successfully!")
    print(f"  PCA reduced dimension: {model.temporal_pca.K}")
    print(f"  Tucker ranks: {model.tucker_decomp.R_ranks}")
    
    # Phase 3: Quick training test
    print("\n" + "=" * 60)
    print("Phase 3: Testing readout (quick training)")
    print("=" * 60)
    
    # Create datasets
    train_dataset = SkeletonDataset(X_train, y_train)
    test_dataset = SkeletonDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.readout.parameters(), lr=0.001)
    
    # Train for a few epochs
    num_epochs = 5
    print(f"\nTraining for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for sequences_batch, labels_batch in train_loader:
            labels_batch = labels_batch.to(device)
            
            # Forward pass
            logits = model(sequences_batch)
            
            # Compute loss
            loss, components = model.compute_total_loss(
                logits, labels_batch, lambda1=0.001, lambda2=0.0001, lambda3=0.01
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels_batch).sum().item()
            train_total += labels_batch.size(0)
        
        train_acc = 100.0 * train_correct / train_total
        avg_loss = train_loss / len(train_loader)
        
        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for sequences_batch, labels_batch in test_loader:
                labels_batch = labels_batch.to(device)
                logits = model(sequences_batch)
                preds = torch.argmax(logits, dim=1)
                test_correct += (preds == labels_batch).sum().item()
                test_total += labels_batch.size(0)
        
        test_acc = 100.0 * test_correct / test_total
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Acc: {test_acc:.2f}%")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    print(f"\nFinal Results:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Number of classes: {metadata['num_classes']}")
    print(f"  Final test accuracy: {test_acc:.2f}%")
    print("\nNote: This is a quick test with limited data and epochs.")
    print("For full training, use train.py with more samples and epochs.")


if __name__ == '__main__':
    main()
