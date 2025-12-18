"""
Deep audit of training process to understand why accuracy is still low
Tests actual training behavior, not just inference
"""

import numpy as np
import torch
import torch.nn.functional as F
from data_utils import load_skeleton_data, normalize_sequences, prepare_data_for_training
from model import BidirectionalRC
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import sys

class SkeletonDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    return list(sequences), np.array(labels)

def test_actual_training(data_path, max_samples=50):
    """Test actual training behavior"""
    print("=" * 80)
    print("DEEP AUDIT: ACTUAL TRAINING BEHAVIOR")
    print("=" * 80)
    print()
    
    # Load data
    sequences, labels, metadata = load_skeleton_data(
        data_path, file_pattern="*.skeleton", max_samples=max_samples
    )
    sequences, _ = normalize_sequences(sequences, method='standard')
    
    try:
        X_train, X_test, y_train, y_test = prepare_data_for_training(
            sequences, labels, test_size=0.2, random_state=42
        )
    except ValueError:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels, test_size=0.2, random_state=42
        )
    
    T_max = int(np.percentile([len(s) for s in X_train], 95))
    
    # Initialize model
    model = BidirectionalRC(
        input_dim=X_train[0].shape[1],
        reservoir_size=200,
        num_classes=metadata['num_classes'],
        fusion_strategy='concat',
        random_seed=42
    )
    model.T_max = T_max
    
    # Process through reservoir
    print("Processing through reservoir...")
    R_samples = []
    model.eval()
    with torch.no_grad():
        for X in X_train[:20]:
            R = model.forward_reservoir(X)
            R_samples.append(R)
    
    # Fit
    model.fit_dimensionality_reduction(R_samples)
    model.fitted = True
    
    # Create dataloader
    train_dataset = SkeletonDataset(X_train[:10], y_train[:10])
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=False, collate_fn=collate_fn)
    
    # Test 1: Check feature diversity
    print("\n" + "=" * 80)
    print("TEST 1: FEATURE DIVERSITY DURING TRAINING")
    print("=" * 80)
    
    model.train()
    if model.representation_learning:
        model.representation_learning.train()
    
    features_list = []
    for sequences_batch, labels_batch in train_loader:
        for seq in sequences_batch:
            f = model.extract_features(seq, training=True)
            if isinstance(f, torch.Tensor):
                f = f.detach().cpu().numpy()
            features_list.append(f)
    
    features_array = np.array(features_list)
    print(f"Features shape: {features_array.shape}")
    print(f"Features mean: {np.mean(features_array):.6f}")
    print(f"Features std: {np.std(features_array):.6f}")
    
    # Check if features are different
    feature_diffs = []
    for i in range(len(features_list)-1):
        diff = np.linalg.norm(features_list[i] - features_list[i+1])
        feature_diffs.append(diff)
    
    print(f"Average feature difference between samples: {np.mean(feature_diffs):.6f}")
    print(f"Min feature difference: {np.min(feature_diffs):.6f}")
    print(f"Max feature difference: {np.max(feature_diffs):.6f}")
    
    if np.min(feature_diffs) < 1e-6:
        print("[WARNING] Some features are nearly identical!")
    
    # Test 2: Check logits diversity
    print("\n" + "=" * 80)
    print("TEST 2: LOGITS DIVERSITY DURING TRAINING")
    print("=" * 80)
    
    model.eval()
    logits_list = []
    with torch.no_grad():
        for sequences_batch, labels_batch in train_loader:
            logits = model(sequences_batch, training=False)
            logits_list.append(logits.cpu().numpy())
    
    logits_array = np.concatenate(logits_list, axis=0)
    print(f"Logits shape: {logits_array.shape}")
    print(f"Logits mean: {np.mean(logits_array):.6f}")
    print(f"Logits std: {np.std(logits_array):.6f}")
    
    # Check if logits are different
    logit_diffs = []
    for i in range(len(logits_array)-1):
        diff = np.linalg.norm(logits_array[i] - logits_array[i+1])
        logit_diffs.append(diff)
    
    print(f"Average logit difference: {np.mean(logit_diffs):.6f}")
    print(f"Min logit difference: {np.min(logit_diffs):.6f}")
    print(f"Max logit difference: {np.max(logit_diffs):.6f}")
    
    if np.min(logit_diffs) < 1e-6:
        print("[WARNING] Some logits are nearly identical!")
    
    # Check predictions
    preds = np.argmax(logits_array, axis=1)
    print(f"Predictions: {preds}")
    print(f"Unique predictions: {len(np.unique(preds))} out of {len(preds)}")
    
    # Test 3: Check gradient flow during actual training step
    print("\n" + "=" * 80)
    print("TEST 3: GRADIENT FLOW DURING TRAINING STEP")
    print("=" * 80)
    
    model.train()
    if model.representation_learning:
        model.representation_learning.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(3):
        print(f"\nEpoch {epoch+1}:")
        for sequences_batch, labels_batch in train_loader:
            labels_tensor = torch.from_numpy(labels_batch).long()
            
            # Forward
            logits = model(sequences_batch, training=True)
            loss = F.cross_entropy(logits, labels_tensor)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Check gradients
            readout_grads = []
            for name, param in model.readout.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.abs().sum().item()
                    readout_grads.append((name, grad_norm))
            
            repr_grads = []
            if model.representation_learning:
                for name, param in model.representation_learning.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.abs().sum().item()
                        repr_grads.append((name, grad_norm))
            
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Readout gradients: {len(readout_grads)}/{sum(1 for p in model.readout.parameters() if p.requires_grad)}")
            print(f"  Representation learning gradients: {len(repr_grads)}/{sum(1 for p in model.representation_learning.parameters() if p.requires_grad) if model.representation_learning else 0}")
            
            if readout_grads:
                max_grad = max(g[1] for g in readout_grads)
                min_grad = min(g[1] for g in readout_grads)
                print(f"  Readout grad range: {min_grad:.6f} to {max_grad:.6f}")
            
            if repr_grads:
                max_grad = max(g[1] for g in repr_grads)
                min_grad = min(g[1] for g in repr_grads)
                print(f"  Representation learning grad range: {min_grad:.6f} to {max_grad:.6f}")
            
            # Check if gradients are too small
            if readout_grads and max(g[1] for g in readout_grads) < 1e-6:
                print("  [WARNING] Readout gradients are very small!")
            if repr_grads and max(g[1] for g in repr_grads) < 1e-6:
                print("  [WARNING] Representation learning gradients are very small!")
            
            optimizer.step()
            
            # Check accuracy
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                acc = accuracy_score(labels_batch, preds.numpy())
                print(f"  Accuracy: {acc:.4f}")
            
            break  # Just one batch per epoch for testing
    
    # Test 4: Check if model is actually learning
    print("\n" + "=" * 80)
    print("TEST 4: LEARNING VERIFICATION")
    print("=" * 80)
    
    # Get initial predictions
    model.eval()
    with torch.no_grad():
        initial_logits = []
        for sequences_batch, labels_batch in train_loader:
            logits = model(sequences_batch, training=False)
            initial_logits.append(logits.cpu().numpy())
        initial_logits = np.concatenate(initial_logits, axis=0)
        initial_preds = np.argmax(initial_logits, axis=1)
        initial_acc = accuracy_score(y_train[:10], initial_preds)
    
    # Train for a few steps
    model.train()
    if model.representation_learning:
        model.representation_learning.train()
    
    for step in range(10):
        for sequences_batch, labels_batch in train_loader:
            labels_tensor = torch.from_numpy(labels_batch).long()
            logits = model(sequences_batch, training=True)
            loss = F.cross_entropy(logits, labels_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            break
    
    # Get final predictions
    model.eval()
    with torch.no_grad():
        final_logits = []
        for sequences_batch, labels_batch in train_loader:
            logits = model(sequences_batch, training=False)
            final_logits.append(logits.cpu().numpy())
        final_logits = np.concatenate(final_logits, axis=0)
        final_preds = np.argmax(final_logits, axis=1)
        final_acc = accuracy_score(y_train[:10], final_preds)
    
    print(f"Initial accuracy: {initial_acc:.4f}")
    print(f"Final accuracy: {final_acc:.4f}")
    print(f"Improvement: {final_acc - initial_acc:.4f}")
    
    if final_acc <= initial_acc:
        print("[WARNING] Model is not learning - accuracy did not improve!")
    
    # Test 5: Check feature extraction consistency
    print("\n" + "=" * 80)
    print("TEST 5: FEATURE EXTRACTION CONSISTENCY")
    print("=" * 80)
    
    model.eval()
    features_inf = []
    for seq in X_train[:5]:
        f = model.extract_features(seq, training=False)
        features_inf.append(f)
    
    model.train()
    if model.representation_learning:
        model.representation_learning.train()
    features_train = []
    for seq in X_train[:5]:
        f = model.extract_features(seq, training=True)
        if isinstance(f, torch.Tensor):
            f = f.detach().cpu().numpy()
        features_train.append(f)
    
    # Compare
    for i in range(5):
        diff = np.linalg.norm(features_inf[i] - features_train[i])
        print(f"Sample {i+1} feature difference (inf vs train): {diff:.6f}")
        if diff > 1e-3:
            print(f"  [WARNING] Large difference between inference and training features!")

if __name__ == '__main__':
    data_path = r"..\nturgbd_skeletons_s001_to_s017\nturgb+d_skeletons"
    test_actual_training(data_path, max_samples=50)










