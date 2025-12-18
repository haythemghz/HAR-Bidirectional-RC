"""
Comprehensive audit of the entire pipeline from data preprocessing to readout
Tests each component to identify where accuracy stagnation occurs
"""

import numpy as np
import torch
import torch.nn.functional as F
from data_utils import load_skeleton_data, normalize_sequences, prepare_data_for_training
from model import BidirectionalRC
from sklearn.metrics import accuracy_score
import sys

def test_data_preprocessing(data_path, max_samples=50):
    """Test data loading and preprocessing"""
    print("=" * 80)
    print("TEST 1: DATA PREPROCESSING")
    print("=" * 80)
    
    sequences, labels, metadata = load_skeleton_data(
        data_path, file_pattern="*.skeleton", max_samples=max_samples
    )
    
    print(f"[OK] Loaded {len(sequences)} samples")
    print(f"[OK] Number of classes: {metadata['num_classes']}")
    print(f"[OK] Sequence shapes: {[s.shape for s in sequences[:3]]}")
    print(f"[OK] Label distribution: {np.bincount(labels)}")
    
    # Normalize
    sequences, scaler = normalize_sequences(sequences, method='standard')
    print(f"[OK] Normalized sequences")
    print(f"[OK] Sample mean after normalization: {np.mean([np.mean(s) for s in sequences]):.6f}")
    print(f"[OK] Sample std after normalization: {np.std([np.std(s) for s in sequences]):.6f}")
    
    # Check for NaN/Inf
    has_nan = any(np.isnan(s).any() for s in sequences)
    has_inf = any(np.isinf(s).any() for s in sequences)
    print(f"[OK] Has NaN: {has_nan}")
    print(f"[OK] Has Inf: {has_inf}")
    
    if has_nan or has_inf:
        print("⚠ WARNING: Data contains NaN or Inf values!")
    
    return sequences, labels, metadata

def test_reservoir_processing(model, sequences):
    """Test reservoir processing"""
    print("\n" + "=" * 80)
    print("TEST 2: RESERVOIR PROCESSING")
    print("=" * 80)
    
    model.eval()
    reservoir_states = []
    
    with torch.no_grad():
        for i, seq in enumerate(sequences[:10]):  # Test first 10
            R = model.forward_reservoir(seq)
            reservoir_states.append(R)
            
            if i == 0:
                print(f"[OK] Input shape: {seq.shape}")
                print(f"[OK] Reservoir output shape: {R.shape}")
                print(f"[OK] Reservoir state mean: {np.mean(R):.6f}")
                print(f"[OK] Reservoir state std: {np.std(R):.6f}")
                print(f"[OK] Reservoir state min: {np.min(R):.6f}")
                print(f"[OK] Reservoir state max: {np.max(R):.6f}")
    
    # Check for issues
    has_nan = any(np.isnan(R).any() for R in reservoir_states)
    has_inf = any(np.isinf(R).any() for R in reservoir_states)
    all_zero = any(np.allclose(R, 0) for R in reservoir_states)
    
    print(f"[OK] Has NaN: {has_nan}")
    print(f"[OK] Has Inf: {has_inf}")
    print(f"[OK] All zero states: {all_zero}")
    
    if has_nan or has_inf or all_zero:
        print("⚠ WARNING: Reservoir states have issues!")
    
    return reservoir_states

def test_feature_extraction(model, sequences, training=False):
    """Test feature extraction"""
    print("\n" + "=" * 80)
    print(f"TEST 3: FEATURE EXTRACTION (training={training})")
    print("=" * 80)
    
    features = []
    
    for i, seq in enumerate(sequences[:10]):
        if training:
            f = model.extract_features(seq, training=True)
            if isinstance(f, torch.Tensor):
                f = f.detach().cpu().numpy()
        else:
            f = model.extract_features(seq, training=False)
        
        features.append(f)
        
        if i == 0:
            print(f"[OK] Input shape: {seq.shape}")
            print(f"[OK] Feature shape: {f.shape}")
            print(f"[OK] Feature mean: {np.mean(f):.6f}")
            print(f"[OK] Feature std: {np.std(f):.6f}")
            print(f"[OK] Feature min: {np.min(f):.6f}")
            print(f"[OK] Feature max: {np.max(f):.6f}")
    
    # Check for issues
    has_nan = any(np.isnan(f).any() for f in features)
    has_inf = any(np.isinf(f).any() for f in features)
    all_zero = any(np.allclose(f, 0) for f in features)
    all_same = len(set([tuple(f.flatten()[:10]) for f in features[:5]])) == 1
    
    print(f"[OK] Has NaN: {has_nan}")
    print(f"[OK] Has Inf: {has_inf}")
    print(f"[OK] All zero: {all_zero}")
    print(f"[OK] All same (first 5): {all_same}")
    
    if has_nan or has_inf or all_zero or all_same:
        print("⚠ WARNING: Features have issues!")
        if all_same:
            print("⚠ CRITICAL: All features are identical - model not learning!")
    
    return features

def test_readout(model, features, labels):
    """Test readout layer"""
    print("\n" + "=" * 80)
    print("TEST 4: READOUT LAYER")
    print("=" * 80)
    
    model.readout.eval()
    features_tensor = torch.from_numpy(np.stack(features)).float()
    labels_tensor = torch.from_numpy(np.array(labels[:len(features)])).long()
    
    with torch.no_grad():
        logits = model.readout(features_tensor)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
    
    print(f"[OK] Features shape: {features_tensor.shape}")
    print(f"[OK] Logits shape: {logits.shape}")
    print(f"[OK] Logits mean: {torch.mean(logits):.6f}")
    print(f"[OK] Logits std: {torch.std(logits):.6f}")
    print(f"[OK] Logits min: {torch.min(logits):.6f}")
    print(f"[OK] Logits max: {torch.max(logits):.6f}")
    print(f"[OK] Probabilities mean: {torch.mean(probs):.6f}")
    print(f"[OK] Predictions: {preds.numpy()}")
    print(f"[OK] True labels: {labels_tensor.numpy()}")
    
    acc = accuracy_score(labels_tensor.numpy(), preds.numpy())
    print(f"[OK] Accuracy: {acc:.4f}")
    
    # Check for issues
    logits_nan = torch.isnan(logits).any()
    logits_inf = torch.isinf(logits).any()
    logits_all_same = torch.allclose(logits[0], logits[1])
    probs_flat = torch.allclose(probs, torch.ones_like(probs) / probs.shape[1])
    
    print(f"[OK] Logits has NaN: {logits_nan}")
    print(f"[OK] Logits has Inf: {logits_inf}")
    print(f"[OK] Logits all same: {logits_all_same}")
    print(f"[OK] Probs uniform: {probs_flat}")
    
    if logits_nan or logits_inf or logits_all_same or probs_flat:
        print("[WARNING] Readout has issues!")
        if logits_all_same:
            print("[CRITICAL] All logits are identical - model outputs same prediction for all samples!")
    
    return logits, preds, acc

def test_gradient_flow(model, sequences, labels):
    """Test if gradients flow properly"""
    print("\n" + "=" * 80)
    print("TEST 5: GRADIENT FLOW")
    print("=" * 80)
    
    model.train()
    if model.representation_learning:
        model.representation_learning.train()
    
    # Forward pass
    features_list = []
    for seq in sequences[:5]:
        f = model.extract_features(seq, training=True)
        if isinstance(f, torch.Tensor):
            features_list.append(f)
        else:
            features_list.append(torch.from_numpy(f).float())
    
    features_tensor = torch.stack(features_list)
    labels_tensor = torch.from_numpy(np.array(labels[:5])).long()
    
    logits = model.readout(features_tensor)
    loss = F.cross_entropy(logits, labels_tensor)
    
    # Backward pass
    model.readout.zero_grad()
    if model.representation_learning:
        model.representation_learning.zero_grad()
    
    loss.backward()
    
    # Check gradients
    readout_grads = [p.grad is not None and p.grad.abs().sum() > 0 
                     for p in model.readout.parameters() if p.requires_grad]
    repr_grads = []
    if model.representation_learning:
        repr_grads = [p.grad is not None and p.grad.abs().sum() > 0 
                     for p in model.representation_learning.parameters() if p.requires_grad]
    
    print(f"[OK] Loss: {loss.item():.6f}")
    print(f"[OK] Readout has gradients: {sum(readout_grads)}/{len(readout_grads)}")
    if repr_grads:
        print(f"[OK] Representation learning has gradients: {sum(repr_grads)}/{len(repr_grads)}")
    else:
        print("[WARNING] Representation learning not found or has no parameters!")
    
    if not all(readout_grads):
        print("⚠ WARNING: Some readout parameters have no gradients!")
    if repr_grads and not all(repr_grads):
        print("⚠ WARNING: Some representation learning parameters have no gradients!")
    
    return loss.item(), all(readout_grads), all(repr_grads) if repr_grads else False

def test_training_step(model, sequences, labels, optimizer):
    """Test a single training step"""
    print("\n" + "=" * 80)
    print("TEST 6: TRAINING STEP")
    print("=" * 80)
    
    model.train()
    if model.representation_learning:
        model.representation_learning.train()
    
    # Get a batch
    batch_seqs = sequences[:5]
    batch_labels = torch.from_numpy(np.array(labels[:5])).long()
    
    # Forward
    logits = model(batch_seqs, training=True)
    loss = F.cross_entropy(logits, batch_labels)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients before clipping
    total_grad_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.data.norm(2).item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    
    print(f"[OK] Loss: {loss.item():.6f}")
    print(f"[OK] Gradient norm: {total_grad_norm:.6f}")
    
    # Clip and step
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    
    # Check predictions
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(batch_labels.numpy(), preds.numpy())
    
    print(f"[OK] Accuracy after step: {acc:.4f}")
    print(f"[OK] Predictions: {preds.numpy()}")
    print(f"[OK] True labels: {batch_labels.numpy()}")
    
    return loss.item(), acc

def main():
    print("COMPREHENSIVE PIPELINE AUDIT")
    print("=" * 80)
    print("Testing entire pipeline from data preprocessing to readout")
    print("=" * 80)
    print()
    
    # Load small dataset
    data_path = r"..\nturgbd_skeletons_s001_to_s017\nturgb+d_skeletons"
    max_samples = 50
    
    try:
        # Test 1: Data preprocessing
        sequences, labels, metadata = test_data_preprocessing(data_path, max_samples)
        
        # Prepare data (use simple split if stratification fails)
        try:
            X_train, X_test, y_train, y_test = prepare_data_for_training(
                sequences, labels, test_size=0.2, random_state=42
            )
        except ValueError:
            # Fallback to simple split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                sequences, labels, test_size=0.2, random_state=42
            )
        
        # Determine T_max
        T_max = int(np.percentile([len(s) for s in X_train], 95))
        print(f"\n[OK] Using T_max = {T_max}")
        
        # Initialize model
        print("\n" + "=" * 80)
        print("INITIALIZING MODEL")
        print("=" * 80)
        model = BidirectionalRC(
            input_dim=X_train[0].shape[1],
            reservoir_size=200,  # Smaller for testing
            num_classes=metadata['num_classes'],
            fusion_strategy='concat',
            random_seed=42
        )
        model.T_max = T_max
        print("[OK] Model initialized")
        
        # Process through reservoir
        print("\nProcessing samples through reservoir...")
        R_samples = []
        model.eval()
        with torch.no_grad():
            for X in X_train[:20]:
                R = model.forward_reservoir(X)
                R_samples.append(R)
        
        # Fit dimensionality reduction (or skip if disabled)
        print("\nFitting feature extraction...")
        model.fit_dimensionality_reduction(R_samples)
        model.fitted = True  # Ensure fitted flag is set
        print("[OK] Feature extraction fitted")
        
        # Test 2: Reservoir processing
        test_reservoir_processing(model, X_train[:10])
        
        # Test 3: Feature extraction (inference)
        features_inf = test_feature_extraction(model, X_train[:10], training=False)
        
        # Test 3b: Feature extraction (training)
        features_train = test_feature_extraction(model, X_train[:10], training=True)
        
        # Test 4: Readout
        logits, preds, acc = test_readout(model, features_inf, y_train[:10])
        
        # Test 5: Gradient flow
        loss, readout_grads, repr_grads = test_gradient_flow(model, X_train[:5], y_train[:5])
        
        # Test 6: Training step
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_loss, train_acc = test_training_step(model, X_train[:5], y_train[:5], optimizer)
        
        # Summary
        print("\n" + "=" * 80)
        print("AUDIT SUMMARY")
        print("=" * 80)
        print(f"Initial readout accuracy: {acc:.4f}")
        print(f"After training step accuracy: {train_acc:.4f}")
        print(f"Readout gradients flowing: {readout_grads}")
        print(f"Representation learning gradients flowing: {repr_grads}")
        print()
        
        if acc < 0.1 and train_acc < 0.1:
            print("⚠ ISSUE: Very low accuracy - model may not be learning")
        if not readout_grads:
            print("⚠ ISSUE: Readout gradients not flowing")
        if not repr_grads:
            print("⚠ ISSUE: Representation learning gradients not flowing")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

