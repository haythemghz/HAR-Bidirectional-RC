"""
Example usage of the Bidirectional RC framework
"""

import numpy as np
import torch
from model import BidirectionalRC
from data_utils import create_dummy_data, normalize_sequences

# Create dummy data for demonstration
print("Creating dummy data...")
sequences, labels, metadata = create_dummy_data(
    num_samples=50,
    num_classes=5,
    num_joints=25,
    random_seed=42
)

# Normalize sequences
sequences, scaler = normalize_sequences(sequences, method='standard')

# Determine T_max (95th percentile of sequence lengths)
lengths = [len(seq) for seq in sequences]
T_max = int(np.percentile(lengths, 95))
print(f"Using T_max = {T_max}")

# Initialize model
print("\nInitializing Bidirectional RC model...")
model = BidirectionalRC(
    input_dim=75,  # 3D * 25 joints
    reservoir_size=500,  # Smaller for demo
    num_classes=metadata['num_classes'],
    fusion_strategy='concat',
    spectral_radius=0.95,
    sparsity=0.05,
    random_seed=42
)

model.T_max = T_max

# Phase 1: Process sequences through reservoir
print("\nProcessing sequences through reservoir...")
R_samples = []
for X in sequences[:30]:  # Use first 30 for training
    R = model.forward_reservoir(X)
    R_samples.append(R)

print(f"Generated {len(R_samples)} reservoir state sequences")

# Phase 2: Fit dimensionality reduction
print("\nFitting dimensionality reduction...")
model.fit_dimensionality_reduction(R_samples)
model.fitted = True
print("Model fitted successfully!")

# Phase 3: Test inference
print("\nTesting inference on a sample...")
test_sequence = sequences[30]
features = model.extract_features(test_sequence)
print(f"Extracted features shape: {features.shape}")

# Forward pass through complete model
logits = model([test_sequence])
prediction = torch.argmax(logits, dim=1)
print(f"Prediction: Class {prediction.item()}")

print("\nExample completed successfully!")











