"""
Debug script to test readout layer directly
"""

import torch
import torch.nn.functional as F
import numpy as np
from readout import AdvancedReadout

print("Testing Readout Layer Directly")
print("=" * 80)

# Create readout
readout = AdvancedReadout(input_dim=2800, hidden_dim1=512, hidden_dim2=256, num_classes=50)

# Create different input features
features1 = torch.randn(1, 2800) * 0.1
features2 = torch.randn(1, 2800) * 0.1 + 0.5  # Different features
features_batch = torch.stack([features1.squeeze(), features2.squeeze()])

print(f"Feature 1 mean: {features1.mean():.6f}, std: {features1.std():.6f}")
print(f"Feature 2 mean: {features2.mean():.6f}, std: {features2.std():.6f}")
print(f"Features different: {not torch.allclose(features1, features2)}")
print()

# Forward pass
readout.eval()
with torch.no_grad():
    logits1 = readout(features1)
    logits2 = readout(features2)
    logits_batch = readout(features_batch)

print(f"Logits 1: {logits1}")
print(f"Logits 2: {logits2}")
print(f"Logits batch: {logits_batch}")
print()
print(f"Logits 1 mean: {logits1.mean():.6f}, std: {logits1.std():.6f}")
print(f"Logits 2 mean: {logits2.mean():.6f}, std: {logits2.std():.6f}")
print(f"Logits identical: {torch.allclose(logits1, logits2)}")
print(f"Batch logits identical: {torch.allclose(logits_batch[0], logits_batch[1])}")
print()

# Check intermediate activations
readout.eval()
with torch.no_grad():
    # Layer 1
    z1_1 = readout.W1(features1) + readout.b1
    z1_1 = z1_1.view(1, readout.hidden_dim1, -1)
    z1_1 = readout.maxout(z1_1)
    z1_1 = F.relu(z1_1)
    
    z1_2 = readout.W1(features2) + readout.b1
    z1_2 = z1_2.view(1, readout.hidden_dim1, -1)
    z1_2 = readout.maxout(z1_2)
    z1_2 = F.relu(z1_2)
    
    print(f"After layer 1 - z1_1 mean: {z1_1.mean():.6f}, std: {z1_1.std():.6f}")
    print(f"After layer 1 - z1_2 mean: {z1_2.mean():.6f}, std: {z1_2.std():.6f}")
    print(f"Layer 1 outputs identical: {torch.allclose(z1_1, z1_2)}")
    print()
    
    # Layer 2
    z2_1 = readout.W2(z1_1.squeeze()) + readout.b2
    z2_1 = readout.kaf(z2_1.unsqueeze(0))
    z2_1 = F.relu(z2_1)
    
    z2_2 = readout.W2(z1_2.squeeze()) + readout.b2
    z2_2 = readout.kaf(z2_2.unsqueeze(0))
    z2_2 = F.relu(z2_2)
    
    print(f"After layer 2 - z2_1 mean: {z2_2.mean():.6f}, std: {z2_2.std():.6f}")
    print(f"After layer 2 - z2_2 mean: {z2_2.mean():.6f}, std: {z2_2.std():.6f}")
    print(f"Layer 2 outputs identical: {torch.allclose(z2_1, z2_2)}")
    print()
    
    # Output
    y1 = readout.W3(z2_1) + readout.b3
    y2 = readout.W3(z2_2) + readout.b3
    
    print(f"Output y1: {y1}")
    print(f"Output y2: {y2}")
    print(f"Outputs identical: {torch.allclose(y1, y2)}")










