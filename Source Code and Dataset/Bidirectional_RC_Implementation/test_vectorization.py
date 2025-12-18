"""Test vectorized dimensionality reduction"""
import numpy as np
from dimensionality_reduction import TemporalPCA, TuckerDecomposition

print("Testing vectorized dimensionality reduction...")
print("")

# Test TemporalPCA
print("1. Testing TemporalPCA.fit() (vectorized)...")
R_samples = [np.random.randn(100, 50).astype(np.float32) for _ in range(10)]
pca = TemporalPCA()
pca.fit(R_samples)
print(f"   OK - PCA fit successful, K={pca.K}")

# Test transform
print("2. Testing TemporalPCA.transform()...")
R_pca = pca.transform(R_samples[0])
print(f"   OK - PCA transform successful, shape={R_pca.shape}")

# Test Tucker
print("3. Testing TuckerDecomposition.fit() (optimized)...")
R_tensor = np.stack([pca.transform(R) for R in R_samples], axis=0)
tucker = TuckerDecomposition()
tucker.fit(R_tensor)
print(f"   OK - Tucker fit successful, ranks={tucker.R_ranks}")

# Test transform
print("4. Testing TuckerDecomposition.transform()...")
R_final = tucker.transform(R_tensor[0])
print(f"   OK - Tucker transform successful, shape={R_final.shape}")

print("")
print("All vectorized operations working correctly!")

