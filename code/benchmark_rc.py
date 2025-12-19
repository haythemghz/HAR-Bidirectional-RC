import time
import numpy as np
import psutil
import os
import sys

# Add path to the module
sys.path.append(os.path.abspath("Source Code and Dataset/Bidirectional_RC_Implementation"))
from dimensionality_reduction import TuckerDecomposition

def benchmark_tucker():
    print("Benchmarking Tucker Decomposition...")
    # Simulated reservoir states for a batch
    T_max = 100
    K = 1000  # After PCA
    P = 100   # Batch size for fitting
    
    R_tensor = np.random.randn(P, T_max, K)
    tucker = TuckerDecomposition(variance_thresholds=(0.95, 0.95, 0.95))
    
    # Measure Fit Time
    start = time.time()
    tucker.fit(R_tensor)
    fit_time = time.time() - start
    print(f"HOSVD Fit Time (Batch of {P}): {fit_time:.4f}s")
    
    # Measure Transform Time (Inference)
    R_i = np.random.randn(T_max, K)
    start = time.time()
    # Warmup
    for _ in range(10):
        _ = tucker.transform(R_i)
    
    N_runs = 100
    start = time.time()
    for _ in range(N_runs):
        _ = tucker.transform(R_i)
    transform_time = (time.time() - start) / N_runs
    print(f"Tucker Transform Time (Inference): {transform_time*1000:.4f}ms")

def benchmark_memory():
    print("\nMeasuring Memory Footprint...")
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    
    # Simulate a full reservoir for NTU-60 (approx)
    # 56000 samples * 100 timesteps * 1000 units? No, usually batch processing.
    # Typical batch = 256
    batch_size = 256
    T = 100
    units = 1000
    
    # Reservoir activations (double precision)
    activations = np.zeros((batch_size, T, units))
    mem_after = process.memory_info().rss / (1024 * 1024)
    print(f"Memory overhead for batch size {batch_size}: {mem_after - mem_before:.2f} MB")
    
    # Weights overhead
    W_res = np.zeros((units, units)) 
    mem_weights = process.memory_info().rss / (1024 * 1024)
    print(f"Memory overhead for 1000 units reservoir weights: {mem_weights - mem_after:.2f} MB (if dense)")

if __name__ == "__main__":
    benchmark_tucker()
    benchmark_memory()
