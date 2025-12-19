"""
Dimensionality Reduction Module
Implements Temporal PCA and Tucker Decomposition
"""

import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import PCA


class TemporalPCA:
    """
    Temporal Principal Component Analysis for compressing temporal dynamics.
    """
    
    def __init__(self, variance_threshold=0.95):
        """
        Initialize Temporal PCA.
        
        Args:
            variance_threshold (float): Fraction of variance to preserve (θ_PCA)
        """
        self.theta_PCA = variance_threshold
        self.V_global = None
        self.K = None
        self.fitted = False
    
    def fit(self, R_samples):
        """
        Fit temporal PCA on training samples (VECTORIZED VERSION).
        
        Args:
            R_samples: List of reservoir sequences, each of shape (T, d)
                       where d is feature dimension (H or 2H)
        """
        P = len(R_samples)
        T_max = R_samples[0].shape[0]
        d = R_samples[0].shape[1]
        
        # VECTORIZED: Stack all samples into a 3D tensor (P, T_max, d)
        R_stack = np.stack(R_samples, axis=0)  # (P, T_max, d)
        
        # VECTORIZED: Compute means for all samples at once
        mu_samples = np.mean(R_stack, axis=1, keepdims=True)  # (P, 1, d)
        R_centered = R_stack - mu_samples  # (P, T_max, d)
        
        # VECTORIZED: Compute covariance matrices for all samples
        # Using einsum for efficient batch matrix multiplication
        # C_i = (1/(T-1)) * R_centered_i^T @ R_centered_i
        # Shape: (P, d, d)
        C_samples = np.einsum('pti,ptj->pij', R_centered, R_centered) / (T_max - 1)
        
        # VECTORIZED: Compute eigenvalues for all samples
        # Note: eigh doesn't support batch, but we can use vectorized operations
        K_samples = []
        for i in range(P):
            eigenvals, _ = np.linalg.eigh(C_samples[i])
            eigenvals = np.sort(eigenvals)[::-1]  # Descending order
            
            # Find K_i that preserves theta_PCA variance
            cumsum_var = np.cumsum(eigenvals) / np.sum(eigenvals)
            K_i = np.argmax(cumsum_var >= self.theta_PCA) + 1
            K_samples.append(K_i)
        
        # Global rank: median of per-sample ranks
        self.K = int(np.median(K_samples))
        
        # Global covariance: mean of all sample covariances
        C_global = np.mean(C_samples, axis=0)  # (d, d)
        
        # Eigendecomposition of global covariance
        eigenvals, eigenvecs = np.linalg.eigh(C_global)
        eigenvecs = eigenvecs[:, ::-1]  # Sort by descending eigenvalues
        
        # Keep top K components
        self.V_global = eigenvecs[:, :self.K]
        self.fitted = True
    
    def transform(self, R):
        """
        Transform a sequence using learned PCA projection.
        
        Args:
            R: Reservoir sequence of shape (T, d)
            
        Returns:
            R_PCA: Reduced sequence of shape (T, K)
        """
        if not self.fitted:
            raise ValueError("TemporalPCA must be fitted before transform")
        
        return R @ self.V_global


class TuckerDecomposition:
    """
    Tucker Decomposition for multi-linear dimensionality reduction.
    """
    
    def __init__(self, variance_thresholds=(0.95, 0.95, 0.95)):
        """
        Initialize Tucker Decomposition.
        
        Args:
            variance_thresholds: Tuple of (θ_1, θ_2, θ_3) for each mode
        """
        self.theta = variance_thresholds
        self.U_factors = [None, None, None]  # U^(1), U^(2), U^(3)
        self.R_ranks = [None, None, None]  # R_1, R_2, R_3
        self.fitted = False
    
    def _mode_unfold(self, tensor, mode):
        """
        Unfold tensor along a given mode.
        
        Args:
            tensor: 3D tensor of shape (I, J, K)
            mode: Mode to unfold (0, 1, or 2)
            
        Returns:
            Matrix of shape (I, J*K) for mode 0, etc.
        """
        if mode == 0:
            return tensor.reshape(tensor.shape[0], -1)
        elif mode == 1:
            return tensor.transpose(1, 0, 2).reshape(tensor.shape[1], -1)
        elif mode == 2:
            return tensor.transpose(2, 0, 1).reshape(tensor.shape[2], -1)
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def _compute_rank(self, singular_values, threshold):
        """
        Compute rank to preserve threshold fraction of variance.
        
        Args:
            singular_values: Singular values
            threshold: Variance threshold
            
        Returns:
            Rank
        """
        total_var = np.sum(singular_values ** 2)
        cumsum_var = np.cumsum(singular_values ** 2) / total_var
        rank = np.argmax(cumsum_var >= threshold) + 1
        return min(rank, len(singular_values))
    
    def fit(self, R_tensor):
        """
        Fit Tucker decomposition using HOSVD (OPTIMIZED VERSION).
        
        Args:
            R_tensor: 3D tensor of shape (P, T_max, K)
        """
        P, T_max, K = R_tensor.shape
        
        # HOSVD: SVD of each mode unfolding
        # Mode 0: Sample mode (P, T_max*K)
        X_mode0 = R_tensor.reshape(P, -1)
        U0, s0, _ = svd(X_mode0, full_matrices=False)
        rank0 = self._compute_rank(s0, self.theta[0])
        self.R_ranks[0] = rank0
        self.U_factors[0] = U0[:, :rank0]
        
        # Mode 1: Time mode (T_max, P*K)
        X_mode1 = R_tensor.transpose(1, 0, 2).reshape(T_max, -1)
        U1, s1, _ = svd(X_mode1, full_matrices=False)
        rank1 = self._compute_rank(s1, self.theta[1])
        self.R_ranks[1] = rank1
        self.U_factors[1] = U1[:, :rank1]
        
        # Mode 2: Feature mode (K, P*T_max)
        X_mode2 = R_tensor.transpose(2, 0, 1).reshape(K, -1)
        U2, s2, _ = svd(X_mode2, full_matrices=False)
        rank2 = self._compute_rank(s2, self.theta[2])
        self.R_ranks[2] = rank2
        self.U_factors[2] = U2[:, :rank2]
        
        self.fitted = True
    
    def transform(self, R_i):
        """
        Transform a single sample using learned Tucker factors.
        
        Args:
            R_i: Single sample of shape (T_max, K)
            
        Returns:
            R_final: Reduced representation of shape (R_2, R_3)
        """
        if not self.fitted:
            raise ValueError("TuckerDecomposition must be fitted before transform")
        
        # Project along mode 2 (time) and mode 3 (features)
        # R_i is (T_max, K)
        # U^(2) is (T_max, R_2)
        # U^(3) is (K, R_3)
        
        R_final = (self.U_factors[1].T @ R_i @ self.U_factors[2])
        
        return R_final


