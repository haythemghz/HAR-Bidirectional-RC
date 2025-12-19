"""
Main Bidirectional Reservoir Computing Model
Integrates all components into a complete framework
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d

from reservoir import BidirectionalReservoir
from fusion import ConcatenationFusion, WeightedFusion, AttentionFusion
from dimensionality_reduction import TemporalPCA, TuckerDecomposition
from representation_learning import EnhancedRepresentationLearning, compute_enhanced_representation
from readout import AdvancedReadout, SimpleReadout


class BidirectionalRC(nn.Module):
    """
    Complete Bidirectional Reservoir Computing framework for skeleton-based HAR.
    """
    
    def __init__(self,
                 input_dim=75,  # 3D * 25 joints
                 reservoir_size=1000,
                 num_classes=27,
                 fusion_strategy='concat',
                 spectral_radius=0.95,
                 sparsity=0.05,
                 input_scaling=0.5,
                 leak_rate_f=0.3,
                 leak_rate_b=0.3,
                 pca_variance_threshold=0.95,
                 tucker_variance_thresholds=(0.95, 0.95, 0.95),
                 attention_dim=64,
                 readout_hidden1=512,
                 readout_hidden2=256,
                 random_seed=None):
        """
        Initialize Bidirectional RC model.
        
        Args:
            input_dim: Input dimension (3N for N joints)
            reservoir_size: Reservoir size H
            num_classes: Number of action classes
            fusion_strategy: 'concat', 'weighted', or 'attention'
            spectral_radius: Spectral radius ρ
            sparsity: Connection sparsity γ
            input_scaling: Input weight scaling σ_in
            leak_rate_f: Forward leak rate α_f
            leak_rate_b: Backward leak rate α_b
            pca_variance_threshold: PCA variance threshold θ_PCA
            tucker_variance_thresholds: Tucker variance thresholds
            attention_dim: Attention hidden dimension H_a
            readout_hidden1: First readout hidden dimension
            readout_hidden2: Second readout hidden dimension
            random_seed: Random seed
        """
        super(BidirectionalRC, self).__init__()
        
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.num_classes = num_classes
        self.fusion_strategy = fusion_strategy
        self.pca_variance_threshold = pca_variance_threshold
        self.tucker_variance_thresholds = tucker_variance_thresholds
        
        # Initialize reservoir
        self.reservoir = BidirectionalReservoir(
            reservoir_size=reservoir_size,
            input_dim=input_dim,
            spectral_radius=spectral_radius,
            sparsity=sparsity,
            input_scaling=input_scaling,
            leak_rate_f=leak_rate_f,
            leak_rate_b=leak_rate_b,
            random_seed=random_seed
        )
        
        # Initialize fusion strategy
        if fusion_strategy == 'concat':
            self.fusion = ConcatenationFusion()
            fusion_output_dim = 2 * reservoir_size
        elif fusion_strategy == 'weighted':
            self.fusion = WeightedFusion(initial_beta=0.5)
            fusion_output_dim = reservoir_size
        elif fusion_strategy == 'attention':
            self.fusion = AttentionFusion(reservoir_size, attention_dim)
            fusion_output_dim = reservoir_size
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        # Dimensionality reduction (will be fitted during training)
        self.temporal_pca = None
        self.tucker_decomp = None
        
        # Representation learning
        # We'll determine input_dim after dimensionality reduction
        self.representation_learning = None
        
        # Readout (will be initialized after determining feature dimension)
        self.readout = None
        
        # Training state
        self.fitted = False
        self.T_max = None
    
    def interpolate_sequence(self, X, T_max):
        """
        Interpolate sequence to fixed length T_max.
        
        Args:
            X: Input sequence of shape (T, input_dim)
            T_max: Target sequence length
            
        Returns:
            X_norm: Normalized sequence of shape (T_max, input_dim)
        """
        T_orig = X.shape[0]
        
        if T_orig == T_max:
            return X
        
        # Create interpolation function
        t_orig = np.linspace(0, 1, T_orig)
        t_new = np.linspace(0, 1, T_max)
        
        X_norm = np.zeros((T_max, self.input_dim))
        for dim in range(self.input_dim):
            interp_func = interp1d(t_orig, X[:, dim], kind='linear', 
                                   bounds_error=False, fill_value='extrapolate')
            X_norm[:, dim] = interp_func(t_new)
        
        return X_norm
    
    def forward_reservoir(self, X):
        """
        Process sequence through bidirectional reservoir.
        
        Args:
            X: Input sequence of shape (T, input_dim)
            
        Returns:
            fused_states: Fused reservoir states
        """
        # Normalize sequence length
        if self.T_max is None:
            raise ValueError("T_max must be set before forward pass")
        
        X_norm = self.interpolate_sequence(X, self.T_max)
        
        # Process through reservoirs
        forward_states, backward_states = self.reservoir.process_sequence(X_norm)
        
        # Fuse states
        if self.fusion_strategy == 'concat':
            fused_states = self.fusion.fuse(forward_states, backward_states)
        elif self.fusion_strategy == 'weighted':
            fused_states = self.fusion.fuse(forward_states, backward_states)
            if isinstance(fused_states, torch.Tensor):
                fused_states = fused_states.detach().cpu().numpy()
        else:  # attention
            fused_states = self.fusion.fuse(forward_states, backward_states)
            if isinstance(fused_states, torch.Tensor):
                fused_states = fused_states.detach().cpu().numpy()
        
        return fused_states
    
    def fit_dimensionality_reduction(self, R_samples):
        """
        Fit dimensionality reduction on training samples.
        
        NOTE: DIMENSIONALITY REDUCTION DISABLED FOR TESTING
        Using raw reservoir states directly.
        
        Args:
            R_samples: List of fused reservoir states, each (T_max, fusion_dim)
        """
        # SKIP DIMENSIONALITY REDUCTION - Use raw reservoir states
        # This is for testing if dimensionality reduction is causing accuracy issues
        
        # Get dimensions from raw reservoir states
        T_max = R_samples[0].shape[0]
        fusion_dim = R_samples[0].shape[1]  # This is the raw feature dimension
        
        # Set dummy values for compatibility
        self.temporal_pca = None
        self.tucker_decomp = None
        
        # Initialize representation learning directly on raw reservoir features
        # Representation learning expects (R_2, R_3) where R_2 is time, R_3 is features
        # So we use the full reservoir states: (T_max, fusion_dim)
        self.representation_learning = EnhancedRepresentationLearning(
            input_dim=fusion_dim,  # Feature dimension (R_3)
            attention_dim=64
        )
        # Set to training mode so it can be trained
        self.representation_learning.train()
        
        # Compute actual feature dimension by running a test sample through representation learning
        # Use full reservoir state (not averaged) - shape (T_max, fusion_dim)
        test_R = R_samples[0].astype(np.float32)  # (T_max, fusion_dim)
        test_tensor = torch.from_numpy(test_R).float().unsqueeze(0)  # (1, T_max, fusion_dim)
        with torch.no_grad():
            test_output = self.representation_learning(test_tensor)
        feature_dim = test_output.shape[1]  # Get actual output dimension
        
        # Initialize readout with AdvancedReadout (Maxout + KAF) as described in the paper
        self.readout = AdvancedReadout(
            input_dim=feature_dim,
            hidden_dim1=512,
            hidden_dim2=256,
            num_classes=self.num_classes,
            maxout_units=5,
            kaf_kernels=20
        )
    
    def extract_features(self, X, training=False):
        """
        Extract final feature representation from input sequence.
        
        NOTE: DIMENSIONALITY REDUCTION BYPASSED - Using raw reservoir states
        
        Args:
            X: Input sequence of shape (T, input_dim)
            training: Whether in training mode (affects gradient flow)
            
        Returns:
            f_final: Final feature vector (numpy if not training, tensor if training)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before feature extraction")
        
        # Reservoir processing
        fused_states = self.forward_reservoir(X)  # (T_max, fusion_dim)
        
        # BYPASS DIMENSIONALITY REDUCTION - Use full reservoir states
        # Representation learning expects (R_2, R_3) = (T_max, fusion_dim)
        R_final = fused_states  # (T_max, fusion_dim)
        
        # Enhanced representation learning - CRITICAL FIX: Support training mode
        if self.representation_learning is not None:
            # CRITICAL: For training, use torch.tensor() instead of torch.from_numpy()
            # torch.from_numpy() creates a tensor that shares memory and can't track gradients properly
            if training:
                # Create new tensor that can track gradients
                R_tensor = torch.tensor(R_final, dtype=torch.float32, requires_grad=True)  # (T_max, fusion_dim)
                self.representation_learning.train()  # Enable gradients
                f_final_tensor = self.representation_learning(R_tensor)
                # Return tensor to allow gradient flow
                return f_final_tensor.squeeze(0) if R_tensor.shape[0] == 1 else f_final_tensor
            else:
                R_tensor = torch.from_numpy(R_final).float()  # (T_max, fusion_dim)
                self.representation_learning.eval()  # Disable gradients for inference
                with torch.no_grad():
                    f_final_tensor = self.representation_learning(R_tensor)
                f_final = f_final_tensor.squeeze(0).cpu().numpy()
                return f_final
        else:
            # Fallback to simple version
            f_final = compute_enhanced_representation(R_avg, None)
            if training:
                return torch.from_numpy(f_final).float()
            return f_final
    
    def forward(self, X, training=False):
        """
        Forward pass through complete model.
        
        Args:
            X: Input sequence of shape (T, input_dim) or batch of sequences
            training: Whether in training mode (default: False for inference)
            
        Returns:
            logits: Output logits
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forward pass")
        
        # Handle single sequence
        if isinstance(X, np.ndarray) and X.ndim == 2:
            X = [X]
        
        # Extract features for all sequences - CRITICAL: Pass training flag
        features_list = []
        for x in X:
            f = self.extract_features(x, training=training)
            features_list.append(f)
        
        # Stack features - handle both numpy and tensor cases
        if isinstance(features_list[0], torch.Tensor):
            # Features are already tensors (training mode)
            features_tensor = torch.stack(features_list)
        else:
            # Features are numpy arrays (inference mode)
            features = np.stack(features_list)
            features_tensor = torch.from_numpy(features).float()
        
        # Readout
        logits = self.readout(features_tensor)
        
        return logits
    
    def compute_fusion_regularization(self, lambda3=0.01):
        """
        Compute fusion-specific regularization term.
        
        Args:
            lambda3: Regularization weight
            
        Returns:
            reg_term: Regularization term
        """
        if self.fusion_strategy == 'concat':
            return 0.0
        elif self.fusion_strategy == 'weighted':
            mu_beta = 0.01
            beta_reg = mu_beta * (self.fusion.beta - 0.5) ** 2
            return lambda3 * beta_reg
        else:  # attention
            mu_a = 0.001
            mu_W = 0.0001
            # Attention entropy (simplified)
            W_reg = mu_W * torch.norm(self.fusion.W_a.weight, p='fro') ** 2
            return lambda3 * W_reg
    
    def compute_total_loss(self, logits, targets, lambda1=0.001, lambda2=0.0001, lambda3=0.01):
        """
        Compute total loss with all regularization terms.
        
        Args:
            logits: Model predictions
            targets: True labels
            lambda1: L2 regularization weight
            lambda2: KAF regularization weight
            lambda3: Fusion regularization weight
            
        Returns:
            total_loss: Total loss
            components: Dictionary of loss components
        """
        # Readout loss
        total_loss, ce_loss, l2_reg, kaf_reg = self.readout.compute_loss(
            logits, targets, lambda1, lambda2
        )
        
        # Fusion regularization
        fusion_reg = self.compute_fusion_regularization(lambda3)
        if isinstance(fusion_reg, torch.Tensor):
            total_loss = total_loss + fusion_reg
        
        components = {
            'ce_loss': ce_loss.item(),
            'l2_reg': l2_reg.item(),
            'kaf_reg': kaf_reg.item(),
            'fusion_reg': fusion_reg.item() if isinstance(fusion_reg, torch.Tensor) else fusion_reg,
            'total_loss': total_loss.item()
        }
        
        return total_loss, components

