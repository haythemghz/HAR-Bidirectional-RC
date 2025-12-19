"""
Bidirectional Reservoir Computing Core Module
Implements the bidirectional Echo State Network (ESN) architecture
"""

import numpy as np
from scipy.sparse import random as sparse_random
from scipy.sparse.linalg import eigs
from scipy.linalg import norm


class BidirectionalReservoir:
    """
    Bidirectional Reservoir Computing architecture for temporal sequence processing.
    
    Implements two parallel reservoirs (forward and backward) that process
    input sequences in opposite temporal directions.
    """
    
    def __init__(self, 
                 reservoir_size=1000,
                 input_dim=75,
                 spectral_radius=0.95,
                 sparsity=0.05,
                 input_scaling=0.5,
                 leak_rate_f=0.3,
                 leak_rate_b=0.3,
                 random_seed=None):
        """
        Initialize bidirectional reservoir.
        
        Args:
            reservoir_size (int): Number of reservoir units (H)
            input_dim (int): Input dimension (3N for N joints)
            spectral_radius (float): Spectral radius for reservoir weights (ρ)
            sparsity (float): Connection sparsity (γ)
            input_scaling (float): Input weight scaling (σ_in)
            leak_rate_f (float): Forward reservoir leak rate (α_f)
            leak_rate_b (float): Backward reservoir leak rate (α_b)
            random_seed (int): Random seed for reproducibility
        """
        self.H = reservoir_size
        self.input_dim = input_dim
        self.rho = spectral_radius
        self.gamma = sparsity
        self.sigma_in = input_scaling
        self.alpha_f = leak_rate_f
        self.alpha_b = leak_rate_b
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize forward reservoir
        self.W_in_f, self.W_res_f, self.b_f = self._initialize_reservoir()
        
        # Initialize backward reservoir
        self.W_in_b, self.W_res_b, self.b_b = self._initialize_reservoir()
    
    def _initialize_reservoir(self):
        """Initialize a single reservoir with proper spectral properties."""
        # Input weights: uniform distribution
        W_in = np.random.uniform(
            low=-self.sigma_in,
            high=self.sigma_in,
            size=(self.H, self.input_dim)
        )
        
        # Reservoir weights: sparse random matrix
        W_res = sparse_random(
            self.H, self.H,
            density=self.gamma,
            format='csr',
            random_state=np.random.randint(0, 2**31)
        ).toarray()
        
        # Normalize to desired spectral radius
        # Compute spectral radius
        eigenvals = eigs(W_res, k=1, return_eigenvectors=False)
        current_rho = np.abs(eigenvals[0])
        
        if current_rho > 0:
            W_res = (self.rho / current_rho) * W_res
        
        # Bias
        b = np.random.uniform(-0.1, 0.1, size=self.H)
        
        return W_in, W_res, b
    
    def forward_pass(self, x_t, r_prev):
        """
        Forward reservoir update.
        
        Args:
            x_t: Input at time t
            r_prev: Previous reservoir state
            
        Returns:
            Updated reservoir state
        """
        pre_activation = (self.W_in_f @ x_t + 
                         self.W_res_f @ r_prev + 
                         self.b_f)
        r_new = (1 - self.alpha_f) * r_prev + self.alpha_f * np.tanh(pre_activation)
        return r_new
    
    def backward_pass(self, x_t, r_next):
        """
        Backward reservoir update.
        
        Args:
            x_t: Input at time t
            r_next: Next reservoir state (for backward direction)
            
        Returns:
            Updated reservoir state
        """
        pre_activation = (self.W_in_b @ x_t + 
                         self.W_res_b @ r_next + 
                         self.b_b)
        r_new = (1 - self.alpha_b) * r_next + self.alpha_b * np.tanh(pre_activation)
        return r_new
    
    def process_sequence(self, X):
        """
        Process a sequence through both forward and backward reservoirs.
        
        Args:
            X: Input sequence of shape (T, input_dim)
            
        Returns:
            forward_states: Forward reservoir states (T, H)
            backward_states: Backward reservoir states (T, H)
        """
        T = X.shape[0]
        
        # Forward pass
        forward_states = np.zeros((T, self.H), dtype=np.float32)
        r_forward = np.zeros(self.H, dtype=np.float32)
        
        for t in range(T):
            r_forward = self.forward_pass(X[t], r_forward)
            forward_states[t] = r_forward
        
        # Backward pass
        backward_states = np.zeros((T, self.H), dtype=np.float32)
        r_backward = np.zeros(self.H, dtype=np.float32)
        
        for t in range(T - 1, -1, -1):
            r_backward = self.backward_pass(X[t], r_backward)
            backward_states[t] = r_backward
        
        return forward_states, backward_states


