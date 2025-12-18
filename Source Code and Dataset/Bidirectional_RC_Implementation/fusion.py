"""
Fusion Strategies Module
Implements three fusion strategies for combining forward and backward reservoir states
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatenationFusion:
    """Simple concatenation fusion strategy."""
    
    def __init__(self):
        self.name = "concat"
    
    def fuse(self, forward_states, backward_states):
        """
        Concatenate forward and backward states.
        
        Args:
            forward_states: (T, H) forward reservoir states
            backward_states: (T, H) backward reservoir states
            
        Returns:
            fused_states: (T, 2H) concatenated states
        """
        # Convert to float32 to save memory (explicit conversion)
        forward_states = np.asarray(forward_states, dtype=np.float32)
        backward_states = np.asarray(backward_states, dtype=np.float32)
        # Concatenate and ensure result is float32
        fused = np.concatenate([forward_states, backward_states], axis=1)
        return np.asarray(fused, dtype=np.float32)
    
    def get_output_dim(self, reservoir_size):
        """Return output dimension after fusion."""
        return 2 * reservoir_size


class WeightedFusion(nn.Module):
    """Learnable weighted combination fusion strategy."""
    
    def __init__(self, initial_beta=0.5):
        """
        Initialize weighted fusion.
        
        Args:
            initial_beta (float): Initial weight for forward states
        """
        super(WeightedFusion, self).__init__()
        self.beta = nn.Parameter(torch.tensor(initial_beta, dtype=torch.float32))
        self.name = "weighted"
    
    def fuse(self, forward_states, backward_states):
        """
        Weighted combination of forward and backward states.
        
        Args:
            forward_states: (T, H) forward reservoir states
            backward_states: (T, H) backward reservoir states
            
        Returns:
            fused_states: (T, H) weighted combination
        """
        if isinstance(forward_states, np.ndarray):
            forward_states = torch.from_numpy(forward_states).float()
            backward_states = torch.from_numpy(backward_states).float()
        
        # Clamp beta to [0, 1]
        beta_clamped = torch.clamp(self.beta, 0.0, 1.0)
        
        fused = beta_clamped * forward_states + (1 - beta_clamped) * backward_states
        return fused.numpy() if isinstance(forward_states, np.ndarray) else fused
    
    def get_output_dim(self, reservoir_size):
        """Return output dimension after fusion."""
        return reservoir_size


class AttentionFusion(nn.Module):
    """Attention-based fusion strategy."""
    
    def __init__(self, reservoir_size, attention_dim=64):
        """
        Initialize attention-based fusion.
        
        Args:
            reservoir_size (int): Reservoir size H
            attention_dim (int): Attention hidden dimension H_a
        """
        super(AttentionFusion, self).__init__()
        self.H = reservoir_size
        self.H_a = attention_dim
        self.name = "attention"
        
        # Attention parameters
        self.W_a = nn.Linear(2 * reservoir_size, attention_dim)
        self.v_a = nn.Parameter(torch.randn(attention_dim))
        self.b_a = nn.Parameter(torch.zeros(attention_dim))
        
        # Initialize with Xavier
        nn.init.xavier_uniform_(self.W_a.weight)
        nn.init.normal_(self.v_a, 0, 0.01)
    
    def fuse(self, forward_states, backward_states):
        """
        Attention-based fusion of forward and backward states.
        
        Args:
            forward_states: (T, H) forward reservoir states
            backward_states: (T, H) backward reservoir states
            
        Returns:
            fused_states: (T, H) attention-weighted combination
        """
        if isinstance(forward_states, np.ndarray):
            forward_states = torch.from_numpy(forward_states).float()
            backward_states = torch.from_numpy(backward_states).float()
        
        T = forward_states.shape[0]
        
        # Concatenate forward and backward
        concat_states = torch.cat([forward_states, backward_states], dim=1)  # (T, 2H)
        
        # Compute attention scores
        attention_input = torch.tanh(self.W_a(concat_states) + self.b_a)  # (T, H_a)
        attention_scores = torch.matmul(attention_input, self.v_a)  # (T,)
        attention_weights = F.softmax(attention_scores, dim=0)  # (T,)
        
        # Apply attention to forward and backward separately
        # Reshape for broadcasting
        attention_weights = attention_weights.unsqueeze(1)  # (T, 1)
        
        # Weighted combination
        forward_weighted = attention_weights * forward_states
        backward_weighted = (1 - attention_weights) * backward_states
        
        fused = forward_weighted + backward_weighted
        
        return fused.numpy() if isinstance(forward_states, np.ndarray) else fused
    
    def get_output_dim(self, reservoir_size):
        """Return output dimension after fusion."""
        return reservoir_size


