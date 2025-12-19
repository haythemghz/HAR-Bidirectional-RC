"""
Enhanced Representation Learning Module
Implements multi-scale pooling and temporal attention
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedRepresentationLearning(nn.Module):
    """
    Enhanced representation learning with multi-scale pooling and temporal attention.
    """
    
    def __init__(self, input_dim, attention_dim=64, kernel_sizes=[3, 5, 7]):
        """
        Initialize enhanced representation learning.
        
        Args:
            input_dim (int): Input feature dimension (R_3)
            attention_dim (int): Attention hidden dimension (H_a)
            kernel_sizes (list): Kernel sizes for multi-scale convolution
        """
        super(EnhancedRepresentationLearning, self).__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.kernel_sizes = kernel_sizes
        
        # Temporal attention parameters
        self.W_att = nn.Linear(input_dim, attention_dim)
        self.v_att = nn.Parameter(torch.randn(attention_dim))
        self.b_att = nn.Parameter(torch.zeros(attention_dim))
        
        # Multi-scale 1D convolutions
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, input_dim, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        # Initialize
        nn.init.xavier_uniform_(self.W_att.weight)
        nn.init.normal_(self.v_att, 0, 0.01)
    
    def forward(self, R_final):
        """
        Extract enhanced representation from reduced temporal sequence.
        
        Args:
            R_final: Reduced sequence of shape (R_2, R_3) or (batch, R_2, R_3)
            
        Returns:
            f_final: Enhanced representation vector
        """
        if isinstance(R_final, np.ndarray):
            R_final = torch.from_numpy(R_final).float()
        
        # Handle batch dimension
        if R_final.dim() == 2:
            R_final = R_final.unsqueeze(0)  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, R_2, R_3 = R_final.shape
        
        # 1. Global Statistical Pooling
        # FIX: Add small epsilon to prevent division by zero in std
        f_global = torch.mean(R_final, dim=1)  # (batch, R_3)
        f_max = torch.max(R_final, dim=1)[0]  # (batch, R_3)
        f_std = torch.std(R_final, dim=1, unbiased=True) + 1e-8  # (batch, R_3) - add epsilon
        
        # 2. Local Multi-Scale Pattern Extraction
        # Transpose for Conv1d: (batch, R_3, R_2)
        R_transposed = R_final.transpose(1, 2)
        
        f_local_list = []
        for conv in self.convs:
            # Convolution
            conv_out = F.relu(conv(R_transposed))  # (batch, R_3, R_2)
            # Global max pooling
            pooled = torch.max(conv_out, dim=2)[0]  # (batch, R_3)
            f_local_list.append(pooled)
        
        f_local = torch.cat(f_local_list, dim=1)  # (batch, R_3 * len(kernel_sizes))
        
        # 3. Temporal Attention Mechanism
        # Compute attention scores
        attention_input = torch.tanh(
            self.W_att(R_final) + self.b_att.unsqueeze(0).unsqueeze(0)
        )  # (batch, R_2, attention_dim)
        
        attention_scores = torch.matmul(
            attention_input, self.v_att.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)  # (batch, R_2)
        
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, R_2)
        
        # Weighted sum
        f_attention = torch.sum(
            attention_weights.unsqueeze(-1) * R_final, dim=1
        )  # (batch, R_3)
        
        # 4. Concatenate all features
        f_final = torch.cat([
            f_global, f_max, f_std, f_local, f_attention
        ], dim=1)  # (batch, feature_dim)
        
        # FIX: Normalize features to prevent gradient explosion
        # Use layer normalization to stabilize feature magnitudes
        if f_final.shape[0] > 1:
            # Batch normalization
            f_final = F.layer_norm(f_final, f_final.shape[1:])
        else:
            # For single sample, normalize by feature statistics
            f_mean = f_final.mean(dim=1, keepdim=True)
            f_std = f_final.std(dim=1, keepdim=True) + 1e-8
            f_final = (f_final - f_mean) / f_std
        
        if squeeze_output:
            f_final = f_final.squeeze(0)
        
        return f_final


def compute_enhanced_representation(R_final, model=None):
    """
    Compute enhanced representation (numpy version for compatibility).
    
    Args:
        R_final: Reduced sequence of shape (R_2, R_3)
        model: EnhancedRepresentationLearning model (optional)
        
    Returns:
        f_final: Enhanced representation vector
    """
    if model is None:
        # Simple version without learnable parameters
        R_2, R_3 = R_final.shape
        
        # Global statistics
        f_global = np.mean(R_final, axis=0)
        f_max = np.max(R_final, axis=0)
        f_std = np.std(R_final, axis=0, ddof=1)
        
        # Simple multi-scale: just use different pooling
        # This is a simplified version
        f_local = np.concatenate([
            np.max(R_final[:R_2//3], axis=0),
            np.max(R_final[R_2//3:2*R_2//3], axis=0),
            np.max(R_final[2*R_2//3:], axis=0)
        ])
        
        # Simple attention: uniform weights
        f_attention = np.mean(R_final, axis=0)
        
        f_final = np.concatenate([f_global, f_max, f_std, f_local, f_attention])
    else:
        model.eval()
        with torch.no_grad():
            f_final = model(R_final).numpy()
    
    return f_final


