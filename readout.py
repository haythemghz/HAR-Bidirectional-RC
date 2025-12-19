"""
Advanced Readout Mechanism Module
Implements MLP with Maxout and Kernel Activation Function (KAF)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class Maxout(nn.Module):
    """
    Maxout activation function.
    """
    
    def __init__(self, num_units=5):
        """
        Initialize Maxout.
        
        Args:
            num_units (int): Number of piecewise linear units (k)
        """
        super(Maxout, self).__init__()
        self.num_units = num_units
    
    def forward(self, x):
        """
        Apply Maxout activation.
        
        Args:
            x: Input tensor of shape (..., input_dim, num_units)
            
        Returns:
            Output tensor of shape (..., input_dim)
        """
        return torch.max(x, dim=-1)[0]


class KernelActivationFunction(nn.Module):
    """
    Kernel Activation Function (KAF) with learnable kernels.
    """
    
    def __init__(self, input_dim, num_kernels=20, gamma_init=1.0):
        """
        Initialize KAF.
        
        Args:
            input_dim (int): Input dimension
            num_kernels (int): Number of kernel centers (D)
            gamma_init (float): Initial bandwidth parameter
        """
        super(KernelActivationFunction, self).__init__()
        self.input_dim = input_dim
        self.num_kernels = num_kernels
        
        # Learnable kernel coefficients - initialize properly
        self.alpha = nn.Parameter(torch.randn(input_dim, num_kernels) * 0.01)  # Smaller init
        
        # Kernel centers - initialize around zero with small variance
        self.register_buffer('centers', torch.randn(input_dim, num_kernels) * 0.5)
        
        # Learnable bandwidth - initialize to reasonable value
        self.gamma = nn.Parameter(torch.tensor(gamma_init))
    
    def forward(self, x):
        """
        Apply KAF activation.
        
        Args:
            x: Input tensor of shape (..., input_dim)
            
        Returns:
            Output tensor of shape (..., input_dim)
        """
        # Expand dimensions for broadcasting
        # x: (..., input_dim)
        # centers: (input_dim, num_kernels)
        # We want: (..., input_dim, num_kernels)
        
        # FIX: KAF computation - ensure proper broadcasting
        # x: (batch, input_dim) or (input_dim,)
        # centers: (input_dim, num_kernels)
        # alpha: (input_dim, num_kernels)
        
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if needed
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = x.shape[0]
        input_dim = x.shape[1]
        
        # FIXED: KAF per-dimension computation
        # Each dimension d has its own kernel centers and coefficients
        # For dimension d: compute kernel activation using x[:, d] and centers[d, :]
        
        output_list = []
        for d in range(input_dim):
            # Get values for dimension d
            x_d = x[:, d]  # (batch,)
            centers_d = self.centers[d, :]  # (num_kernels,)
            alpha_d = self.alpha[d, :]  # (num_kernels,)
            
            # Compute distances: (batch, num_kernels)
            x_d_expanded = x_d.unsqueeze(1)  # (batch, 1)
            centers_d_expanded = centers_d.unsqueeze(0)  # (1, num_kernels)
            distances_d = (x_d_expanded - centers_d_expanded) ** 2  # (batch, num_kernels)
            
            # Gaussian kernel: (batch, num_kernels)
            kernel_values_d = torch.exp(-self.gamma * distances_d)  # (batch, num_kernels)
            
            # Weighted sum: (batch,)
            output_d = torch.sum(alpha_d.unsqueeze(0) * kernel_values_d, dim=1)  # (batch,)
            output_list.append(output_d)
        
        # Stack: (batch, input_dim)
        output = torch.stack(output_list, dim=1)
        
        if squeeze_output:
            output = output.squeeze(0)
        
        return output


class SpectralNormalization(nn.Module):
    """
    Spectral normalization wrapper for weight matrices.
    """
    
    def __init__(self, module, n_power_iterations=1):
        """
        Initialize spectral normalization.
        
        Args:
            module: Linear layer to normalize
            n_power_iterations: Number of power iterations
        """
        super(SpectralNormalization, self).__init__()
        self.module = module
        self.n_power_iterations = n_power_iterations
        
        # Initialize u and v for power iteration
        weight = module.weight.data
        u = torch.randn(weight.shape[0])
        v = torch.randn(weight.shape[1])
        
        self.register_buffer('u', F.normalize(u, dim=0))
        self.register_buffer('v', F.normalize(v, dim=0))
    
    def _compute_spectral_norm(self):
        """Compute spectral norm using power iteration."""
        weight = self.module.weight
        
        for _ in range(self.n_power_iterations):
            # Update v
            v = F.normalize(torch.mv(weight.t(), self.u), dim=0)
            self.v.data = v
            
            # Update u
            u = F.normalize(torch.mv(weight, self.v), dim=0)
            self.u.data = u
        
        # Compute spectral norm
        sigma = torch.dot(self.u, torch.mv(weight, self.v))
        return sigma
    
    def forward(self, x):
        """Forward pass with spectral normalization."""
        sigma = self._compute_spectral_norm()
        weight_normalized = self.module.weight / sigma
        return F.linear(x, weight_normalized, self.module.bias)


class SimpleReadout(nn.Module):
    """
    Simple readout mechanism with standard ReLU activations.
    Replaces KAF with simpler, more stable activations.
    """
    
    def __init__(self,
                 input_dim,
                 hidden_dim1=512,
                 hidden_dim2=256,
                 num_classes=27,
                 dropout1=0.3,
                 dropout2=0.2):
        """
        Initialize simple readout.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim1 (int): First hidden layer dimension
            hidden_dim2 (int): Second hidden layer dimension
            num_classes (int): Number of output classes
            dropout1 (float): Dropout probability for first layer
            dropout2 (float): Dropout probability for second layer
        """
        super(SimpleReadout, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.num_classes = num_classes
        
        # Layer 1: Linear + ReLU
        self.W1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(dropout1)
        
        # Layer 2: Linear + ReLU
        self.W2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout2)
        
        # Layer 3: Output
        self.W3 = nn.Linear(hidden_dim2, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        # Use He initialization for ReLU activations
        nn.init.kaiming_uniform_(self.W1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.W2.weight, nonlinearity='relu')
        # Xavier initialization for output layer
        nn.init.xavier_uniform_(self.W3.weight, gain=1.0)
        
        # Initialize biases to zero
        nn.init.constant_(self.W1.bias, 0.0)
        nn.init.constant_(self.W2.bias, 0.0)
        nn.init.constant_(self.W3.bias, 0.0)
    
    def forward(self, x):
        """
        Forward pass through readout network.
        
        Args:
            x: Input features of shape (batch, input_dim)
            
        Returns:
            logits: Output logits of shape (batch, num_classes)
        """
        # Normalize input features
        if x.shape[0] > 1:
            x_mean = x.mean(dim=0, keepdim=True)
            x_std = x.std(dim=0, keepdim=True) + 1e-8
            x_norm = (x - x_mean) / x_std
        else:
            x_mean = x.mean(dim=0, keepdim=True)
            x_std = x.std(dim=0, keepdim=True) + 1e-8
            x_norm = (x - x_mean) / x_std
        
        # Layer 1: Linear + ReLU + BatchNorm + Dropout
        z1 = self.W1(x_norm)
        if z1.shape[0] > 1 and z1.dim() == 2:
            z1 = self.bn1(z1)
        z1 = F.relu(z1)
        z1 = self.dropout1(z1)

        # Layer 2: Linear + ReLU + BatchNorm + Dropout
        z2 = self.W2(z1)
        if z2.shape[0] > 1 and z2.dim() == 2:
            z2 = self.bn2(z2)
        z2 = F.relu(z2)
        z2 = self.dropout2(z2)

        # Layer 3: Output (no activation)
        y = self.W3(z2)
        
        return y
    
    def compute_loss(self, logits, targets, lambda1=0.001, lambda2=0.0001):
        """
        Compute total loss with regularization.
        
        Args:
            logits: Model predictions
            targets: True labels
            lambda1: L2 regularization weight
            lambda2: KAF regularization weight (ignored)
            
        Returns:
            Total loss
        """
        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets)
        
        # L2 regularization on weights
        l2_reg = lambda1 * (
            torch.norm(self.W1.weight, p='fro') ** 2 +
            torch.norm(self.W2.weight, p='fro') ** 2 +
            torch.norm(self.W3.weight, p='fro') ** 2
        )
        
        # No KAF regularization for SimpleReadout
        kaf_reg = torch.tensor(0.0).to(logits.device)
        
        total_loss = ce_loss + l2_reg
        
        return total_loss, ce_loss, l2_reg, kaf_reg


class AdvancedReadout(nn.Module):
    """
    Advanced readout mechanism with Maxout and KAF activations.
    """
    
    def __init__(self,
                 input_dim,
                 hidden_dim1=512,
                 hidden_dim2=256,
                 num_classes=27,
                 maxout_units=5,
                 kaf_kernels=20,
                 dropout1=0.3,
                 dropout2=0.2):
        """
        Initialize advanced readout.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim1 (int): First hidden layer dimension
            hidden_dim2 (int): Second hidden layer dimension
            num_classes (int): Number of output classes
            maxout_units (int): Number of Maxout units (k)
            kaf_kernels (int): Number of KAF kernels (D)
            dropout1 (float): Dropout probability for first layer
            dropout2 (float): Dropout probability for second layer
        """
        super(AdvancedReadout, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.num_classes = num_classes
        
        # Layer 1: Maxout
        # For Maxout, we need num_units times the output dimension
        self.W1 = nn.Linear(input_dim, hidden_dim1 * maxout_units)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim1 * maxout_units))
        self.maxout = Maxout(num_units=maxout_units)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(dropout1)
        
        # Layer 2: KAF
        self.W2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.b2 = nn.Parameter(torch.zeros(hidden_dim2))
        self.kaf = KernelActivationFunction(hidden_dim2, num_kernels=kaf_kernels)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout2)
        
        # Layer 3: Output
        self.W3 = nn.Linear(hidden_dim2, num_classes)
        self.b3 = nn.Parameter(torch.zeros(num_classes))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using improved initialization."""
        # Use He initialization for better gradient flow
        nn.init.kaiming_uniform_(self.W1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.W2.weight, nonlinearity='relu')
        # FIX: Use proper initialization for output layer (not too small!)
        nn.init.xavier_uniform_(self.W3.weight, gain=1.0)  # Changed from 0.1 to 1.0
        
        # Initialize biases properly
        nn.init.constant_(self.b3, 0.0)  # Output bias zero
        nn.init.constant_(self.b1, 0.01)  # Small positive for hidden layers
        nn.init.constant_(self.b2, 0.01)
    
    def forward(self, x):
        """
        Forward pass through readout network.
        
        Args:
            x: Input features of shape (batch, input_dim)
            
        Returns:
            logits: Output logits of shape (batch, num_classes)
        """
        # FIX: Normalize input features to prevent gradient explosion
        # Use batch normalization if batch size > 1, otherwise normalize by feature statistics
        if x.shape[0] > 1:
            # Batch normalization across features
            x_mean = x.mean(dim=0, keepdim=True)
            x_std = x.std(dim=0, keepdim=True) + 1e-8
            x_norm = (x - x_mean) / x_std
        else:
            # For single sample, normalize by feature statistics (mean=0, std=1 per feature)
            x_mean = x.mean(dim=0, keepdim=True)
            x_std = x.std(dim=0, keepdim=True) + 1e-8
            x_norm = (x - x_mean) / x_std
        
        # Layer 1: Maxout
        z1 = self.W1(x_norm) + self.b1
        z1 = z1.view(z1.shape[0], self.hidden_dim1, -1)  # Reshape for Maxout
        z1 = self.maxout(z1)
        # FIX: Add ReLU to ensure non-negative and prevent dead neurons
        z1 = F.relu(z1)
        # BatchNorm only if batch size > 1 and has correct dimensions
        if z1.shape[0] > 1 and z1.dim() == 2:
            z1 = self.bn1(z1)
        z1 = self.dropout1(z1)

        # Layer 2: KAF
        z2 = self.W2(z1) + self.b2
        z2 = self.kaf(z2)
        # FIX: Add ReLU after KAF
        z2 = F.relu(z2)
        # BatchNorm only if batch size > 1 and has correct dimensions
        if z2.shape[0] > 1 and z2.dim() == 2:
            z2 = self.bn2(z2)
        z2 = self.dropout2(z2)

        # Layer 3: Output
        y = self.W3(z2) + self.b3
        
        return y
    
    def compute_loss(self, logits, targets, lambda1=0.001, lambda2=0.0001):
        """
        Compute total loss with regularization.
        
        Args:
            logits: Model predictions
            targets: True labels
            lambda1: L2 regularization weight
            lambda2: KAF regularization weight
            
        Returns:
            Total loss
        """
        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets)
        
        # L2 regularization on weights
        l2_reg = lambda1 * (
            torch.norm(self.W1.weight, p='fro') ** 2 +
            torch.norm(self.W2.weight, p='fro') ** 2 +
            torch.norm(self.W3.weight, p='fro') ** 2
        )
        
        # KAF regularization
        kaf_reg = lambda2 * torch.norm(self.kaf.alpha, p=2) ** 2
        
        total_loss = ce_loss + l2_reg + kaf_reg
        
        return total_loss, ce_loss, l2_reg, kaf_reg


