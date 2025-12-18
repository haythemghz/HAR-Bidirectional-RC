"""
Bidirectional Reservoir Computing for Human Action Recognition
"""

from .model import BidirectionalRC
from .reservoir import BidirectionalReservoir
from .fusion import ConcatenationFusion, WeightedFusion, AttentionFusion
from .dimensionality_reduction import TemporalPCA, TuckerDecomposition
from .representation_learning import EnhancedRepresentationLearning
from .readout import AdvancedReadout, SimpleReadout, Maxout, KernelActivationFunction

__version__ = '1.0.0'
__all__ = [
    'BidirectionalRC',
    'BidirectionalReservoir',
    'ConcatenationFusion',
    'WeightedFusion',
    'AttentionFusion',
    'TemporalPCA',
    'TuckerDecomposition',
    'EnhancedRepresentationLearning',
    'AdvancedReadout',
    'SimpleReadout',
    'Maxout',
    'KernelActivationFunction'
]


