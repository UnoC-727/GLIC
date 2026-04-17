"""
Layer modules init.
"""

from .basic_layers import LayerNorm2d, GatedFFN, GatedTransformCNN, OLP
from .wavelet_layers import WLS, iWLS
from .graph_layers import GraphAggregator, GraphDepthwiseFeedForward, GraphAttentionLayer
from .graph_feature_aggregation import GFA

# Backward compatibility aliases
from .graph_layers import IPGGrapher, GDFN, GAL
from .graph_feature_aggregation import GFA as MGB

__all__ = [
    'LayerNorm2d', 'GatedFFN', 'GatedTransformCNN', 'OLP',
    'WLS', 'iWLS',
    'GraphAggregator', 'GraphDepthwiseFeedForward', 'GraphAttentionLayer', 'GFA',
    # Backward compatibility
    'IPGGrapher', 'GDFN', 'GAL', 'MGB'
]