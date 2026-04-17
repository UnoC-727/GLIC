"""
Model modules init.
"""

from .glic_main import GLICModel, graph_compression_AuxT
from .encoder_decoder import GLICAnalysisTransform, GLICSynthesisTransform

# Backward compatibility aliases
from .encoder_decoder import GLICEncoder, GLICDecoder

__all__ = [
    'GLICModel', 'graph_compression_AuxT',
    'GLICAnalysisTransform', 'GLICSynthesisTransform',
    # Backward compatibility
    'GLICEncoder', 'GLICDecoder'
]