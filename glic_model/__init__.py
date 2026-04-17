"""
GLIC Model V2 - Refactored version of the GLIC compression model.

This module provides a cleaner, more modular structure for the Graph-based
Learned Image Compression (GLIC) model with auxiliary transforms.
"""

from .models.glic_main import GLICModel

__version__ = "2.0.0"
__all__ = ["GLICModel"]