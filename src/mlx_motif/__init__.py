"""mlx-motif — the canonical MLX port of Motif models."""

from mlx_motif.loader import load
from mlx_motif.model import Model, ModelArgs

__version__ = "0.1.0"
__all__ = ["Model", "ModelArgs", "load", "__version__"]
