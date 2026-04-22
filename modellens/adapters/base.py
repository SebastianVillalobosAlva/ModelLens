import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseAdapter(ABC):
    """Abstract base class that defines what every adapter must implement."""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    @property
    @abstractmethod
    def type_of_adapter(self) -> str:
        """Return the adapter name (e.g., 'pytorch', 'huggingface')."""
        pass

    @abstractmethod
    def get_layer_names(self, model: torch.nn.Module) -> List[str]:
        """Return all named layers in the model."""
        pass

    @abstractmethod
    def get_attention_layers(self, model: torch.nn.Module) -> List[str]:
        """Return names of attention layers specifically."""
        pass

    @abstractmethod
    def get_unembedding(self, model: torch.nn.Module) -> Optional[torch.Tensor]:
        """Return the unembedding matrix for logit lens. None if not available."""
        pass

    @abstractmethod
    def forward(self, model: torch.nn.Module, inputs, **kwargs) -> torch.Tensor:
        """Run a forward pass and return the output."""
        pass

    @abstractmethod
    def tokenize(self, inputs, **kwargs) -> Dict:
        """Convert raw inputs into model-ready format. Returns dict of tensors."""
        pass
