import torch
from typing import Dict, List, Optional
from modellens.adapters.base import BaseAdapter


class PyTorchAdapter(BaseAdapter):
    """Adapter for vanilla PyTorch nn.Module models."""

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)

    @property
    def type_of_adapter(self) -> str:
        return "pytorch"

    def get_layer_names(self, model: torch.nn.Module) -> List[str]:
        """Return all named layers, excluding the root module."""
        return [name for name, _ in model.named_modules() if name]

    def get_attention_layers(self, model: torch.nn.Module) -> List[str]:
        """
        Best-effort detection of attention layers by name convention.
        Custom models should override this or pass layer names explicitly.
        """
        keywords = [
            "attn",
            "attention",
            "self_attn",
            "mha",
        ]  # Add possible names of attention layers
        return [
            name
            for name, _ in model.named_modules()
            if any(kw in name.lower() for kw in keywords)
        ]

    def get_unembedding(self, model: torch.nn.Module) -> Optional[torch.Tensor]:
        """
        Try to find an unembedding matrix by common naming conventions.
        Returns None if not found — user can provide it manually.
        """
        common_names = [
            "unembed",
            "lm_head",
            "output_proj",
            "decoder",
            "fc_out",
        ]  # Add possible names of embedding layers
        for name, module in model.named_modules():
            if any(cn in name.lower() for cn in common_names):
                if hasattr(module, "weight"):
                    return module.weight.detach()
        return None

    def forward(self, model: torch.nn.Module, inputs, **kwargs) -> torch.Tensor:
        """Run a standard forward pass."""
        return model(inputs, **kwargs)

    def tokenize(self, inputs, **kwargs) -> Dict:
        """
        No-op for vanilla PyTorch — assumes inputs are already tensors.
        Returns them wrapped in a dict for consistency with the interface.
        """
        if isinstance(inputs, torch.Tensor):
            return {"input": inputs}
        return {"input": torch.tensor(inputs)}
