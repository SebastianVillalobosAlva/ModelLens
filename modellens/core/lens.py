import torch
from typing import Dict, List, Optional, Type, Callable
from modellens.core.hooks import HookManager
from modellens.adapters.base import BaseAdapter
from modellens.adapters.pytorch_adapter import PyTorchAdapter
from modellens.adapters.huggingface_adapter import HuggingFaceAdapter


class ModelLens:
    """Main interface for analyzing neural network internals."""

    ADAPTERS = {
        "pytorch": PyTorchAdapter,
        "huggingface": HuggingFaceAdapter,
    }

    def __init__(self, model: torch.nn.Module, backend: Optional[str] = None):
        self.model = model
        self.model.eval()
        self.hooks = HookManager()
        self.adapter = self._resolve_adapter(model, backend)

    def _resolve_adapter(
        self, model: torch.nn.Module, backend: Optional[str]
    ) -> BaseAdapter:
        """Auto-detect or use specified backend adapter."""
        if backend:
            if backend not in self.ADAPTERS:
                raise ValueError(
                    f"Unknown backend '{backend}'. "
                    f"Available: {list(self.ADAPTERS.keys())}"
                )
            return self.ADAPTERS[backend](model)

        # Auto-detection
        if self._is_huggingface(model):
            return HuggingFaceAdapter(model)
        return PyTorchAdapter(model)

    def _is_huggingface(self, model: torch.nn.Module) -> bool:
        """Check if model is a HuggingFace PreTrainedModel."""
        return hasattr(model, "config") and hasattr(model, "generate")

    # ---- Hook Convenience Methods ----
    def attach_layers(self, layer_names: List[str]) -> "ModelLens":
        """Attach hooks to specific layers. Returns self for chaining."""
        self.hooks.attach(self.model, layer_names)
        return self

    def attach_all(self) -> "ModelLens":
        """Attach hooks to all layers. Returns self for chaining."""
        self.hooks.attach_all(self.model)
        return self

    def attach_by_type(self, layer_type: Type[torch.nn.Module]) -> "ModelLens":
        """Attach hooks to all layers of a given type. Returns self for chaining."""
        self.hooks.attach_by_type(self.model, layer_type)
        return self

    def attach_custom(self, layer_name: str, hook_fn: Callable) -> "ModelLens":
        """Attach a custom hook function to a specific layer. Returns self for chaining."""
        self.hooks.attach_custom(self.model, layer_name, hook_fn)
        return self

    # ---- Model Info ----
    def summary(self) -> Dict:
        """Return a summary of the model and current hook state."""
        return {
            "backend": self.adapter.type_of_adapter,
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "layer_names": self.adapter.get_layer_names(self.model),
            "hooks_attached": len(self.hooks),
            "activations_captured": list(self.hooks.activations.keys()),
        }

    def layer_names(self) -> List[str]:
        """List all available layer names in the model."""
        return self.adapter.get_layer_names(self.model)

    # ---- Forward Pass ----
    def run(self, inputs, **kwargs) -> torch.Tensor:
        """Run a forward pass and capture activations."""
        self.hooks.reset_activations()
        with torch.no_grad():
            output = self.adapter.forward(self.model, inputs, **kwargs)
        return output

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Return all captured activations from the last forward pass."""
        return self.hooks.activations

    def get_layer_activation(self, layer_name: str) -> Optional[torch.Tensor]:
        """Return activation for a specific layer from the last forward pass."""
        return self.hooks.get(layer_name)

    # ---- Analysis Methods (delegate to analysis modules) ----
    def logit_lens(self, inputs, **kwargs):
        """Run logit lens analysis on the model."""
        from modellens.analysis.logit_lens import run_logit_lens

        return run_logit_lens(self, inputs, **kwargs)

    def attention_map(self, inputs, **kwargs):
        """Extract attention maps from the model."""
        from modellens.analysis.attention import run_attention_analysis

        return run_attention_analysis(self, inputs, **kwargs)

    def activation_patch(self, clean_input, corrupted_input, layer_names, **kwargs):
        """Run activation patching analysis."""
        from modellens.analysis.activation_patching import run_activation_patching

        return run_activation_patching(
            self, clean_input, corrupted_input, layer_names, **kwargs
        )

    def residual_stream(self, inputs, **kwargs):
        """Analyze residual stream contributions."""
        from modellens.analysis.residual_stream import run_residual_analysis

        return run_residual_analysis(self, inputs, **kwargs)

    def embeddings(self, inputs, **kwargs):
        """Inspect embedding representations."""
        from modellens.analysis.embeddings import run_embeddings_analysis

        return run_embeddings_analysis(self, inputs, **kwargs)

    # ---- Cleanup ----
    def clear(self) -> None:
        """Remove all hooks and clear cached activations."""
        self.hooks.clear()

    def __repr__(self) -> str:
        return (
            f"ModelLens(backend={self.adapter.type_of_adapter}, "
            f"hooks={len(self.hooks)}, "
            f"params={sum(p.numel() for p in self.model.parameters()):,})"
        )

    def __del__(self):
        self.clear()
