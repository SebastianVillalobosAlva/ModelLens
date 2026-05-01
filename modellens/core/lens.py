import torch
from typing import Dict, List, Optional, Type, Callable
from modellens.core.hooks import HookManager
from modellens.adapters.base import BaseAdapter, AnalysisCapability


class ModelLens:
    """
    Main interface for analyzing neural network internals.

    Point it at any PyTorch model — transformer, CNN, LSTM, MLP —
    and it auto-detects which analyses are available.

    Usage:
        lens = ModelLens(model)
        print(lens.summary())  # See what's available
        results = lens.layer_probe(inputs)  # Run an analysis
    """

    def __init__(self, model: torch.nn.Module, backend: Optional[str] = None):
        self.model = model
        self.model.eval()
        self.hooks = HookManager()
        self.adapter = self._resolve_adapter(model, backend)

    # ---- Adapter Resolution ----

    def _resolve_adapter(
        self, model: torch.nn.Module, backend: Optional[str]
    ) -> BaseAdapter:
        """Auto-detect or use specified backend adapter."""
        if backend:
            return self._get_adapter_by_name(backend, model)

        # Auto-detection chain
        if self._is_huggingface(model):
            from modellens.adapters.huggingface_adapter import HuggingFaceAdapter

            return HuggingFaceAdapter(model)

        # Fallback: PyTorchAdapter handles all vanilla nn.Module models
        # and auto-detects architecture family (CNN, RNN, MLP, etc.)
        from modellens.adapters.pytorch_adapter import PyTorchAdapter

        return PyTorchAdapter(model)

    def _get_adapter_by_name(self, backend: str, model: torch.nn.Module) -> BaseAdapter:
        """Resolve adapter from string name."""
        from modellens.adapters.pytorch_adapter import PyTorchAdapter
        from modellens.adapters.huggingface_adapter import HuggingFaceAdapter

        adapters = {
            "pytorch": PyTorchAdapter,
            "huggingface": HuggingFaceAdapter,
        }

        if backend not in adapters:
            raise ValueError(
                f"Unknown backend '{backend}'. " f"Available: {list(adapters.keys())}"
            )
        return adapters[backend](model)

    def _is_huggingface(self, model: torch.nn.Module) -> bool:
        """Check if model is a HuggingFace PreTrainedModel."""
        try:
            from transformers import PreTrainedModel

            if isinstance(model, PreTrainedModel):
                return True
        except ImportError:
            pass

        if hasattr(model, "config"):
            config = model.config
            return hasattr(config, "model_type") and hasattr(config, "hidden_size")

        return False

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
        """Attach a custom hook function. Returns self for chaining."""
        self.hooks.attach_custom(self.model, layer_name, hook_fn)
        return self

    # ---- Model Info ----

    def summary(self) -> Dict:
        """Return a summary of the model, adapter, and available analyses."""
        caps = self.adapter.capabilities()
        return {
            "backend": self.adapter.type_of_adapter,
            "architecture": self.adapter.architecture_family,
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "layer_count": len(self.adapter.get_layer_names()),
            "hooks_attached": len(self.hooks),
            "activations_captured": list(self.hooks.activations.keys()),
            "available_analyses": sorted([c.value for c in caps]),
            "unavailable_analyses": sorted(
                [c.value for c in AnalysisCapability if c not in caps]
            ),
        }

    def layer_names(self) -> List[str]:
        """List all available layer names in the model."""
        return self.adapter.get_layer_names()

    def patchable_layers(self) -> List[str]:
        """List layers suitable for activation patching."""
        return self.adapter.get_patchable_layers()

    def available_analyses(self) -> List[str]:
        """List analyses available for the loaded model."""
        return sorted([c.value for c in self.adapter.capabilities()])

    # ---- Forward Pass ----

    def run(self, inputs, **kwargs) -> torch.Tensor:
        """Run a forward pass and capture activations at hooked layers."""
        self.hooks.reset_activations()
        with torch.no_grad():
            output = self.adapter.forward(inputs, **kwargs)
        return output

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Return all captured activations from the last forward pass."""
        return self.hooks.activations

    def get_layer_activation(self, layer_name: str) -> Optional[torch.Tensor]:
        """Return activation for a specific layer from the last forward pass."""
        return self.hooks.get(layer_name)

    # ---- Analysis Methods ----
    # Each method checks capabilities before delegating to analysis modules.

    # -- Universal analyses --

    def activation_patch(
        self, clean_input, corrupted_input, layer_names=None, **kwargs
    ):
        """
        Run activation patching to measure causal impact of each layer.
        Available for all architectures.
        """
        self.adapter.require(AnalysisCapability.ACTIVATION_PATCHING, "activation_patch")
        from modellens.analysis.activation_patching import run_activation_patching

        if layer_names is None:
            layer_names = self.adapter.get_patchable_layers()
        return run_activation_patching(
            self, clean_input, corrupted_input, layer_names, **kwargs
        )

    # -- Projection-based analyses --

    def layer_probe(self, inputs, **kwargs):
        """
        Project intermediate layer representations through the output
        projection to see what the model would predict at each layer.

        Generalized version of logit lens — works on any architecture
        with a hidden → output mapping (transformers, RNNs, MLPs, CNNs).
        """
        self.adapter.require(AnalysisCapability.LAYER_PROBING, "layer_probe")
        from modellens.analysis.logit_lens import run_logit_lens

        return run_logit_lens(self, inputs, **kwargs)

    def logit_lens(self, inputs, **kwargs):
        """Alias for layer_probe(). Kept for backward compatibility."""
        return self.layer_probe(inputs, **kwargs)

    # -- Embedding analyses --

    def embeddings(self, inputs, **kwargs):
        """Analyze input embedding representations."""
        self.adapter.require(AnalysisCapability.EMBEDDINGS, "embeddings")
        from modellens.analysis.embeddings import run_embeddings_analysis

        return run_embeddings_analysis(self, inputs, **kwargs)

    # -- Transformer-specific --

    def attention_map(self, inputs, **kwargs):
        """Extract attention weight maps from transformer layers."""
        self.adapter.require(AnalysisCapability.ATTENTION_MAPS, "attention_map")
        from modellens.analysis.attention import run_attention_analysis

        return run_attention_analysis(self, inputs, **kwargs)

    # -- Residual stream --

    def residual_stream(self, inputs, **kwargs):
        """
        Analyze how information flows through the residual stream.
        Available for architectures with skip connections
        (transformers, ResNets).
        """
        self.adapter.require(AnalysisCapability.RESIDUAL_STREAM, "residual_stream")
        from modellens.analysis.residual_stream import run_residual_analysis

        return run_residual_analysis(self, inputs, **kwargs)

    # -- Recurrent-specific --

    def gate_analysis(self, inputs, **kwargs):
        """
        Analyze gate activations in recurrent models (LSTM, GRU).
        Shows forget gate, input gate, and output gate behavior.
        """
        self.adapter.require(AnalysisCapability.GATE_ANALYSIS, "gate_analysis")
        from modellens.analysis.gates import run_gate_analysis

        return run_gate_analysis(self, inputs, **kwargs)

    # -- CNN-specific --

    def filter_analysis(self, inputs, **kwargs):
        """
        Analyze CNN filters: feature maps, activation statistics,
        dead filter detection.
        """
        self.adapter.require(AnalysisCapability.FILTER_ANALYSIS, "filter_analysis")
        from modellens.analysis.filters import run_filter_analysis

        return run_filter_analysis(self, inputs, **kwargs)

    def feature_maps(self, inputs, **kwargs):
        """
        Track how feature maps evolve through CNN layers.
        Shows spatial resolution reduction and channel expansion.
        """
        self.adapter.require(AnalysisCapability.FEATURE_MAP_ANALYSIS, "feature_maps")
        from modellens.analysis.filters import run_feature_map_analysis

        return run_feature_map_analysis(self, inputs, **kwargs)

    # ---- Cleanup ----

    def clear(self) -> None:
        """Remove all hooks and clear cached activations."""
        self.hooks.clear()

    def __repr__(self) -> str:
        caps = self.adapter.capabilities()
        return (
            f"ModelLens(\n"
            f"  backend={self.adapter.type_of_adapter},\n"
            f"  architecture={self.adapter.architecture_family},\n"
            f"  params={sum(p.numel() for p in self.model.parameters()):,},\n"
            f"  analyses={sorted([c.value for c in caps])}\n"
            f")"
        )

    def __del__(self):
        self.clear()
