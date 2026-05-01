import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set

from modellens.adapters.base import BaseAdapter, AnalysisCapability


class PyTorchAdapter(BaseAdapter):
    """
    Generic adapter for vanilla PyTorch nn.Module models.

    Auto-detects what the model contains (conv layers, recurrent layers,
    attention layers, etc.) and declares capabilities accordingly.
    For custom architectures, subclass this and override detection methods.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        self._arch_family = self._detect_architecture()

    # ---- Identity ----

    @property
    def type_of_adapter(self) -> str:
        return "pytorch"

    @property
    def architecture_family(self) -> str:
        return self._arch_family

    # ---- Architecture Detection ----

    def _detect_architecture(self) -> str:
        """
        Infer architecture family from module types present in the model.
        Priority order handles hybrid models (e.g., CNN + Linear classifier
        → 'convolutional', not 'feedforward').
        """
        has_conv = self._has_module_type((nn.Conv1d, nn.Conv2d, nn.Conv3d))
        has_recurrent = self._has_module_type((nn.LSTM, nn.GRU, nn.RNN))
        has_attention = any(
            "attn" in name.lower() or "attention" in name.lower()
            for name, _ in self.model.named_modules()
        )

        if has_attention:
            return "transformer"
        if has_recurrent:
            return "recurrent"
        if has_conv:
            return "convolutional"
        return "feedforward"

    def _has_module_type(self, types: tuple) -> bool:
        """Check if the model contains any module of the given types."""
        return any(
            isinstance(module, types) for _, module in self.model.named_modules()
        )

    # ---- Capabilities ----

    def capabilities(self) -> Set[AnalysisCapability]:
        """
        Infer capabilities from what the model actually contains.
        """
        caps = {
            AnalysisCapability.HOOKS,
            AnalysisCapability.ACTIVATION_PATCHING,
        }

        # Embedding analysis — if model has an Embedding layer
        if self._has_module_type((nn.Embedding,)):
            caps.add(AnalysisCapability.EMBEDDINGS)

        # Layer probing — if we can find an output projection
        if self._find_output_linear() is not None:
            caps.add(AnalysisCapability.LAYER_PROBING)

        # Architecture-specific capabilities
        if self._arch_family == "transformer":
            caps.add(AnalysisCapability.ATTENTION_MAPS)
            caps.add(AnalysisCapability.RESIDUAL_STREAM)

        if self._arch_family == "recurrent":
            caps.add(AnalysisCapability.GATE_ANALYSIS)

        if self._arch_family == "convolutional":
            caps.add(AnalysisCapability.FILTER_ANALYSIS)
            caps.add(AnalysisCapability.FEATURE_MAP_ANALYSIS)

        # Residual connections for non-transformers
        if self.has_residual_connections():
            caps.add(AnalysisCapability.RESIDUAL_STREAM)

        return caps

    # ---- Universal Methods ----

    def get_layer_names(self) -> List[str]:
        """Return all named layers, excluding the root module."""
        return [name for name, _ in self.model.named_modules() if name]

    def get_patchable_layers(self) -> List[str]:
        """
        Return layers suitable for activation patching based on
        detected architecture.
        """
        if self._arch_family == "transformer":
            return self._find_by_keywords(
                ["attn", "attention", "self_attn", "mlp", "feed_forward", "ffn"]
            )

        if self._arch_family == "recurrent":
            return [
                name
                for name, module in self.model.named_modules()
                if isinstance(module, (nn.LSTM, nn.GRU, nn.RNN, nn.Linear))
                and name  # Exclude root
            ]

        if self._arch_family == "convolutional":
            return [
                name
                for name, module in self.model.named_modules()
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear))
                and name
            ]

        # Feedforward: every Linear layer is patchable
        return [
            name
            for name, module in self.model.named_modules()
            if isinstance(module, nn.Linear) and name
        ]

    def forward(self, inputs, **kwargs) -> torch.Tensor:
        """Run a standard forward pass."""
        if isinstance(inputs, dict):
            return self.model(**inputs, **kwargs)
        return self.model(inputs, **kwargs)

    def tokenize(self, inputs, **kwargs) -> Dict:
        """
        No-op for vanilla PyTorch — assumes inputs are already tensors.
        Wraps in a dict for consistency with the interface.
        """
        if isinstance(inputs, torch.Tensor):
            return {"input": inputs}
        return {"input": torch.tensor(inputs)}

    # ---- Optional Methods ----

    def get_attention_layers(self) -> List[str]:
        """Best-effort detection of attention layers by name convention."""
        if self._arch_family != "transformer":
            return super().get_attention_layers()  # Raises NotImplementedError

        return self._find_by_keywords(["attn", "attention", "self_attn", "mha"])

    def get_gate_layers(self) -> List[str]:
        """Return recurrent layers whose gates can be analyzed."""
        if self._arch_family != "recurrent":
            return super().get_gate_layers()

        return [
            name
            for name, module in self.model.named_modules()
            if isinstance(module, (nn.LSTM, nn.GRU))
        ]

    def get_conv_layers(self) -> List[str]:
        """Return all convolutional layer names."""
        if self._arch_family != "convolutional":
            return super().get_conv_layers()

        return [
            name
            for name, module in self.model.named_modules()
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
        ]

    def get_output_projection(self) -> Optional[torch.Tensor]:
        """
        Try to find the output projection matrix by common naming
        conventions. Returns None if not found — user can provide
        it manually via set_output_projection().
        """
        module = self._find_output_linear()
        if module is not None:
            return module.weight.detach()
        return None

    def get_embedding_layer(self) -> Optional[torch.nn.Module]:
        """Find the first Embedding layer in the model."""
        for _, module in self.model.named_modules():
            if isinstance(module, nn.Embedding):
                return module
        return None

    def has_residual_connections(self) -> bool:
        """
        Detect residual connections by looking for common patterns.
        """
        for name, _ in self.model.named_modules():
            if any(
                kw in name.lower()
                for kw in ["residual", "shortcut", "downsample", "skip"]
            ):
                return True
        return False

    def get_sequential_layers(self) -> List[str]:
        """
        Return major computational layers in order.
        Groups by architecture type for meaningful ordering.
        """
        if self._arch_family == "convolutional":
            return [
                name
                for name, module in self.model.named_modules()
                if isinstance(
                    module,
                    (
                        nn.Conv1d,
                        nn.Conv2d,
                        nn.Conv3d,
                        nn.MaxPool2d,
                        nn.AvgPool2d,
                        nn.AdaptiveAvgPool2d,
                        nn.Linear,
                        nn.BatchNorm2d,
                    ),
                )
                and name
            ]

        if self._arch_family == "recurrent":
            return [
                name
                for name, module in self.model.named_modules()
                if isinstance(
                    module,
                    (
                        nn.Embedding,
                        nn.LSTM,
                        nn.GRU,
                        nn.RNN,
                        nn.Linear,
                    ),
                )
                and name
            ]

        # Feedforward and transformer fallback
        return [
            name
            for name, module in self.model.named_modules()
            if isinstance(module, (nn.Linear, nn.Embedding)) and name
        ]

    # ---- Private Helpers ----

    def _find_by_keywords(self, keywords: List[str]) -> List[str]:
        """Find layers whose names contain any of the given keywords."""
        return [
            name
            for name, _ in self.model.named_modules()
            if any(kw in name.lower() for kw in keywords)
        ]

    def _find_output_linear(self) -> Optional[nn.Linear]:
        """
        Find the output projection layer by name convention,
        falling back to the last Linear layer in the model.
        """
        common_names = [
            "unembed",
            "lm_head",
            "output_proj",
            "decoder",
            "fc_out",
            "classifier",
            "head",
            "output",
            "fc",
        ]

        # Try name-based detection first
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if any(cn in name.lower() for cn in common_names):
                    return module

        # Fallback: last Linear layer
        last_linear = None
        for _, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                last_linear = module

        return last_linear
