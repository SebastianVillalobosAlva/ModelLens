import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set
from enum import Enum


class AnalysisCapability(Enum):
    """All possible analyses ModelLens can run."""

    # Universal — every architecture supports these
    HOOKS = "hooks"
    ACTIVATION_PATCHING = "activation_patching"

    # Projection-based — requires a hidden → output mapping
    LAYER_PROBING = "layer_probing"

    # Embedding inspection — requires an embedding layer
    EMBEDDINGS = "embeddings"

    # Transformer-specific
    ATTENTION_MAPS = "attention_maps"

    # Residual connection analysis — requires skip connections
    RESIDUAL_STREAM = "residual_stream"

    # Recurrent-specific — LSTM, GRU gate activations
    GATE_ANALYSIS = "gate_analysis"

    # CNN-specific
    FILTER_ANALYSIS = "filter_analysis"
    FEATURE_MAP_ANALYSIS = "feature_map_analysis"


class UnsupportedAnalysisError(Exception):
    """Raised when an analysis is not supported by the current adapter."""

    def __init__(
        self, analysis: str, adapter_type: str, available: Set[AnalysisCapability]
    ):
        available_str = ", ".join(c.value for c in available)
        super().__init__(
            f"'{analysis}' is not supported for {adapter_type} models. "
            f"Available analyses: [{available_str}]"
        )


class BaseAdapter(ABC):
    """
    Abstract base class for all model adapters.

    Subclasses must implement the universal methods and declare
    their capabilities. Optional methods only need implementation
    if the corresponding capability is declared.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model

    # ---- Identity ----

    @property
    @abstractmethod
    def type_of_adapter(self) -> str:
        """Return adapter name (e.g., 'huggingface', 'cnn', 'lstm', 'mlp')."""
        pass

    @property
    @abstractmethod
    def architecture_family(self) -> str:
        """
        Return the broad architecture type.
        Examples: 'transformer', 'convolutional', 'recurrent', 'feedforward'
        """
        pass

    # ---- Capabilities ----

    @abstractmethod
    def capabilities(self) -> Set[AnalysisCapability]:
        """
        Declare which analyses this adapter supports.

        Example for a transformer:
            {HOOKS, EMBEDDINGS, LAYER_PROBING, ATTENTION_MAPS,
             ACTIVATION_PATCHING, RESIDUAL_STREAM}

        Example for an LSTM:
            {HOOKS, EMBEDDINGS, LAYER_PROBING, GATE_ANALYSIS,
             ACTIVATION_PATCHING}

        Example for a CNN:
            {HOOKS, LAYER_PROBING, ACTIVATION_PATCHING,
             FILTER_ANALYSIS, FEATURE_MAP_ANALYSIS}
        """
        pass

    def supports(self, capability: AnalysisCapability) -> bool:
        """Check if this adapter supports a specific analysis."""
        return capability in self.capabilities()

    def require(self, capability: AnalysisCapability, analysis_name: str) -> None:
        """
        Assert that a capability is supported. Raises UnsupportedAnalysisError
        with a clear message if not. Call this at the top of analysis functions.
        """
        if not self.supports(capability):
            raise UnsupportedAnalysisError(
                analysis_name, self.type_of_adapter, self.capabilities()
            )

    # ---- Universal Methods (all adapters must implement) ----

    @abstractmethod
    def get_layer_names(self) -> List[str]:
        """Return all named layers in the model."""
        pass

    @abstractmethod
    def get_patchable_layers(self) -> List[str]:
        """
        Return layers suitable for activation patching.

        Transformers: attn + mlp sublayers
        RNNs: individual cells or gate projections
        MLPs: each linear layer
        CNNs: conv + linear layers
        """
        pass

    @abstractmethod
    def forward(self, inputs, **kwargs) -> torch.Tensor:
        """Run a forward pass and return the output."""
        pass

    @abstractmethod
    def tokenize(self, inputs, **kwargs) -> Dict:
        """Convert raw inputs into model-ready format."""
        pass

    # ---- Optional Methods ----
    # Implement these if the corresponding capability is declared.
    # Default implementations raise NotImplementedError with a clear message.

    def get_attention_layers(self) -> List[str]:
        """Return attention layer names. Requires ATTENTION_MAPS capability."""
        raise NotImplementedError(
            f"{self.type_of_adapter} adapter does not support attention maps. "
            f"This method is only available for transformer-based models."
        )

    def get_gate_layers(self) -> List[str]:
        """
        Return gate layer names (forget, input, output gates).
        Requires GATE_ANALYSIS capability.
        """
        raise NotImplementedError(
            f"{self.type_of_adapter} adapter does not support gate analysis. "
            f"This method is only available for recurrent models (LSTM, GRU)."
        )

    def get_conv_layers(self) -> List[str]:
        """Return convolutional layer names. Requires FILTER_ANALYSIS capability."""
        raise NotImplementedError(
            f"{self.type_of_adapter} adapter does not support filter analysis. "
            f"This method is only available for convolutional models."
        )

    def get_output_projection(self) -> Optional[torch.Tensor]:
        """
        Return the matrix that maps hidden states → output space.

        Generalized version of the old get_unembedding():
        - Transformers: unembedding matrix (vocab_size, hidden_dim)
        - RNNs: output linear layer weight
        - MLPs: final linear layer weight
        - CNNs: classifier layer weight

        Requires LAYER_PROBING capability.
        """
        raise NotImplementedError(
            f"{self.type_of_adapter} adapter does not support layer probing."
        )

    def get_embedding_layer(self) -> Optional[torch.nn.Module]:
        """
        Return the input embedding layer.
        Requires EMBEDDINGS capability.
        """
        raise NotImplementedError(
            f"{self.type_of_adapter} adapter does not support embedding analysis."
        )

    def has_residual_connections(self) -> bool:
        """Whether the architecture uses skip/residual connections."""
        return False

    def get_sequential_layers(self) -> List[str]:
        """
        Return layer names in execution order.

        Critical for residual stream analysis — never rely on
        named_modules() iteration order.

        Transformers: [block_0, block_1, ..., block_n]
        RNNs: [cell_0, cell_1, ...]
        MLPs: [linear_0, linear_1, ...]
        CNNs: [conv_0, pool_0, conv_1, ..., fc]
        """
        raise NotImplementedError(
            f"{self.type_of_adapter} adapter does not provide ordered layers."
        )

    # ---- Backward Compatibility ----

    def get_unembedding(self) -> Optional[torch.Tensor]:
        """
        Deprecated: use get_output_projection() instead.
        Kept for backward compatibility with existing analysis modules.
        """
        return self.get_output_projection()
