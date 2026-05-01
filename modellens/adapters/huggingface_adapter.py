from typing import Dict, List, Optional, Set
import torch
from modellens.adapters.base import BaseAdapter, AnalysisCapability

# Maps HuggingFace config.model_type to layer naming patterns
# Each entry: (block_pattern, attn_suffix, mlp_suffix)
_ARCHITECTURE_PATTERNS = {
    # GPT-2 family
    "gpt2": ("transformer.h.{i}", "attn", "mlp"),
    "gpt_neo": ("transformer.h.{i}", "attn.attention", "mlp"),
    "gpt_neox": ("gpt_neox.layers.{i}", "attention", "mlp"),
    "gptj": ("transformer.h.{i}", "attn", "mlp"),
    # LLaMA family
    "llama": ("model.layers.{i}", "self_attn", "mlp"),
    "mistral": ("model.layers.{i}", "self_attn", "mlp"),
    "gemma": ("model.layers.{i}", "self_attn", "mlp"),
    "gemma2": ("model.layers.{i}", "self_attn", "mlp"),
    "phi": ("model.layers.{i}", "self_attn", "mlp"),
    "phi3": ("model.layers.{i}", "self_attn", "mlp"),
    "qwen2": ("model.layers.{i}", "self_attn", "mlp"),
    "stablelm": ("model.layers.{i}", "self_attn", "mlp"),
    "cohere": ("model.layers.{i}", "self_attn", "mlp"),
    # BERT family
    "bert": ("bert.encoder.layer.{i}", "attention", "intermediate"),
    "roberta": ("roberta.encoder.layer.{i}", "attention", "intermediate"),
    "distilbert": ("distilbert.transformer.layer.{i}", "attention", "ffn"),
    # T5 / encoder-decoder
    "t5": ("encoder.block.{i}", "layer.0.SelfAttention", "layer.1.DenseReluDense"),
    # OPT
    "opt": ("model.decoder.layers.{i}", "self_attn", "fc1"),
    # Falcon
    "falcon": ("transformer.h.{i}", "self_attention", "mlp"),
}


class HuggingFaceAdapter(BaseAdapter):
    """Adapter for HuggingFace PreTrainedModel models."""

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        self._tokenizer = None
        self._model_type = self._detect_model_type()
        self._num_layers = self._detect_num_layers()
        self._pattern = _ARCHITECTURE_PATTERNS.get(self._model_type)

    # ---- Identity ----

    @property
    def type_of_adapter(self) -> str:
        return "huggingface"

    @property
    def architecture_family(self) -> str:
        return "transformer"

    # ---- Configuration ----

    def set_tokenizer(self, tokenizer) -> None:
        """Attach a HuggingFace tokenizer for automatic input processing."""
        self._tokenizer = tokenizer

    # ---- Capabilities ----

    def capabilities(self) -> Set[AnalysisCapability]:
        caps = {
            AnalysisCapability.HOOKS,
            AnalysisCapability.ACTIVATION_PATCHING,
            AnalysisCapability.LAYER_PROBING,
            AnalysisCapability.ATTENTION_MAPS,
            AnalysisCapability.EMBEDDINGS,
        }
        if self.has_residual_connections():
            caps.add(AnalysisCapability.RESIDUAL_STREAM)
        return caps

    def has_residual_connections(self) -> bool:
        """All standard transformers have residual connections."""
        return True

    # ---- Detection Helpers ----

    def _detect_model_type(self) -> Optional[str]:
        """Get model_type from HuggingFace config."""
        if hasattr(self.model, "config") and hasattr(self.model.config, "model_type"):
            return self.model.config.model_type
        return None

    def _detect_num_layers(self) -> int:
        """Detect number of transformer blocks."""
        config = getattr(self.model, "config", None)
        if config is None:
            return 0

        # Different config attribute names across architectures
        for attr in ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]:
            if hasattr(config, attr):
                return getattr(config, attr)
        return 0

    # ---- Universal Methods ----

    def get_layer_names(self) -> List[str]:
        """Return all named layers, excluding the root module."""
        return [name for name, _ in self.model.named_modules() if name]

    def get_patchable_layers(self) -> List[str]:
        """
        Return attn + mlp sublayers suitable for activation patching.
        Uses config.model_type for precise detection, falls back to
        name-based search.
        """
        if self._pattern and self._num_layers:
            block_pat, attn_suffix, mlp_suffix = self._pattern
            layers = []
            for i in range(self._num_layers):
                block = block_pat.format(i=i)
                layers.append(f"{block}.{attn_suffix}")
                layers.append(f"{block}.{mlp_suffix}")
            # Verify they actually exist in the model
            available = set(dict(self.model.named_modules()).keys())
            return [l for l in layers if l in available]

        # Fallback: name-based search
        return self._find_by_keywords(
            ["attn", "attention", "self_attn", "mlp", "feed_forward", "ffn"]
        )

    def forward(self, inputs, **kwargs) -> torch.Tensor:
        """
        Run a forward pass. Handles raw strings (if tokenizer is set)
        and pre-tokenized inputs.
        """
        if isinstance(inputs, str) and self._tokenizer:
            tokens = self._tokenizer(inputs, return_tensors="pt")
            output = self.model(**tokens, **kwargs)
        elif isinstance(inputs, dict) or hasattr(inputs, "input_ids"):
            output = self.model(**inputs, **kwargs)
        else:
            output = self.model(inputs, **kwargs)

        if hasattr(output, "logits"):
            return output.logits
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        return output

    def tokenize(self, inputs, **kwargs) -> Dict:
        """Tokenize inputs using the attached HuggingFace tokenizer."""
        if self._tokenizer is None:
            raise ValueError(
                "No tokenizer set. Use adapter.set_tokenizer(tokenizer) first."
            )
        return self._tokenizer(inputs, return_tensors="pt", **kwargs)

    # ---- Optional Methods (all supported for transformers) ----

    def get_attention_layers(self) -> List[str]:
        """Return attention layer names using config-based detection."""
        if self._pattern and self._num_layers:
            block_pat, attn_suffix, _ = self._pattern
            layers = [
                f"{block_pat.format(i=i)}.{attn_suffix}"
                for i in range(self._num_layers)
            ]
            available = set(dict(self.model.named_modules()).keys())
            return [l for l in layers if l in available]

        # Fallback
        return self._find_by_keywords(["attn", "attention", "self_attn"])

    def get_output_projection(self) -> Optional[torch.Tensor]:
        """
        Return the unembedding matrix that maps hidden states → vocab space.
        Checks common locations across HuggingFace architectures.
        """
        # GPT-2, LLaMA, Mistral, etc.
        if hasattr(self.model, "lm_head") and hasattr(self.model.lm_head, "weight"):
            return self.model.lm_head.weight.detach()

        # BERT-style masked LM
        if hasattr(self.model, "cls"):
            if hasattr(self.model.cls, "predictions"):
                decoder = self.model.cls.predictions.decoder
                if hasattr(decoder, "weight"):
                    return decoder.weight.detach()

        # Shared embedding weight (tied weights)
        if hasattr(self.model, "get_output_embeddings"):
            out_emb = self.model.get_output_embeddings()
            if out_emb is not None and hasattr(out_emb, "weight"):
                return out_emb.weight.detach()

        return None

    def get_embedding_layer(self) -> Optional[torch.nn.Module]:
        """Return the input embedding layer."""
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings()
        return None

    def get_sequential_layers(self) -> List[str]:
        """
        Return transformer blocks in execution order.
        Uses config to build an explicit ordered list — never relies
        on named_modules() iteration order.
        """
        if self._pattern and self._num_layers:
            block_pat = self._pattern[0]
            layers = [block_pat.format(i=i) for i in range(self._num_layers)]
            available = set(dict(self.model.named_modules()).keys())
            return [l for l in layers if l in available]

        # Fallback: find numbered block patterns
        import re

        blocks = []
        for name, _ in self.model.named_modules():
            if re.match(r".*\.\d+$", name):
                blocks.append(name)
        return blocks

    # ---- Private Helpers ----

    def _find_by_keywords(self, keywords: List[str]) -> List[str]:
        """Find layers whose names contain any of the given keywords."""
        return [
            name
            for name, _ in self.model.named_modules()
            if any(kw in name.lower() for kw in keywords)
        ]
