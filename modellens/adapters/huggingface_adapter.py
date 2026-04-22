from typing import Dict, List, Optional

import torch

from modellens.adapters.base import BaseAdapter


class HuggingFaceAdapter(BaseAdapter):
    """Adapter for HuggingFace PreTrainedModel models."""

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        # Store tokenizer reference — set via set_tokenizer()
        self._tokenizer = None

    @property
    def type_of_adapter(self) -> str:
        return "huggingface"

    def set_tokenizer(self, tokenizer) -> None:
        """Attach a HuggingFace tokenizer for automatic input processing."""
        self._tokenizer = tokenizer

    def get_layer_names(self, model: torch.nn.Module) -> List[str]:
        """Return all named layers, excluding the root module."""
        return [name for name, _ in model.named_modules() if name]

    def get_attention_layers(self, model: torch.nn.Module) -> List[str]:
        """
        Find attention layers using HuggingFace's config when available,
        falling back to name-based detection.
        """
        # Try config-based detection first
        if hasattr(model, "config") and hasattr(model.config, "n_layer"):
            # GPT-2 style naming
            return [f"transformer.h.{i}.attn" for i in range(model.config.n_layer)]

        if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
            # BERT style naming
            return [
                f"encoder.layer.{i}.attention"
                for i in range(model.config.num_hidden_layers)
            ]

        # Fallback to name-based search
        keywords = ["attn", "attention", "self_attn"]
        return [
            name
            for name, _ in model.named_modules()
            if any(kw in name.lower() for kw in keywords)
        ]

    def get_unembedding(self, model: torch.nn.Module) -> Optional[torch.Tensor]:
        """
        Find the unembedding matrix. HuggingFace models typically store this
        in lm_head or cls for different model types.
        """
        # GPT-2, GPT-Neo, etc.
        if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
            return model.lm_head.weight.detach()

        # BERT-style masked LM
        if hasattr(model, "cls"):
            if hasattr(model.cls, "predictions"):
                decoder = model.cls.predictions.decoder
                if hasattr(decoder, "weight"):
                    return decoder.weight.detach()

        # Shared embedding weight (tied weights)
        if hasattr(model, "get_output_embeddings"):
            out_emb = model.get_output_embeddings()
            if out_emb is not None and hasattr(out_emb, "weight"):
                return out_emb.weight.detach()

        return None

    def forward(self, model: torch.nn.Module, inputs, **kwargs) -> torch.Tensor:
        """
        Run a forward pass. Handles both raw strings (if tokenizer is set)
        and pre-tokenized inputs.
        """
        # If inputs are strings and we have a tokenizer, tokenize first
        if isinstance(inputs, str) and self._tokenizer:
            tokens = self._tokenizer(inputs, return_tensors="pt")
            output = model(**tokens, **kwargs)
        elif isinstance(inputs, dict) or hasattr(inputs, "input_ids"):
            output = model(**inputs, **kwargs)
        else:
            output = model(inputs, **kwargs)

        # HuggingFace models return dataclass-like objects
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
