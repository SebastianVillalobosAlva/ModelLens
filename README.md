# ModelLens

An open-source interpretability toolkit for PyTorch neural networks. ModelLens provides a unified interface for analyzing model internals across different architectures, sitting between architecture-specific tools like TransformerLens and general-purpose libraries like Captum.

## Features

- **Activation Hooks** — Capture intermediate activations from any layer using PyTorch forward hooks
- **Logit Lens** — Project hidden states through the unembedding matrix to observe how predictions evolve across layers
- **Attention Analysis** — Extract and analyze attention weight maps across heads and layers
- **Activation Patching** — Causal intervention analysis to identify which components drive specific behaviors
- **Embeddings Inspection** — Analyze input embedding representations and token similarity
- **Residual Stream Analysis** — Trace information flow through skip connections and measure per-layer contributions

## Supported Backends

- **HuggingFace** — Any `PreTrainedModel` (GPT-2, BERT, LLaMA, etc.)
- **PyTorch** — Vanilla `nn.Module` models

## Installation

```bash
git clone https://github.com/your-username/modellens.git
cd modellens
pip install -e .
```

For visualization and development dependencies:

```bash
pip install -e ".[viz,dev]"
```

## Quick Start

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from modellens import ModelLens

# Load a model
model = GPT2LMHeadModel.from_pretrained("gpt2-medium", attn_implementation="eager")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

# Wrap it with ModelLens
lens = ModelLens(model)
lens.adapter.set_tokenizer(tokenizer)

# Run logit lens analysis
tokens = tokenizer("The capital of France is", return_tensors="pt")
lens.attach_all()
results = lens.logit_lens(tokens, top_k=5)
```

## Analysis Modules

### Logit Lens

See what the model would predict at each intermediate layer:

```python
from modellens.analysis.logit_lens import run_logit_lens, decode_logit_lens

results = run_logit_lens(lens, tokens, top_k=5)
decoded = decode_logit_lens(results, tokenizer=tokenizer)

for layer, predictions in decoded.items():
    print(f"{layer} -> {predictions[0]}")
```

### Attention Analysis

Extract attention maps and identify focused vs diffuse heads:

```python
from modellens.analysis.attention import run_attention_analysis, head_summary

attn_results = run_attention_analysis(lens, "The capital of France is")
summary = head_summary(attn_results)
```

### Activation Patching

Identify causally important components by patching corrupted activations:

```python
from modellens.analysis.activation_patching import run_activation_patching

clean = tokenizer("The capital of France is", return_tensors="pt")
corrupted = tokenizer("The MX of France is", return_tensors="pt")

results = run_activation_patching(lens, clean, corrupted, metric_fn=my_metric)
```

### Residual Stream Analysis

Measure how much each layer contributes to the residual stream:

```python
from modellens.analysis.residual_stream import run_residual_analysis, identify_critical_layers

results = run_residual_analysis(lens, tokens, layer_names=block_layers)
critical = identify_critical_layers(results, threshold=0.05)
```

### Embeddings Inspection

Analyze token embeddings and their relationships:

```python
from modellens.analysis.embeddings import run_embeddings_analysis

results = run_embeddings_analysis(lens, tokens)
print(results["similarity_matrix"])
```

## Project Structure

```
modellens/
├── modellens/
│   ├── core/           # ModelLens class and hook infrastructure
│   ├── adapters/       # HuggingFace and PyTorch backend adapters
│   ├── analysis/       # Interpretability analysis modules
│   ├── visualization/  # Plotting and visualization utilities
│   └── utils/          # Shared helper functions
├── app/                # Gradio web interface
├── tests/              # Unit tests
└── examples/           # Demo notebooks
```