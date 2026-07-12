# ModelLens

**Architecture-agnostic neural network interpretability toolkit.**

Point ModelLens at any PyTorch model — transformer, CNN, LSTM, GRU, or MLP — and get the right interpretability analyses for that architecture automatically.

Unlike tools like [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) that focus exclusively on transformers, ModelLens is designed for researchers and engineers who need interpretability across architecture families.

## Flagship example: do CAA and LoRA steer through the same circuit?

ModelLens is architecture-agnostic, but its sharpest use is **causal**: given two versions of a model, did an intervention actually *rewire the circuit* behind a behavior, or just nudge the output? That question can't be answered by behavioral evals — you have to look inside.

This is the flagship result from the companion [Stoic-Steering](https://github.com/SebastianVillalobosAlva/Stoic-Steering) project (Exp 12), which steers Llama-3.2-1B toward Stoic philosophy two ways — Contrastive Activation Addition (CAA) and LoRA fine-tuning — and uses ModelLens's `discover_circuit` to compare what each does *mechanistically*.

```python
from modellens import ModelLens

# Three variants of Llama-3.2-1B from the Stoic-Steering project:
#   base, CAA-steered (layer 12, coefficient 0.11), and LoRA fine-tuned.
lens_base = ModelLens(base_model)
lens_caa  = ModelLens(caa_model)
lens_lora = ModelLens(lora_model)

# A Stoic-vs-neutral contrast pair (clean = Stoic framing, corrupted = neutral).
clean     = tokenizer("The obstacle is within my control, so", return_tensors="pt")
corrupted = tokenizer("The weather today is", return_tensors="pt")

# Discover the causal circuit behind the Stoic continuation in each variant.
circuit_base = lens_base.discover_circuit(clean, corrupted)
circuit_caa  = lens_caa.discover_circuit(clean, corrupted)
circuit_lora = lens_lora.discover_circuit(clean, corrupted)
```

**Finding.** Comparing the discovered circuits:

- **CAA at coefficient 0.11 is a circuit-level no-op** — the steered model's circuit matches the base model's. Adding a steering vector at one layer shifts the output distribution without rewiring the causal pathway.
- **LoRA rewires the Stoic-content circuit** — the fine-tuned model routes the behavior through a measurably different set of components, and the rewiring is **largest for Seneca** among the three philosophers.

Two interventions that look similar at the output level operate through **different internal mechanisms** — exactly the kind of claim ModelLens exists to make. See the [Stoic-Steering README](https://github.com/SebastianVillalobosAlva/Stoic-Steering), which names ModelLens as its companion interpretability toolkit.

> **New to ModelLens?** See [How It Works](#how-it-works) below for the one-liner API (`ModelLens(model)`) and per-architecture examples.

## Installation

```bash
git clone https://github.com/SebastianVillalobosAlva/ModelLens.git
cd ModelLens
pip install -e .
```

## How It Works

ModelLens uses an **adapter pattern** to support different architectures. When you pass a model, it auto-detects the architecture family and declares which analyses are available:

```python
# Transformer
lens = ModelLens(gpt2_model)
lens.available_analyses()
# ['activation_patching', 'attention_maps', 'embeddings',
#  'hooks', 'layer_probing', 'residual_stream']

# CNN
lens = ModelLens(resnet)
lens.available_analyses()
# ['activation_patching', 'feature_map_analysis', 'filter_analysis',
#  'hooks', 'layer_probing']

# LSTM
lens = ModelLens(lstm_model)
lens.available_analyses()
# ['activation_patching', 'embeddings', 'gate_analysis',
#  'hooks', 'layer_probing']

# MLP
lens = ModelLens(feedforward_net)
lens.available_analyses()
# ['activation_patching', 'hooks', 'layer_probing']
```

Calling an unsupported analysis gives a clear error:
```python
lens = ModelLens(my_cnn)
lens.attention_map(inputs)
# UnsupportedAnalysisError: 'attention_map' is not supported for
# convolutional models. Available analyses: [filter_analysis, ...]
```

## Analyses

### Layer Probing (Generalized Logit Lens)

Project intermediate layer representations through the output projection to see what the model would predict at each layer. Works on any architecture with a hidden → output mapping.

```python
results = lens.layer_probe(inputs, top_k=5)

from modellens.analysis.logit_lens import decode_logit_lens
decoded = decode_logit_lens(results, tokenizer=tokenizer)
for layer, preds in decoded.items():
    print(f"{layer}: {preds[:3]}")
```

### Attention Maps (Transformers)

Extract attention weight maps from transformer layers.

```python
attn = lens.attention_map(inputs)
for layer, data in attn["attention_maps"].items():
    print(f"{layer}: {data['num_heads']} heads, seq_len={data['seq_length']}")
```

> **Note:** Models using SDPA or Flash Attention must be loaded with `attn_implementation="eager"` to return attention weights.

### Activation Patching

Replace activations from a clean run with those from a corrupted run to measure each layer's causal impact. Works on all architectures.

```python
clean = tokenizer("The Eiffel Tower is in", return_tensors="pt")
corrupted = tokenizer("The Colosseum is in", return_tensors="pt")

results = lens.activation_patch(clean, corrupted)
for layer, effect in results["patch_effects"].items():
    print(f"{layer}: {effect['normalized_effect']:.3f}")
```

### Residual Stream Analysis (Transformers, ResNets)

Measure how much each layer contributes to the final representation through the residual stream.

```python
residual = lens.residual_stream(inputs)
for layer, data in residual["contributions"].items():
    print(f"{layer}: relative_contribution={data['relative_contribution']:.4f}")

from modellens.analysis.residual_stream import identify_critical_layers
critical = identify_critical_layers(residual, threshold=0.1)
print(f"Critical layers: {critical}")
```

### Embedding Analysis

Inspect input embedding representations and compute similarity between token positions.

```python
emb = lens.embeddings(inputs)
print(f"Embedding dim: {emb['embed_dim']}")
print(f"Similarity matrix: {emb['similarity_matrix'].shape}")
```

### Filter Analysis (CNNs)

Analyze convolutional filters: feature map statistics, dead filter detection, and filter weight inspection.

```python
lens = ModelLens(cnn_model)
img = torch.randn(1, 3, 224, 224)

filters = lens.filter_analysis(img)
print(f"Total filters: {filters['total_filters']}")
print(f"Dead filters: {filters['total_dead_filters']}")

from modellens.analysis.filters import find_most_active_filters
top = find_most_active_filters(filters, "conv1", top_k=5)
```

### Feature Map Evolution (CNNs)

Track how spatial representations evolve through the network.

```python
fmaps = lens.feature_maps(img)
print(f"Spatial reduction: {fmaps['spatial_reduction']}x")
print(f"Channel expansion: {fmaps['channel_expansion']}x")

for entry in fmaps["evolution"]:
    print(f"  {entry['layer']}: {entry['channels']}ch, "
          f"{entry['spatial_h']}x{entry['spatial_w']}, "
          f"sparsity={entry['sparsity']:.2f}")
```

### Gate Analysis (LSTMs, GRUs)

Decompose gate activations in recurrent models to understand how they process sequences.

```python
lens = ModelLens(lstm_model)
tokens = torch.randint(0, vocab_size, (1, 50))

gates = lens.gate_analysis(tokens)
result = gates["layer_results"]["lstm"]

print(f"Hidden norm trend: {result['hidden_evolution']['norm_trend']}")
for stat in result["gate_weight_stats"]:
    layer = stat["layer"]
    for gate in ["input", "forget", "cell", "output"]:
        print(f"  Layer {layer} {gate}: norm={stat[gate]['input_weight_norm']:.3f}")
```

### Sparse Autoencoders (Dictionary Learning)

Train a sparse autoencoder on a model's activations, then inspect the features it learned. Training builds and fits a new network, so it's a module-level call — not a lens method. Inspection comes in three complementary views.

```python
from modellens.analysis.sparse_autoencoder import train_sae

# Fit an overcomplete SAE on a host model's activations.
# layer_name defaults to a residual-stream layer when the model has one.
sae, summary = train_sae(lens, inputs, layer_name="transformer.h.5",
                         expansion=4, steps=500)
print(f"recon {summary['initial_recon_loss']:.3f} -> {summary['final_recon_loss']:.3f}, "
      f"dead={summary['dead_features']}")

# 1) What features mean — max-activating inputs/tokens per feature
feats = lens.sae_features(inputs, sae, layer_name="transformer.h.5", tokenizer=tokenizer)
for feature, data in list(feats["feature_summary"].items())[:3]:
    print(feature, data["top_activations"][0])

# Point the lens at the trained SAE itself for the dictionary-level views:
sae_lens = ModelLens(sae)
sae_lens.available_analyses()  # includes 'dictionary_analysis'

# 2) SAE health — dead features, firing rates, activation stats.
# Pass activation vectors in the SAE's input space, shape (N, input_dim)
# (in practice, activations gathered from the host at the trained layer).
activations = torch.randn(64, sae.input_dim)
health = sae_lens.dictionary_features(activations, top_k=10)
print(f"{len(health['dead_features'])} dead / {health['num_features']} features")

# 3) The learned dictionary — the decoder direction each feature writes
directions = sae_lens.feature_directions()
print(directions["directions"].shape)  # (num_features, input_dim)
```

## Supported Architectures

| Architecture | Adapter | Analyses |
|---|---|---|
| **Transformers** (GPT-2, LLaMA, Mistral, Gemma, BERT, etc.) | HuggingFaceAdapter | Layer probing, attention maps, activation patching, residual stream, embeddings |
| **CNNs** (custom, ResNet, VGG, etc.) | PyTorchAdapter | Filter analysis, feature map evolution, layer probing, activation patching |
| **LSTMs / GRUs / RNNs** | PyTorchAdapter | Gate analysis, layer probing, activation patching, embeddings |
| **MLPs / Feedforward** | PyTorchAdapter | Layer probing, activation patching |
| **Autoencoders** (overcomplete / sparse) | PyTorchAdapter | Dictionary analysis, feature directions, layer probing, activation patching |

Overcomplete autoencoders are detected automatically (input dim == output dim, with a wider hidden layer) and gain SAE dictionary analysis. Other composite architectures (VAEs, GANs) work out of the box since they're built from supported layer types.

For deep transformer-specific analysis with 50+ model support, we recommend [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens). ModelLens is designed for researchers who need interpretability across architecture families.

## MCP Server

ModelLens ships an [MCP](https://modelcontextprotocol.io) server so
interpretability analyses can be driven conversationally (e.g. by Claude).
It exposes `logit_lens`, `layer_evolution`, `discover_circuit`, and
`sae_analysis` as tools; each loads a small HuggingFace model by name, runs
one analysis, and returns JSON-safe output. `sae_analysis` trains a sparse
autoencoder and returns three consistent views of it — feature meaning, health,
and the learned dictionary.

```bash
pip install -e ".[mcp]"   # install the optional MCP dependency
modellens-mcp             # start the server (stdio transport)
```

Point any MCP client at the `modellens-mcp` command to make the tools available.

## Roadmap

- **State Space Models** (Mamba, S4) — state evolution and selective scan analysis
- **Graph Neural Networks** — message passing and node embedding analysis

## Architecture

```
modellens/
├── core/
│   ├── lens.py          # ModelLens — main entry point
│   └── hooks.py         # HookManager — activation capture
├── adapters/
│   ├── base.py          # BaseAdapter + AnalysisCapability enum
│   ├── huggingface_adapter.py  # HuggingFace transformers
│   └── pytorch_adapter.py      # Generic PyTorch (CNN, LSTM, MLP, autoencoder)
├── analysis/
│   ├── logit_lens.py        # Layer probing (generalized logit lens)
│   ├── layer_evolution.py   # Prediction-distribution evolution across layers
│   ├── attention.py         # Attention map extraction
│   ├── activation_patching.py  # Causal intervention
│   ├── circuit_discovery.py    # Automatic circuit discovery
│   ├── residual_stream.py   # Residual stream analysis
│   ├── embeddings.py        # Embedding inspection
│   ├── filters.py           # CNN filter + feature map analysis
│   ├── gates.py             # LSTM/GRU gate analysis
│   └── sparse_autoencoder.py   # SAE training + dictionary analysis
├── helpers/
│   ├── activations.py   # Shared activation-shape utilities
│   └── layers.py        # Shared layer-selection utilities
└── mcp/
    └── server.py        # MCP server (modellens-mcp)
```

## Testing

```bash
pip install pytest
pytest tests/ -v
```

## License

MIT
