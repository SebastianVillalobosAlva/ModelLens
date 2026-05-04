# ModelLens

**Architecture-agnostic neural network interpretability toolkit.**

Point ModelLens at any PyTorch model — transformer, CNN, LSTM, GRU, or MLP — and get the right interpretability analyses for that architecture automatically.

Unlike tools like [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) that focus exclusively on transformers, ModelLens is designed for researchers and engineers who need interpretability across architecture families.

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from modellens import ModelLens

model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

lens = ModelLens(model)
lens.adapter.set_tokenizer(tokenizer)

print(lens)
# ModelLens(
#   backend=huggingface,
#   architecture=transformer,
#   params=124,439,808,
#   analyses=['activation_patching', 'attention_maps', 'embeddings',
#             'hooks', 'layer_probing', 'residual_stream']
# )

inputs = tokenizer("The capital of France is", return_tensors="pt")
results = lens.layer_probe(inputs, top_k=5)
```

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

## Supported Architectures

| Architecture | Adapter | Analyses |
|---|---|---|
| **Transformers** (GPT-2, LLaMA, Mistral, Gemma, BERT, etc.) | HuggingFaceAdapter | Layer probing, attention maps, activation patching, residual stream, embeddings |
| **CNNs** (custom, ResNet, VGG, etc.) | PyTorchAdapter | Filter analysis, feature map evolution, layer probing, activation patching |
| **LSTMs / GRUs / RNNs** | PyTorchAdapter | Gate analysis, layer probing, activation patching, embeddings |
| **MLPs / Feedforward** | PyTorchAdapter | Layer probing, activation patching |

Composite architectures (Autoencoders, VAEs, GANs) work out of the box since they're built from supported layer types.

For deep transformer-specific analysis with 50+ model support, we recommend [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens). ModelLens is designed for researchers who need interpretability across architecture families.

## Roadmap

- **State Space Models** (Mamba, S4) — state evolution and selective scan analysis
- **Graph Neural Networks** — message passing and node embedding analysis
- **MCP Server** — wrap ModelLens as an MCP server so Claude can run interpretability analyses directly

## Architecture

```
modellens/
├── core/
│   ├── lens.py          # ModelLens — main entry point
│   └── hooks.py         # HookManager — activation capture
├── adapters/
│   ├── base.py          # BaseAdapter + AnalysisCapability enum
│   ├── huggingface_adapter.py  # HuggingFace transformers
│   └── pytorch_adapter.py      # Generic PyTorch (CNN, LSTM, MLP)
└── analysis/
    ├── logit_lens.py        # Layer probing (generalized logit lens)
    ├── attention.py         # Attention map extraction
    ├── activation_patching.py  # Causal intervention
    ├── residual_stream.py   # Residual stream analysis
    ├── embeddings.py        # Embedding inspection
    ├── filters.py           # CNN filter + feature map analysis
    └── gates.py             # LSTM/GRU gate analysis
```

## Testing

```bash
pip install pytest
pytest tests/ -v
```

## License

MIT