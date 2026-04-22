import torch
import torch.nn as nn


class ToyTransformerBlock(nn.Module):
    """Single transformer block with attention, MLP, and residual connections."""

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x):
        # Attention + residual
        normed = self.ln_1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # MLP + residual
        normed = self.ln_2(x)
        x = x + self.mlp(normed)

        return x


class ToyTransformer(nn.Module):
    """
    Minimal transformer for testing ModelLens with vanilla PyTorch.
    Not trained — random weights, just for verifying the toolkit works.
    """

    def __init__(self, vocab_size=100, hidden_dim=64, num_heads=4, num_layers=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.blocks = nn.ModuleList(
            [ToyTransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x shape: (batch, seq_len) of token ids
        x = self.embed(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits


if __name__ == "__main__":
    # Quick sanity check
    model = ToyTransformer()
    input_ids = torch.randint(0, 100, (1, 10))  # batch=1, seq_len=10
    output = model(input_ids)
    print(f"Input shape:  {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nNamed modules:")
    for name, _ in model.named_modules():
        if name:
            print(f"  {name}")
