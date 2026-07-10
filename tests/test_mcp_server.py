import asyncio

import pytest

# The MCP server is an optional feature (pip install "modellens[mcp]").
pytest.importorskip("mcp")

from modellens.mcp import server as mcp_server


def test_server_lists_expected_tools():
    """The server object builds and exposes the curated tool set."""
    tools = asyncio.run(mcp_server.server.list_tools())
    names = {t.name for t in tools}

    expected = {"logit_lens", "layer_evolution", "discover_circuit", "sae_features"}
    assert expected.issubset(names)


def test_to_jsonable_is_serializable():
    """Tensors in analysis output are converted to JSON-safe structures."""
    import json
    import torch

    raw = {
        "small": torch.arange(4),
        "big": torch.randn(10, 100),
        "nested": {"vals": [torch.tensor(1.5), "text", 3]},
        7: "int-key",
    }
    safe = mcp_server._to_jsonable(raw)
    json.dumps(safe)  # must not raise

    assert safe["small"] == [0, 1, 2, 3]
    assert safe["big"]["_tensor"] is True
    assert safe["big"]["shape"] == [10, 100]
