"""Tests for container-node flattening."""

from __future__ import annotations

from paperbanana.core.diagram_ir import flatten_container_nodes
from paperbanana.core.types import DiagramIR


def test_flatten_container_and_rewire_edges():
    ir = DiagramIR.model_validate(
        {
            "title": "Test",
            "layout_direction": "LR",
            "nodes": [
                {"id": "input", "label": "Input"},
                {"id": "enc", "label": "Encoder Stack"},
                {"id": "mhsa", "label": "MHSA"},
                {"id": "ffn", "label": "FFN"},
                {"id": "dec", "label": "Decoder Stack"},
                {"id": "linear", "label": "Linear"},
            ],
            "edges": [
                {"source": "input", "target": "enc"},
                {"source": "enc", "target": "mhsa"},
                {"source": "mhsa", "target": "ffn"},
                {"source": "enc", "target": "dec"},
                {"source": "dec", "target": "linear"},
            ],
            "groups": [
                {"id": "g_enc", "label": "Encoder Block", "node_ids": ["enc", "mhsa", "ffn"]},
                {"id": "g_dec", "label": "Decoder Block", "node_ids": ["dec"]},
            ],
        }
    )
    out = flatten_container_nodes(ir)
    ids = {n.id for n in out.nodes}
    assert "enc" not in ids
    assert "dec" in ids  # single-member group is not flattened
    edges = {(e.source, e.target) for e in out.edges}
    assert ("input", "mhsa") in edges
    assert ("ffn", "dec") in edges
