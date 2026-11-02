"""Tests for the IR critic parser and pipeline adapter."""

from __future__ import annotations

import json

from paperbanana.agents.ir_critic import IRCriticAgent, IRCritiqueResult


def _ir() -> dict:
    return {
        "title": "Test",
        "layout_direction": "LR",
        "nodes": [
            {"id": "n1", "label": "Input"},
            {"id": "n2", "label": "Model"},
            {"id": "n3", "label": "Output"},
        ],
        "edges": [
            {"source": "n1", "target": "n2"},
            {"source": "n2", "target": "n3"},
        ],
        "groups": [],
    }


def test_parse_no_revision():
    result = IRCriticAgent._parse(
        json.dumps({"needs_revision": False, "issues": [], "revised_ir": None}),
        fallback_title="Test",
    )
    assert result.revised_ir is None
    assert result.issues == []


def test_parse_revision():
    ir = _ir()
    ir["nodes"][1]["label"] = "Core Model"
    result = IRCriticAgent._parse(
        json.dumps(
            {
                "needs_revision": True,
                "issues": ["The middle module is labelled incorrectly."],
                "revised_ir": ir,
            }
        ),
        fallback_title="Test",
    )
    assert result.revised_ir is not None
    assert result.revised_ir.nodes[1].label == "Core Model"
    assert result.issues


def test_parse_invalid_json():
    result = IRCriticAgent._parse("not json", fallback_title="Test")
    assert result == IRCritiqueResult()


def test_to_critique_result_requires_revised_ir():
    critique = IRCriticAgent.to_critique_result(IRCritiqueResult(issues=["missing node"]))
    assert critique.critic_suggestions == []
    assert critique.revised_description is None
