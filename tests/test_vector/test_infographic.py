"""Tests for the rich infographic SVG renderer."""

from __future__ import annotations

import html
import shutil
from pathlib import Path

import pytest

from paperbanana.core.types import DiagramIR
from paperbanana.vector.infographic_svg import render_infographic_svg, write_infographic


def _ir() -> DiagramIR:
    return DiagramIR.model_validate(
        {
            "title": "Test Architecture",
            "layout_direction": "LR",
            "nodes": [
                {"id": "n1", "label": "Input Tokens"},
                {"id": "n2", "label": r"Router (O(n \sqrt(n)))"},
                {"id": "n3", "label": "Output"},
            ],
            "edges": [
                {"source": "n1", "target": "n2"},
                {"source": "n2", "target": "n3"},
            ],
            "groups": [{"id": "g1", "label": "Encoder", "node_ids": ["n2"]}],
        }
    )


def test_render_contains_exact_labels_and_title(tmp_path: Path):
    svg = render_infographic_svg(_ir())
    text = html.unescape(svg)
    assert "Test Architecture" in text
    assert "Input Tokens" in text
    assert "Output" in text
    assert "√(n)" in text
    assert 'width="1600"' in svg and 'height="900"' in svg


@pytest.mark.skipif(shutil.which("rsvg-convert") is None, reason="rsvg-convert not installed")
def test_write_infographic_emits_svg_png_pdf(tmp_path: Path):
    svg = write_infographic(_ir(), tmp_path / "figure")
    assert svg.exists()
    assert svg.with_suffix(".png").exists()
    assert svg.with_suffix(".pdf").exists()
