"""Tests for card-mask compositing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from paperbanana.local.mask_composite import composite_with_card_masks


def test_card_geometry_wins_inside_masks(tmp_path: Path):
    structure = tmp_path / "structure.png"
    styled = tmp_path / "styled.png"
    out = tmp_path / "out.png"
    Image.new("RGB", (100, 100), (255, 255, 255)).save(structure)
    Image.new("RGB", (100, 100), (220, 20, 20)).save(styled)

    result = composite_with_card_masks(
        structure,
        styled,
        out,
        card_rects=[(10, 10, 80, 80)],
        panel_rects=[(5, 5, 90, 90)],
        card_structure_weight=1.0,
        panel_structure_weight=0.3,
    )
    assert result is not None
    pixels = np.asarray(Image.open(result).convert("RGB"))
    # Center is inside the card and must come from the structure (white).
    assert (pixels[50, 50] > 245).all()
    # Top-left corner is outside the card and keeps the styled image.
    assert pixels[1, 1][0] > 200
