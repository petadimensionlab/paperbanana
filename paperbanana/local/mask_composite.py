"""Mask compositing for hybrid infographic renders.

The diffusion pass supplies texture, but the card and panel geometry must stay
identical to the deterministic structure image so the SVG text remains aligned.
This module keeps the structure's card/panel shapes and borders while blending
a controlled amount of the styled image inside them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import structlog
from PIL import Image, ImageDraw

logger = structlog.get_logger()


def _rounded_mask(
    size: tuple[int, int],
    rect: tuple[float, float, float, float],
    radius: int,
    *,
    inset: int = 0,
) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    x, y, w, h = (int(round(v)) for v in rect)
    x += inset
    y += inset
    w -= 2 * inset
    h -= 2 * inset
    if w <= 0 or h <= 0:
        return mask
    draw.rounded_rectangle((x, y, x + w, y + h), radius=max(0, radius - inset), fill=255)
    return mask


def composite_with_card_masks(
    structure_image: str | Path,
    styled_image: str | Path,
    output_image: str | Path,
    card_rects: list[tuple[float, float, float, float]],
    panel_rects: list[tuple[float, float, float, float]],
    *,
    card_structure_weight: float = 0.72,
    panel_structure_weight: float = 0.30,
    border_weight: float = 0.95,
) -> Optional[Path]:
    """Composite styled texture under fixed structure card/panel geometry."""
    structure = Image.open(structure_image).convert("RGB")
    styled = Image.open(styled_image).convert("RGB").resize(structure.size)
    width, height = structure.size
    structure_np = np.asarray(structure, dtype=np.float32)
    styled_np = np.asarray(styled, dtype=np.float32)
    out = styled_np.copy()

    # Panels: keep most of the styled texture but preserve the panel fill edge.
    for rect in panel_rects:
        panel_mask = np.asarray(_rounded_mask((width, height), rect, 18), dtype=bool)
        out[panel_mask] = (
            panel_structure_weight * structure_np[panel_mask]
            + (1.0 - panel_structure_weight) * styled_np[panel_mask]
        )
        panel_border = np.asarray(
            _rounded_mask((width, height), rect, 18), dtype=bool
        ) & ~np.asarray(_rounded_mask((width, height), rect, 18, inset=2), dtype=bool)
        out[panel_border] = (
            border_weight * structure_np[panel_border]
            + (1.0 - border_weight) * styled_np[panel_border]
        )

    # Cards: mostly the clean white structure, with a little styled texture.
    for rect in card_rects:
        card_mask = np.asarray(_rounded_mask((width, height), rect, 12), dtype=bool)
        out[card_mask] = (
            card_structure_weight * structure_np[card_mask]
            + (1.0 - card_structure_weight) * styled_np[card_mask]
        )
        card_border = card_mask & ~np.asarray(
            _rounded_mask((width, height), rect, 12, inset=2), dtype=bool
        )
        out[card_border] = (
            border_weight * structure_np[card_border]
            + (1.0 - border_weight) * styled_np[card_border]
        )

    result = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    target = Path(output_image)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.save(target)
    logger.info(
        "Mask-composited styled image",
        cards=len(card_rects),
        panels=len(panel_rects),
        card_structure_weight=card_structure_weight,
    )
    return target
