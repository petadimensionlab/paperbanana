"""Optional local diffusion helpers for infographic panel illustrations.

These helpers shell out to mflux and never call a cloud API.  They are optional:
if mflux or the model cache is unavailable, the renderer falls back to the
committed static panel assets.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import structlog
from PIL import Image, ImageFilter

from paperbanana.core.types import DiagramIR

logger = structlog.get_logger()

DEFAULT_MODEL = "flux2-klein-4b"


def find_mflux_bin(explicit: Optional[str] = None) -> Optional[str]:
    candidates = [
        explicit,
        os.environ.get("MFLUX_BIN"),
        shutil.which("mflux-generate-flux2"),
        "/tmp/mflux-venv/bin/mflux-generate-flux2",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def _cache_dir() -> Path:
    root = Path(os.environ.get("PAPERBANANA_CACHE_DIR", Path.home() / ".cache" / "paperbanana"))
    path = root / "panel-illustrations"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _panel_prompt(label: str, nodes: list[str]) -> str:
    node_hint = ", ".join(nodes[:5])
    return (
        f"abstract academic infographic illustration for a section called {label}; "
        f"concepts: {node_hint}; soft pastel watercolor, clean white background, "
        "subtle geometric shapes, minimal, no text, no letters, no words"
    )


def generate_panel_image(
    prompt: str,
    *,
    mflux_bin: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    width: int = 512,
    height: int = 256,
    steps: int = 4,
    seed: int = 42,
    timeout: float = 600.0,
) -> Optional[Path]:
    """Generate one panel illustration locally, cached by prompt hash."""
    binary = find_mflux_bin(mflux_bin)
    if not binary:
        logger.info("mflux not available; using static panel art")
        return None
    key = hashlib.sha256(
        f"v2|{prompt}|{model}|{width}x{height}|{steps}|{seed}".encode()
    ).hexdigest()[:16]
    out = _cache_dir() / f"{key}.png"
    if out.exists() and out.stat().st_size > 0:
        return out
    result = subprocess.run(
        [
            binary,
            "--model",
            model,
            "--quantize",
            "4",
            "--steps",
            str(steps),
            "--width",
            str(width),
            "--height",
            str(height),
            "--seed",
            str(seed),
            "--prompt",
            prompt,
            "--negative-prompt",
            "text, letters, words, typography, labels, watermark, diagram, chart",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0 or not out.exists():
        logger.warning(
            "Local panel illustration failed",
            returncode=result.returncode,
            stderr=(result.stderr or "")[-300:],
        )
        return None
    try:
        # Diffusion sometimes paints pseudo-text even with a negative prompt.
        # Blur it into a texture; all real text is drawn by the SVG overlay.
        image = Image.open(out).convert("RGB")
        image = image.filter(ImageFilter.GaussianBlur(radius=8))
        image.save(out)
    except OSError:
        logger.warning("Could not blur generated panel image", path=str(out))
    return out


def generate_panel_images(
    ir: DiagramIR,
    *,
    mflux_bin: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> dict[str, str]:
    """Generate panel illustrations keyed by group label (best effort)."""
    from paperbanana.vector.infographic_svg import _panel_key

    node_by_id = {n.id: n for n in ir.nodes}
    images: dict[str, str] = {}
    groups = ir.groups or []
    for group in groups:
        labels = [node_by_id[nid].label for nid in group.node_ids if nid in node_by_id]
        prompt = _panel_prompt(group.label, labels)
        image = generate_panel_image(prompt, mflux_bin=mflux_bin, model=model)
        if image is not None:
            images[group.label] = str(image)
            images.setdefault(_panel_key(group.label), str(image))
    return images
