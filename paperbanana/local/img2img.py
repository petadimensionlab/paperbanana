"""Optional local img2img/ControlNet styling for infographic output."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger()

DEFAULT_MODEL = "flux2-klein-4b"
DEFAULT_PROMPT = (
    "polished academic infographic, soft pastel watercolor, clean vector cards, "
    "subtle shadows and depth, calm professional palette, no text, no letters, "
    "no words, no labels, preserve the layout and boxes"
)


def find_img2img_bin(explicit: Optional[str] = None) -> Optional[str]:
    candidates = [
        explicit,
        shutil.which("mflux-generate-flux2"),
        "/tmp/mflux-venv/bin/mflux-generate-flux2",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def style_infographic(
    structure_image: str | Path,
    output_image: str | Path,
    *,
    prompt: str = DEFAULT_PROMPT,
    strength: float = 0.30,
    model: str = DEFAULT_MODEL,
    mflux_bin: Optional[str] = None,
    width: int = 1536,
    height: int = 864,
    steps: int = 4,
    seed: int = 42,
    timeout: float = 900.0,
) -> Optional[Path]:
    """Run local img2img on a textless structure image. Returns None on failure."""
    binary = find_img2img_bin(mflux_bin)
    if not binary:
        logger.info("mflux img2img not available; keeping vector-only render")
        return None
    source = Path(structure_image)
    target = Path(output_image)
    if not source.exists():
        logger.warning("Structure image not found", path=str(source))
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
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
        "--image",
        str(source),
        str(strength),
        "--prompt",
        prompt,
        "--output",
        str(target),
    ]
    if "flux2" not in model.lower():
        cmd.extend(
            [
                "--negative-prompt",
                "text, letters, words, labels, typography, watermark, diagram, chart, screenshot",
            ]
        )
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0 or not target.exists():
        logger.warning(
            "Local img2img failed",
            returncode=result.returncode,
            stderr=(result.stderr or "")[-300:],
        )
        return None
    return target
