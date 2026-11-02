"""Tests for the deterministic IR rendering modes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperbanana.core.config import Settings
from paperbanana.core.pipeline import PaperBananaPipeline
from paperbanana.core.types import DiagramType, GenerationInput
from paperbanana.providers.image_gen.dummy import DummyImageGen
from paperbanana.vector.graphviz_render import find_dot_executable


class _MockVLM:
    name = "mock"
    model_name = "mock"
    supports_json_mode = True

    def __init__(self, response: str):
        self._response = response
        self.calls = 0

    async def generate(
        self,
        prompt,
        images=None,
        system_prompt=None,
        temperature=1.0,
        max_tokens=4096,
        response_format=None,
    ):
        self.calls += 1
        return self._response


def _ir_json() -> str:
    return json.dumps(
        {
            "title": "Test pipeline",
            "layout_direction": "LR",
            "nodes": [
                {"id": "n1", "label": "Input Data"},
                {"id": "n2", "label": "Encoder"},
                {"id": "n3", "label": "Output"},
            ],
            "edges": [
                {"source": "n1", "target": "n2"},
                {"source": "n2", "target": "n3"},
            ],
            "groups": [],
        }
    )


@pytest.mark.asyncio
@pytest.mark.skipif(find_dot_executable() is None, reason="graphviz not installed")
async def test_ir_fast_renders_without_image_gen_or_critic(tmp_path: Path):
    vlm = _MockVLM(_ir_json())
    settings = Settings(
        render_mode="ir_fast",
        vlm_provider="ollama",
        image_provider="google_imagen",
        vector_export="both",
        output_dir=str(tmp_path),
        save_prompts=False,
        save_iterations=True,
        export_tikz=True,
        refinement_iterations=1,
    )
    pipeline = PaperBananaPipeline(
        settings=settings, vlm_client=vlm, image_gen_fn=DummyImageGen()
    )

    result = await pipeline.generate(
        GenerationInput(
            source_context="A three-stage pipeline: input, encoder, output.",
            communicative_intent="Overview of the pipeline",
            diagram_type=DiagramType.METHODOLOGY,
        )
    )

    run_dir = Path(result.image_path).parent
    assert Path(result.image_path).exists()
    assert Path(result.image_path).with_suffix(".svg").exists()
    assert (run_dir / "final_output.svg").exists()
    assert (run_dir / "final_output.pdf").exists()
    assert (run_dir / "diagram_ir.json").exists()
    assert result.tikz_path is not None
    tex = Path(result.tikz_path).read_text(encoding="utf-8")
    assert "Input Data" in tex and "Encoder" in tex and "Output" in tex
    # Fast mode: one planner call, no critic, no image model.
    assert vlm.calls == 1
    assert result.metadata["image_provider"] == "none"


@pytest.mark.asyncio
@pytest.mark.skipif(find_dot_executable() is None, reason="graphviz not installed")
async def test_ir_quality_uses_structurer_and_critic(tmp_path: Path):
    vlm = _MockVLM(_ir_json())
    settings = Settings(
        render_mode="ir_quality",
        vlm_provider="ollama",
        image_provider="google_imagen",
        vector_export="svg",
        output_dir=str(tmp_path),
        save_prompts=False,
        save_iterations=True,
        refinement_iterations=1,
    )
    pipeline = PaperBananaPipeline(
        settings=settings, vlm_client=vlm, image_gen_fn=DummyImageGen()
    )

    result = await pipeline.generate(
        GenerationInput(
            source_context="A three-stage pipeline: input, encoder, output.",
            communicative_intent="Overview of the pipeline",
            diagram_type=DiagramType.METHODOLOGY,
        )
    )

    run_dir = Path(result.image_path).parent
    assert Path(result.image_path).exists()
    assert (run_dir / "diagram_ir.json").exists()
    # Quality mode uses Retriever + Planner + Stylist + Structurer + Critic.
    assert vlm.calls >= 4
