"""Fast IR planner: one VLM call from methodology text to DiagramIR."""

from __future__ import annotations

from typing import Optional

import structlog

from paperbanana.agents.base import BaseAgent
from paperbanana.core.diagram_ir import (
    extract_diagram_ir,
    format_diagram_ir_for_regeneration,
)
from paperbanana.core.types import DiagramIR, DiagramType, ReferenceExample
from paperbanana.core.utils import extract_json, truncate_text
from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()

MAX_CONTEXT_CHARS = 12_000


class FastIRPlannerAgent(BaseAgent):
    """Plan a diagram directly as structured IR with a single VLM call.

    This is the speed-oriented counterpart to ``PlannerAgent`` + ``StructurerAgent``.
    It skips exemplar retrieval and styling, and returns a textual rendering of
    the IR so the rest of the pipeline can keep its normal description-based
    interfaces.
    """

    def __init__(
        self, vlm_provider: VLMProvider, prompt_dir: str = "prompts", prompt_recorder=None
    ):
        super().__init__(vlm_provider, prompt_dir, prompt_recorder=prompt_recorder)
        self.last_ir: Optional[DiagramIR] = None

    @property
    def agent_name(self) -> str:
        return "fast_ir"

    async def run(
        self,
        source_context: str,
        caption: str,
        examples: Optional[list[ReferenceExample]] = None,
        diagram_type: DiagramType = DiagramType.METHODOLOGY,
        supported_ratios: Optional[list[str]] = None,
        input_images: Optional[list[str]] = None,
    ) -> tuple[str, Optional[str]]:
        """Return ``(description, recommended_ratio)``.

        ``examples``, ``supported_ratios``, and ``input_images`` are accepted for
        signature compatibility with ``PlannerAgent`` but intentionally ignored:
        the fast path sends text only and does not use image exemplars.
        """
        template = self.load_prompt("diagram")
        last_error: Optional[str] = None

        for attempt in range(2):
            repair = ""
            if attempt and last_error:
                repair = (
                    "\n\n## Previous response was invalid\n"
                    f"Fix this error and return corrected JSON only: {last_error}\n"
                )
            prompt = self.format_prompt(
                template,
                prompt_label=f"fast_ir_attempt_{attempt + 1}",
                source_context=truncate_text(source_context, MAX_CONTEXT_CHARS),
                caption=caption,
            )
            prompt += repair
            raw = await self.vlm.generate(
                prompt=prompt,
                images=None,
                temperature=0.1 if attempt == 0 else 0.0,
                max_tokens=4096,
                response_format="json",
            )
            try:
                ir = self._parse_ir(raw, caption)
                self.last_ir = ir
                description = format_diagram_ir_for_regeneration(ir)
                logger.info(
                    "Fast IR planned",
                    nodes=len(ir.nodes),
                    edges=len(ir.edges),
                    groups=len(ir.groups),
                    attempt=attempt + 1,
                )
                return description, None
            except (ValueError, TypeError, KeyError) as exc:
                last_error = str(exc)
                logger.warning(
                    "Fast IR validation failed",
                    attempt=attempt + 1,
                    error=last_error,
                )

        # Never fail the whole run on a malformed local model response: fall back
        # to the deterministic heuristic parser used by the vector exporter.
        logger.warning("Fast IR falling back to heuristic parser", error=last_error)
        ir = extract_diagram_ir(
            source_context if not caption else f"{caption}\n\n{source_context}",
            title=caption or "Methodology Diagram",
        )
        self.last_ir = ir
        return format_diagram_ir_for_regeneration(ir), None

    @staticmethod
    def _parse_ir(raw: str, caption: str) -> DiagramIR:
        data = extract_json(raw)
        if not isinstance(data, dict):
            raise ValueError("response did not contain a JSON object")
        if not data.get("title"):
            data["title"] = caption or "Methodology Diagram"
        return DiagramIR.model_validate(data)
