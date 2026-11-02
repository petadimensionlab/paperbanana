"""IR critic: review a rendered diagram and return corrected structured IR."""

from __future__ import annotations

import json
from typing import Optional

import structlog
from pydantic import BaseModel, Field

from paperbanana.agents.base import BaseAgent
from paperbanana.core.diagram_ir import format_diagram_ir_for_regeneration
from paperbanana.core.types import CritiqueResult, DiagramIR
from paperbanana.core.utils import extract_json, load_image, truncate_text
from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()

MAX_CONTEXT_CHARS = 8_000


class IRCritiqueResult(BaseModel):
    """Result of an IR-focused review."""

    issues: list[str] = Field(default_factory=list)
    revised_ir: Optional[DiagramIR] = None

    @property
    def needs_revision(self) -> bool:
        return bool(self.revised_ir and self.issues)


class IRCriticAgent(BaseAgent):
    """Compare the rendered PNG with the IR and return a corrected IR.

    The critic is intentionally blind to styling: local Graphviz output has a
    plain theme, and asking a VLM for gradients/icons would cause endless
    revisions that the renderer cannot satisfy.
    """

    def __init__(
        self, vlm_provider: VLMProvider, prompt_dir: str = "prompts", prompt_recorder=None
    ):
        super().__init__(vlm_provider, prompt_dir, prompt_recorder=prompt_recorder)

    @property
    def agent_name(self) -> str:
        return "ir_critic"

    async def run(
        self,
        image_path: str,
        diagram_ir: DiagramIR,
        source_context: str,
        caption: str,
    ) -> IRCritiqueResult:
        image = load_image(image_path)
        template = self.load_prompt("diagram")
        prompt = self.format_prompt(
            template,
            prompt_label="ir_critic",
            caption=caption,
            diagram_ir=json.dumps(diagram_ir.model_dump(), ensure_ascii=False, indent=2),
            source_context=truncate_text(source_context, MAX_CONTEXT_CHARS),
        )

        json_ok = getattr(self.vlm, "supports_json_mode", True)
        response = await self.vlm.generate(
            prompt=prompt,
            images=[image],
            temperature=0.1,
            max_tokens=8192,
            response_format="json" if json_ok else None,
        )
        return self._parse(response, fallback_title=diagram_ir.title)

    @classmethod
    def _parse(cls, response: Optional[str], fallback_title: str) -> IRCritiqueResult:
        data = extract_json(response or "")
        if not isinstance(data, dict):
            logger.warning("IR critic returned invalid JSON")
            return IRCritiqueResult()

        issues = [str(s).strip() for s in (data.get("issues") or []) if str(s).strip()]
        raw_ir = data.get("revised_ir")
        revised: Optional[DiagramIR] = None
        if isinstance(raw_ir, dict):
            candidate = dict(raw_ir)
            candidate.setdefault("title", fallback_title or "Methodology Diagram")
            try:
                revised = DiagramIR.model_validate(candidate)
            except (ValueError, TypeError) as exc:
                logger.warning("IR critic returned invalid revised_ir", error=str(exc))
                issues = issues or [f"IR critic returned invalid revised_ir: {exc}"]

        if not data.get("needs_revision", bool(issues)):
            return IRCritiqueResult(issues=issues, revised_ir=None)
        return IRCritiqueResult(issues=issues, revised_ir=revised)

    @staticmethod
    def to_critique_result(result: IRCritiqueResult) -> CritiqueResult:
        """Adapt the IR review to the pipeline's existing critique record."""
        if result.revised_ir is None:
            if result.issues:
                logger.info(
                    "IR critic found issues but returned no revised IR; keeping current diagram",
                    issues=result.issues[:5],
                )
            return CritiqueResult()
        return CritiqueResult(
            critic_suggestions=result.issues,
            revised_description=format_diagram_ir_for_regeneration(result.revised_ir),
        )
