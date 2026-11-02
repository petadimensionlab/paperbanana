"""IR visualizer: render DiagramIR to deterministic vector/raster output."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import structlog

from paperbanana.agents.base import BaseAgent
from paperbanana.agents.structurer import StructurerAgent
from paperbanana.agents.visualizer import VisualizerAgent
from paperbanana.core.types import DiagramIR, DiagramType
from paperbanana.core.utils import ensure_dir, save_json
from paperbanana.providers.base import ImageGenProvider, VLMProvider
from paperbanana.providers.image_gen.dummy import DummyImageGen
from paperbanana.vector.graphviz_render import (
    diagram_ir_to_dot,
    find_dot_executable,
    render_dot_to_file,
)
from paperbanana.vector.infographic_svg import write_infographic

logger = structlog.get_logger()


class IRVisualizerAgent(BaseAgent):
    """Turn a description (or a precomputed IR) into a Graphviz-rendered figure.

    The agent is API-compatible with ``VisualizerAgent.run`` so the existing
    Visualizer<->Critic loop can use it unchanged.  Statistical plots are
    delegated to the normal matplotlib-backed visualizer.
    """

    def __init__(
        self,
        vlm_provider: VLMProvider,
        prompt_dir: str = "prompts",
        output_dir: str = "outputs",
        prompt_recorder=None,
        dpi: int = 200,
        image_gen: Optional[ImageGenProvider] = None,
        plot_vlm: Optional[VLMProvider] = None,
        theme: str = "graphviz",
        background_image: Optional[str] = None,
        icon_dir: Optional[str] = None,
        panel_dir: Optional[str] = None,
        show_edge_labels: bool = False,
    ):
        super().__init__(vlm_provider, prompt_dir, prompt_recorder=prompt_recorder)
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.theme = theme
        self.background_image = background_image
        self.icon_dir = icon_dir
        self.panel_dir = panel_dir
        self.show_edge_labels = show_edge_labels
        self.structurer = StructurerAgent(
            vlm_provider, prompt_dir=prompt_dir, prompt_recorder=prompt_recorder
        )
        self._plot_visualizer = VisualizerAgent(
            image_gen or DummyImageGen(),
            plot_vlm or vlm_provider,
            prompt_dir=prompt_dir,
            output_dir=str(self.output_dir),
            prompt_recorder=prompt_recorder,
        )
        self._last_ir: Optional[DiagramIR] = None
        self._pending_ir: Optional[DiagramIR] = None
        self._last_vector_paths: dict[str, str] = {}
        self._source_context = ""
        self._caption = ""

    @property
    def agent_name(self) -> str:
        return "ir_visualizer"

    def set_output_dir(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self._plot_visualizer.set_output_dir(output_dir)

    def set_context(self, source_context: str, caption: str) -> None:
        """Provide context required by the Structurer prompt."""
        self._source_context = source_context
        self._caption = caption

    def set_pending_ir(self, diagram_ir: Optional[DiagramIR]) -> None:
        """Use a precomputed IR for the next run (speed mode)."""
        self._pending_ir = diagram_ir

    @property
    def last_ir(self) -> Optional[DiagramIR]:
        return self._last_ir

    async def run(
        self,
        description: str,
        diagram_type: DiagramType = DiagramType.METHODOLOGY,
        raw_data: Optional[dict] = None,
        output_path: Optional[str] = None,
        iteration: int = 0,
        seed: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
        vector_formats: Optional[list[str]] = None,
        sketch_guided: bool = False,
    ) -> str:
        self._last_vector_paths = {}

        if diagram_type == DiagramType.STATISTICAL_PLOT:
            path = await self._plot_visualizer.run(
                description=description,
                diagram_type=diagram_type,
                raw_data=raw_data,
                output_path=output_path,
                iteration=iteration,
                seed=seed,
                aspect_ratio=aspect_ratio,
                vector_formats=vector_formats,
                sketch_guided=sketch_guided,
            )
            self._last_vector_paths = dict(self._plot_visualizer._last_vector_paths)
            return path

        if self.theme != "infographic" and not find_dot_executable():
            raise RuntimeError(
                "Graphviz `dot` is required for render_mode=ir_*. "
                "Install it (macOS: brew install graphviz; Debian: apt install graphviz)."
            )

        ir = self._pending_ir
        self._pending_ir = None
        if ir is None:
            ir = await self.structurer.run(
                description=description,
                source_context=self._source_context,
                caption=self._caption,
            )
        self._last_ir = ir

        iter_dir = ensure_dir(self.output_dir / f"iter_{iteration}")
        save_json(ir.model_dump(), iter_dir / "diagram_ir.json")

        png_path = output_path or str(self.output_dir / f"diagram_iter_{iteration}.png")

        if self.theme == "infographic":
            svg_path = write_infographic(
                ir,
                png_path,
                background_image=self.background_image,
                icon_dir=self.icon_dir,
                panel_dir=self.panel_dir,
                show_edge_labels=self.show_edge_labels,
                width=max(1600, self.dpi * 16),
            )
            self._last_vector_paths["svg"] = str(svg_path)
            pdf_path = svg_path.with_suffix(".pdf")
            if pdf_path.exists():
                self._last_vector_paths["pdf"] = str(pdf_path)
        else:
            dot_source = diagram_ir_to_dot(ir)
            (iter_dir / "diagram.dot").write_text(dot_source, encoding="utf-8")
            if not render_dot_to_file(dot_source, png_path, "png", dpi=self.dpi):
                raise RuntimeError(f"Graphviz failed to render {png_path}")

            # Always emit an SVG next to the raster: the local profiles are
            # vector-first, and the Critic can read the PNG.
            svg_path = str(Path(png_path).with_suffix(".svg"))
            if render_dot_to_file(dot_source, svg_path, "svg", dpi=self.dpi):
                self._last_vector_paths["svg"] = svg_path

            for fmt in vector_formats or []:
                if fmt == "svg":
                    continue
                candidate = str(Path(png_path).with_suffix(f".{fmt}"))
                if render_dot_to_file(dot_source, candidate, fmt, dpi=self.dpi):
                    self._last_vector_paths[fmt] = candidate

        logger.info(
            "IR diagram rendered",
            iteration=iteration,
            nodes=len(ir.nodes),
            edges=len(ir.edges),
            png=png_path,
            vectors=self._last_vector_paths,
        )
        return png_path
