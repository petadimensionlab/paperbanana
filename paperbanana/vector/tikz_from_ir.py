"""Render DiagramIR to standalone TikZ using Graphviz's layout coordinates.

This is deliberately not an LLM-to-TikZ conversion: labels and edges come from
the IR, while Graphviz only supplies node positions and sizes.  The result is
deterministic, editable, and safe for publication figures.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import structlog

from paperbanana.core.types import DiagramIR
from paperbanana.vector.graphviz_render import (
    _build_dot_id_map,
    diagram_ir_to_dot,
    find_dot_executable,
)

logger = structlog.get_logger()

_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _escape_latex(text: str) -> str:
    return "".join(_LATEX_ESCAPES.get(ch, ch) for ch in text)


def _has_tex_package(package: str) -> bool:
    kpsewhich = shutil.which("kpsewhich")
    if not kpsewhich:
        return False
    result = subprocess.run(
        [kpsewhich, package],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return bool(result.stdout.strip())


def _dot_plain(dot_source: str) -> str:
    dot_bin = find_dot_executable()
    if not dot_bin:
        raise RuntimeError("Graphviz `dot` is required for TikZ export")
    result = subprocess.run(
        [dot_bin, "-Tplain"],
        input=dot_source,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"dot -Tplain failed: {(result.stderr or '')[:300]}")
    return result.stdout


def render_ir_to_tikz(
    diagram_ir: DiagramIR,
    *,
    font_package: Optional[str] = None,
    node_fill: str = "black!3",
    group_fill: str = "blue!4",
) -> str:
    """Return a standalone TikZ document for *diagram_ir*."""
    dot_source = diagram_ir_to_dot(diagram_ir)
    plain = _dot_plain(dot_source)

    id_map = _build_dot_id_map(diagram_ir)
    positions: dict[str, tuple[float, float, float, float]] = {}

    for raw_line in plain.splitlines():
        if not raw_line.startswith("node "):
            continue
        parts = shlex.split(raw_line)
        # node name x y width height label style shape color fillcolor
        if len(parts) < 7:
            continue
        name = parts[1]
        try:
            x, y, width, height = (float(v) for v in parts[2:6])
        except ValueError:
            continue
        positions[name] = (x, y, width, height)

    if _has_tex_package("standalone.cls"):
        document_class = r"\documentclass[border=5pt]{standalone}"
    else:
        # Minimal TeX Live installs may lack standalone.cls; still emit a
        # compilable document rather than failing the export.
        document_class = r"\documentclass{article}"
    lines: list[str] = [
        document_class,
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{tikz}",
        r"\usetikzlibrary{arrows.meta,positioning,fit,backgrounds}",
    ]
    if font_package:
        lines.append(rf"\usepackage{{{font_package}}}")
    lines.append(r"\pagestyle{empty}")
    lines.extend(
        [
            r"\begin{document}",
            r"\begin{tikzpicture}[x=1in,y=1in,>=Stealth,",
            r"  every node/.style={font=\small, align=center, inner sep=2pt},",
            r"  box/.style={draw=black!60, rounded corners=2pt, fill=" + node_fill + "},",
            r"  group/.style={draw=black!35, dashed, rounded corners=4pt, fill="
            + group_fill
            + r", inner sep=8pt},",
            r"]",
        ]
    )

    for node in diagram_ir.nodes:
        dot_name = id_map.get(node.id)
        if not dot_name or dot_name not in positions:
            continue
        x, y, width, height = positions[dot_name]
        shape_style = "box" if (node.shape or "rounded") == "rounded" else "box, sharp corners"
        lines.append(
            rf"  \node[{shape_style}, minimum width={width:.3f}in, "
            rf"minimum height={height:.3f}in] ({dot_name}) at ({x:.3f},{y:.3f}) "
            rf"{{{_escape_latex(node.label)}}};"
        )

    # Group containers are drawn behind the nodes (fit needs the node names).
    for group in diagram_ir.groups:
        members = [id_map[nid] for nid in group.node_ids if nid in id_map]
        if not members:
            continue
        fit_expr = "".join(f"({m})" for m in members)
        lines.append(
            rf"  \begin{{scope}}[on background layer]"
            rf"\node[group, fit={fit_expr}, label={{[font=\small\bfseries]above:"
            rf"{_escape_latex(group.label)}}}] {{}};"
            rf"\end{{scope}}"
        )

    edge_labels: dict[tuple[str, str], str] = {}
    for edge in diagram_ir.edges:
        if edge.label:
            edge_labels[(edge.source, edge.target)] = edge.label
    for edge in diagram_ir.edges:
        src = id_map.get(edge.source)
        dst = id_map.get(edge.target)
        if not src or not dst or src not in positions or dst not in positions:
            continue
        label = edge_labels.get((edge.source, edge.target))
        if label:
            lines.append(
                rf"  \draw[->] ({src}) -- node[midway, above, font=\scriptsize] "
                rf"{{{_escape_latex(label)}}} ({dst});"
            )
        else:
            lines.append(rf"  \draw[->] ({src}) -- ({dst});")

    lines.extend([r"\end{tikzpicture}", r"\end{document}", ""])
    return "\n".join(lines)


def write_tikz_from_ir(
    diagram_ir: DiagramIR,
    output_path: str | Path,
    **kwargs,
) -> Path:
    """Write a standalone `.tex` file for *diagram_ir*."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_ir_to_tikz(diagram_ir, **kwargs), encoding="utf-8")
    return path


def compile_tikz(tex_path: str | Path, output_dir: Optional[str | Path] = None) -> Path:
    """Compile a `.tex` file with pdflatex (best effort, raises on failure)."""
    tex = Path(tex_path)
    out_dir = Path(output_dir) if output_dir else tex.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={out_dir}",
            str(tex),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdflatex failed: {(result.stdout or '')[-500:]}")
    return out_dir / f"{tex.stem}.pdf"
