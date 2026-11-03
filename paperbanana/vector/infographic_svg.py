"""Rich, deterministic infographic renderer for DiagramIR.

The renderer keeps every label and edge from the IR, but uses a designed
16:9 canvas instead of Graphviz's default look: pastel panels, rounded cards,
a title, subtle shadows, and optional icons.  It is fully offline and emits
SVG, PNG, and PDF.
"""

from __future__ import annotations

import base64
import functools
import html
import math
import re
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import structlog

from paperbanana.core.types import DiagramIR, DiagramIRNode

logger = structlog.get_logger()

CANVAS_W = 1600
CANVAS_H = 900
MARGIN_X = 48
TITLE_H = 92
COL_GAP = 28
NODE_H = 70
NODE_GAP = 18
PANEL_PAD = 18
PANEL_LABEL_H = 34
FONT = "'Hiragino Sans', 'Noto Sans JP', 'Helvetica Neue', Arial, sans-serif"

PALETTE = [
    ("#e8f1ff", "#7aa7e8", "#1e3a5f"),
    ("#eaf8ef", "#76c39a", "#1f4d36"),
    ("#fff3e6", "#e8a76b", "#5a351c"),
    ("#f3ecff", "#a98ae0", "#3f2b63"),
    ("#fff9dd", "#d9bf5a", "#4d4215"),
    ("#ffecef", "#e58a9a", "#5c2630"),
]

_ICONS = {
    "input": "M4 7h10M4 12h10M4 17h6M17 7l3 5-3 5",
    "output": "M3 12h12M12 7l5 5-5 5M18 4v16",
    "database": (
        "M4 6c0-2 3-3 8-3s8 1 8 3-3 3-8 3-8-1-8-3z"
        "M4 6v6c0 2 3 3 8 3s8-1 8-3V6"
        "M4 12v6c0 2 3 3 8 3s8-1 8-3v-6"
    ),
    "attention": "M12 3v18M3 12h18M5.6 5.6l12.8 12.8M18.4 5.6L5.6 18.4",
    "routing": "M4 12h5M15 12h5M9 8l3 4-3 4M15 8l-3 4 3 4",
    "encoder": "M4 5h16v14H4zM8 9h8M8 13h5M8 17h3",
    "decoder": "M4 5h16v14H4zM16 9H8M16 13h-5M16 17h-3",
    "model": "M4 7h16M4 12h16M4 17h16M8 4v16M16 4v16",
    "training": "M4 18l5-6 4 3 6-8M4 20h16",
    "loss": "M4 18l5-8 4 4 6-10M4 20h16",
    "memory": "M6 4h12v16H6zM9 8h6M9 12h6M9 16h3",
    "output token": "M4 12h12M12 7l5 5-5 5M18 4v16",
    "input token": "M4 7h10M4 12h10M4 17h6M17 7l3 5-3 5",
}


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def _pretty_label(text: str) -> str:
    """Make common LaTeX fragments readable in SVG text."""
    replacements = {
        r"\sqrt": "√",
        r"\times": "×",
        r"\rightarrow": "→",
        r"\to": "→",
        r"\le": "≤",
        r"\ge": "≥",
        r"\cdot": "·",
        r"\alpha": "α",
        r"\beta": "β",
        r"\gamma": "γ",
        r"\lambda": "λ",
        r"\theta": "θ",
        r"\sum": "Σ",
    }
    result = text
    for source, target in replacements.items():
        result = result.replace(source, target)
    result = result.replace("$", "")
    result = re.sub(r"\\[a-zA-Z]+", "", result)
    return result.replace("\\", "").replace("{", "").replace("}", "").strip()


def _icon_key(label: str) -> str:
    low = _pretty_label(label).lower()
    # Order matters: more specific phrases first.
    ordered = (
        ("input token", "input"),
        ("output token", "output"),
        ("probability", "probability"),
        ("softmax", "probability"),
        ("database", "database"),
        ("memory", "memory"),
        ("attention", "attention"),
        ("routing", "routing"),
        ("router", "routing"),
        ("encoder", "encoder"),
        ("decoder", "decoder"),
        ("feed-forward", "model"),
        ("feed forward", "model"),
        ("transformer", "model"),
        ("training", "training"),
        ("loss", "loss"),
        ("token", "token"),
        ("input", "input"),
        ("output", "output"),
    )
    for phrase, key in ordered:
        if phrase in low:
            return key
    return "model"


def _icon_for(label: str) -> str:
    return _ICONS.get(_icon_key(label), "M5 5h14v14H5zM8 12h8")


def _panel_key(label: str) -> str:
    low = label.lower()
    if "encoder" in low:
        return "encoder"
    if "decoder" in low:
        return "decoder"
    return "overview"


@functools.lru_cache(maxsize=64)
def _image_data_uri(path: str) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _wrap(label: str, width_chars: int, max_lines: int = 2) -> list[str]:
    if len(label) <= width_chars:
        return [label]
    words = label.split()
    if len(words) > 1:
        lines = textwrap.wrap(label, width=max(8, width_chars), break_long_words=False)
    else:
        # Japanese / long unbroken labels: wrap by characters.
        lines = [label[i : i + width_chars] for i in range(0, len(label), width_chars)]
    return lines[:max_lines]


@dataclass
class _NodeBox:
    node: DiagramIRNode
    x: float
    y: float
    w: float
    h: float


@dataclass
class _Panel:
    label: str
    nodes: list[DiagramIRNode]
    color_idx: int
    x: float = 0
    y: float = 0
    w: float = 0
    h: float = 0


def _build_panels(ir: DiagramIR) -> list[_Panel]:
    node_by_id = {n.id: n for n in ir.nodes}
    grouped_ids: set[str] = set()
    panels: list[_Panel] = []
    for idx, group in enumerate(ir.groups):
        members = [node_by_id[nid] for nid in group.node_ids if nid in node_by_id]
        if members:
            panels.append(_Panel(group.label or group.id, members, idx))
            grouped_ids.update(n.id for n in members)
    ungrouped = [n for n in ir.nodes if n.id not in grouped_ids]
    if ungrouped and panels:
        # Attach an ungrouped input/output node to the group it feeds or
        # receives from, so a one-node "Overview" panel is avoided.
        node_to_panel = {n.id: p_idx for p_idx, panel in enumerate(panels) for n in panel.nodes}
        remaining: list[DiagramIRNode] = []
        for node in ungrouped:
            neighbours = [e.target for e in ir.edges if e.source == node.id] + [
                e.source for e in ir.edges if e.target == node.id
            ]
            attached = False
            for neighbour in neighbours:
                p_idx = node_to_panel.get(neighbour)
                if p_idx is not None:
                    panels[p_idx].nodes.insert(0, node)
                    node_to_panel[node.id] = p_idx
                    attached = True
                    break
            if not attached:
                remaining.append(node)
        ungrouped = remaining
    if ungrouped:
        # Put ungrouped nodes in a leading "Overview" panel.
        panels.insert(0, _Panel("Overview", ungrouped, len(panels)))
    if not panels:
        panels = [_Panel("Method", list(ir.nodes), 0)]
    return panels


def _layout(ir: DiagramIR) -> tuple[list[_Panel], dict[str, _NodeBox], list[tuple[str, str, str]]]:
    panels = _build_panels(ir)
    panel_count = len(panels)
    # Panels are arranged in a grid: at most 3 columns on a 16:9 canvas.
    cols = min(3, panel_count)
    rows = math.ceil(panel_count / cols)
    usable_w = CANVAS_W - 2 * MARGIN_X - (cols - 1) * COL_GAP
    usable_h = CANVAS_H - TITLE_H - 24 - (rows - 1) * COL_GAP
    panel_w = usable_w / cols
    panel_h = usable_h / rows

    boxes: dict[str, _NodeBox] = {}
    edges: list[tuple[str, str, str]] = [(e.source, e.target, e.label or "") for e in ir.edges]

    for p_idx, panel in enumerate(panels):
        col = p_idx % cols
        row = p_idx // cols
        px = MARGIN_X + col * (panel_w + COL_GAP)
        py = TITLE_H + row * (panel_h + COL_GAP)
        panel.x, panel.y, panel.w, panel.h = px, py, panel_w, panel_h

        inner_x = px + PANEL_PAD
        inner_y = py + PANEL_LABEL_H + PANEL_PAD
        inner_w = panel_w - 2 * PANEL_PAD
        inner_h = panel_h - PANEL_LABEL_H - 2 * PANEL_PAD

        n = len(panel.nodes)
        if n == 0:
            continue
        # One column for <=5 nodes, two sub-columns for larger panels.
        sub_cols = 1 if n <= 5 else 2
        sub_rows = math.ceil(n / sub_cols)
        node_w = (inner_w - (sub_cols - 1) * 12) / sub_cols
        node_h = min(NODE_H, (inner_h - (sub_rows - 1) * NODE_GAP) / max(1, sub_rows))
        node_h = max(46, node_h)
        content_h = sub_rows * node_h + (sub_rows - 1) * NODE_GAP
        start_y = inner_y + max(0, (inner_h - content_h) / 2)
        for i, node in enumerate(panel.nodes):
            sub_col = i // sub_rows
            sub_row = i % sub_rows
            nx = inner_x + sub_col * (node_w + 12)
            ny = start_y + sub_row * (node_h + NODE_GAP)
            boxes[node.id] = _NodeBox(node, nx, ny, node_w, node_h)

    return panels, boxes, edges


def _node_svg(
    box: _NodeBox,
    color: tuple[str, str, str],
    icon_dir: Optional[str] = None,
    *,
    draw_card: bool = True,
    draw_text: bool = True,
    draw_icon: bool = True,
) -> str:
    node = box.node
    _fill, stroke, text_color = color
    show_icon = draw_icon and box.w >= 170
    icon_w = 44 if show_icon else 0
    text_cx = box.x + icon_w + (box.w - icon_w) / 2
    width_chars = max(8, int((box.w - icon_w - 14) / 8.4))
    lines = _wrap(_pretty_label(node.label), width_chars, max_lines=2)
    line_h = 16 if len(lines) > 1 else 18
    font_size = 14 if len(lines) > 1 else 16
    text_y = box.y + box.h / 2 - (len(lines) - 1) * line_h / 2 + 5
    tspans = "".join(
        f'<tspan x="{text_cx:.1f}" y="{text_y + i * line_h:.1f}">{_esc(line)}</tspan>'
        for i, line in enumerate(lines)
    )
    icon = _icon_for(node.label)
    icon_x = box.x + 8
    icon_y = box.y + box.h / 2 - 18
    icon_svg = ""
    icon_file = None
    if icon_dir:
        candidate = Path(icon_dir) / f"{_icon_key(node.label)}.png"
        if candidate.exists():
            icon_file = candidate
    if show_icon and icon_file is not None:
        icon_svg = (
            f'<circle cx="{icon_x + 18:.1f}" cy="{box.y + box.h / 2:.1f}" r="20" '
            f'fill="{_fill}" opacity="0.9"/>'
            f'<image x="{icon_x:.1f}" y="{icon_y:.1f}" width="36" height="36" '
            f'href="{_image_data_uri(str(icon_file))}" opacity="0.95"/>'
        )
    elif show_icon:
        icon_svg = f"""
      <g transform="translate({icon_x:.1f},{icon_y:.1f})" fill="none" stroke="{stroke}"
         stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" opacity="0.85">
        <path d="{icon}" transform="scale(1.1)"/>
      </g>"""
    card_svg = ""
    if draw_card:
        card_svg = f"""
      <rect x="{box.x:.1f}" y="{box.y:.1f}" width="{box.w:.1f}" height="{box.h:.1f}"
            rx="12" fill="#ffffff" fill-opacity="0.94" stroke="{stroke}" stroke-width="1.4"
            filter="url(#cardShadow)"/>
      <rect x="{box.x:.1f}" y="{box.y:.1f}" width="5" height="{box.h:.1f}"
            rx="2.5" fill="{stroke}"/>"""
    text_svg = ""
    if draw_text:
        text_svg = f"""
      <text x="{text_cx:.1f}" y="{text_y:.1f}" text-anchor="middle"
            font-family="{FONT}" font-size="{font_size}" font-weight="600"
            fill="{text_color}">{tspans}</text>"""
    return f"""
    <g class="node">{card_svg}
      {icon_svg}{text_svg}
    </g>"""


def _edge_svg(
    src: _NodeBox,
    dst: _NodeBox,
    label: str,
) -> str:
    sx = src.x + src.w / 2
    sy = src.y + src.h / 2
    tx = dst.x + dst.w / 2
    ty = dst.y + dst.h / 2
    if abs(tx - sx) > abs(ty - sy):
        if tx >= sx:
            x1, y1 = src.x + src.w, sy
            x2, y2 = dst.x, ty
        else:
            x1, y1 = src.x, sy
            x2, y2 = dst.x + dst.w, ty
        c1x, c1y = x1 + (x2 - x1) * 0.35, y1
        c2x, c2y = x2 - (x2 - x1) * 0.35, y2
    else:
        if ty >= sy:
            x1, y1 = sx, src.y + src.h
            x2, y2 = tx, dst.y
        else:
            x1, y1 = sx, src.y
            x2, y2 = tx, dst.y + dst.h
        c1x, c1y = x1, y1 + (y2 - y1) * 0.35
        c2x, c2y = x2, y2 - (y2 - y1) * 0.35
    path = f"M{x1:.1f},{y1:.1f} C{c1x:.1f},{c1y:.1f} {c2x:.1f},{c2y:.1f} {x2:.1f},{y2:.1f}"
    label_svg = ""
    if label:
        label = _pretty_label(label)
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2 - 8
        label_w = len(label) * 7.4 + 12
        label_x = mx - label_w / 2
        label_svg = f"""
      <rect x="{label_x:.1f}" y="{my - 12:.1f}" width="{label_w:.1f}"
            height="18" rx="6" fill="#ffffff" fill-opacity="0.9"/>
      <text x="{mx:.1f}" y="{my + 1:.1f}" text-anchor="middle" font-family="{FONT}"
            font-size="11" fill="#475569">{_esc(label)}</text>"""
    return f"""
    <path d="{path}" fill="none" stroke="#64748b" stroke-width="1.8"
          marker-end="url(#arrow)" opacity="0.9"/>{label_svg}"""


def render_infographic_svg(
    ir: DiagramIR,
    *,
    background_image: Optional[str] = None,
    icon_dir: Optional[str] = None,
    panel_dir: Optional[str] = None,
    panel_images: Optional[dict[str, str]] = None,
    show_edge_labels: bool = False,
    mode: str = "full",
    background_opacity: float = 0.20,
    draw_edges: Optional[bool] = None,
) -> str:
    """Return a rich 16:9 SVG for *ir*.

    mode:
      - full: background, panels, cards, text, edges
      - structure: panels/cards/edges only (for img2img/ControlNet)
      - overlay: text/icons over an existing styled image
    """
    overlay = mode == "overlay"
    structure = mode == "structure"
    edges_visible = (not overlay) if draw_edges is None else draw_edges
    panels, boxes, edges = _layout(ir)
    panel_svg: list[str] = []
    clip_defs: list[str] = []
    for p_idx, panel in enumerate(panels):
        fill, stroke, text_color = PALETTE[panel.color_idx % len(PALETTE)]
        clip_id = f"panelClip{p_idx}"
        clip_defs.append(
            f'<clipPath id="{clip_id}"><rect x="{panel.x:.1f}" y="{panel.y:.1f}" '
            f'width="{panel.w:.1f}" height="{panel.h:.1f}" rx="18"/></clipPath>'
        )
        panel_image = ""
        panel_file: Optional[Path] = None
        if panel_images:
            for key in (panel.label, _panel_key(panel.label)):
                if key in panel_images and Path(panel_images[key]).exists():
                    panel_file = Path(panel_images[key])
                    break
        if panel_file is None and panel_dir:
            candidate = Path(panel_dir) / f"{_panel_key(panel.label)}.png"
            if candidate.exists():
                panel_file = candidate
        if panel_file is not None:
            panel_image = (
                f'<image x="{panel.x:.1f}" y="{panel.y:.1f}" width="{panel.w:.1f}" '
                f'height="{panel.h:.1f}" preserveAspectRatio="xMidYMid slice" '
                f'clip-path="url(#{clip_id})" opacity="0.22" '
                f'href="{_image_data_uri(str(panel_file))}"/>'
            )
        panel_rect = ""
        if overlay:
            # Cover diffusion-generated pseudo-text in the panel header.
            panel_rect = (
                f'<rect x="{panel.x:.1f}" y="{panel.y:.1f}" width="{panel.w:.1f}" '
                f'height="{PANEL_LABEL_H + 30:.1f}" rx="12" fill="{fill}" '
                'opacity="0.96"/>'
            )
        elif not overlay:
            panel_rect = (
                f'<rect x="{panel.x:.1f}" y="{panel.y:.1f}" width="{panel.w:.1f}" '
                f'height="{panel.h:.1f}" rx="18" fill="{fill}" stroke="{stroke}" '
                'stroke-opacity="0.55" stroke-width="1.2"/>'
                + (panel_image if not structure else "")
            )
        panel_label = ""
        if not structure:
            panel_label = (
                f'<text x="{panel.x + panel.w / 2:.1f}" y="{panel.y + 23:.1f}" '
                f'text-anchor="middle" font-family="{FONT}" font-size="17" '
                f'font-weight="700" fill="{text_color}">{_esc(panel.label)}</text>'
            )
        panel_svg.append(
            f"""
    <g class="panel">{panel_rect}{panel_label}
    </g>"""
        )

    node_svg = [
        _node_svg(
            box,
            PALETTE[i % len(PALETTE)],
            icon_dir=icon_dir,
            draw_card=not overlay,
            draw_text=not structure,
            draw_icon=not structure,
        )
        for i, box in enumerate(boxes.values())
    ]
    edge_svg = []
    if edges_visible:
        edge_svg = [
            _edge_svg(boxes[src], boxes[dst], label if show_edge_labels else "")
            for src, dst, label in edges
            if src in boxes and dst in boxes
        ]

    bg = ""
    if background_image:
        import base64
        from pathlib import Path as _Path

        bg_path = _Path(background_image)
        if bg_path.exists():
            encoded = base64.b64encode(bg_path.read_bytes()).decode("ascii")
            opacity = 1.0 if overlay else background_opacity
            bg = (
                f'<image href="data:image/png;base64,{encoded}" x="0" y="0" '
                f'width="{CANVAS_W}" height="{CANVAS_H}" '
                f'preserveAspectRatio="xMidYMid slice" opacity="{opacity}"/>'
            )
        else:
            logger.warning("Infographic background not found", path=background_image)

    bg_rects = ""
    if not overlay:
        bg_rects = (
            '<rect width="100%" height="100%" fill="url(#bg)"/>'
            '<rect width="100%" height="100%" fill="url(#dots)"/>'
        )
    title_svg = ""
    if not structure:
        title_svg = (
            f'<text x="{CANVAS_W / 2}" y="52" text-anchor="middle" '
            f'font-family="{FONT}" font-size="32" font-weight="800" '
            f'fill="#0f172a">{_esc(ir.title)}</text>'
            f'<line x1="{CANVAS_W / 2 - 36}" y1="66" x2="{CANVAS_W / 2 + 36}" '
            'y2="66" stroke="#94a3b8" stroke-width="2" stroke-linecap="round"/>'
        )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}"
     viewBox="0 0 {CANVAS_W} {CANVAS_H}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#fbfdff"/>
      <stop offset="1" stop-color="#eef4fa"/>
    </linearGradient>
    <pattern id="dots" width="22" height="22" patternUnits="userSpaceOnUse">
      <circle cx="1.5" cy="1.5" r="1.1" fill="#cbd5e1" opacity="0.35"/>
    </pattern>
    <filter id="cardShadow" x="-10%" y="-20%" width="130%" height="150%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.10"/>
    </filter>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#64748b"/>
    </marker>
    {"".join(clip_defs)}
  </defs>
  {bg_rects}
  {bg}
  {title_svg}
  {"".join(panel_svg)}
  {"".join(edge_svg)}
  {"".join(node_svg)}
</svg>
"""


def write_infographic(
    ir: DiagramIR,
    output_path: str | Path,
    *,
    background_image: Optional[str] = None,
    icon_dir: Optional[str] = None,
    panel_dir: Optional[str] = None,
    panel_images: Optional[dict[str, str]] = None,
    show_edge_labels: bool = False,
    mode: str = "full",
    background_opacity: float = 0.20,
    draw_edges: Optional[bool] = None,
    width: int = 3200,
) -> Path:
    """Write SVG/PNG/PDF next to *output_path* (base suffix is replaced)."""
    base = Path(output_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    svg_path = base.with_suffix(".svg")
    svg_path.write_text(
        render_infographic_svg(
            ir,
            background_image=background_image,
            icon_dir=icon_dir,
            panel_dir=panel_dir,
            panel_images=panel_images,
            show_edge_labels=show_edge_labels,
            mode=mode,
            background_opacity=background_opacity,
            draw_edges=draw_edges,
        ),
        encoding="utf-8",
    )

    rsvg = shutil.which("rsvg-convert")
    if rsvg:
        subprocess.run(
            [rsvg, "-w", str(width), "-o", str(base.with_suffix(".png")), str(svg_path)],
            check=False,
            timeout=120,
        )
        subprocess.run(
            [rsvg, "-f", "pdf", "-o", str(base.with_suffix(".pdf")), str(svg_path)],
            check=False,
            timeout=120,
        )
    else:
        logger.warning("rsvg-convert not found; only SVG was written", path=str(svg_path))
    return svg_path
