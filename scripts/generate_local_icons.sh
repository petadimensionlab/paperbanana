#!/usr/bin/env bash
# Regenerate the local infographic icon/panel assets with mflux.
# Requires: uv venv /tmp/mflux-venv && uv pip install --python /tmp/mflux-venv/bin/python mflux
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MFLUX="${MFLUX_BIN:-/tmp/mflux-venv/bin/mflux-generate-flux2}"
MODEL="${MFLUX_MODEL:-flux2-klein-4b}"
ICON_DIR="$ROOT/assets/local/icons"
PANEL_DIR="$ROOT/assets/local/panels"
mkdir -p "$ICON_DIR" "$PANEL_DIR"

icon() {
  local key="$1" prompt="$2"
  "$MFLUX" --model "$MODEL" --quantize 4 --steps 4 \
    --width 256 --height 256 --seed 42 \
    --prompt "flat vector icon of ${prompt}, simple line art, medium blue #4f7fd6, centered, white background, no text, no letters, minimal" \
    --output "$ICON_DIR/${key}.png" >/dev/null
}

panel() {
  local key="$1" prompt="$2"
  "$MFLUX" --model "$MODEL" --quantize 4 --steps 4 \
    --width 512 --height 256 --seed 42 \
    --prompt "${prompt}, soft pastel watercolor, lots of white space, no text, no letters, no words" \
    --output "$PANEL_DIR/${key}.png" >/dev/null
}

icon input "an inbox tray with an arrow entering"
icon output "an arrow leaving a document"
icon database "a database cylinder"
icon attention "a neural attention graph with connected nodes"
icon routing "branching router arrows"
icon encoder "stacked neural network encoder layers"
icon decoder "stacked generative decoder layers"
icon model "a transformer block grid"
icon training "a rising training curve"
icon loss "a descending loss curve"
icon memory "a computer memory chip"
icon probability "a probability bar chart"
icon token "text token blocks"

panel overview "abstract academic illustration of ideas and data flowing into a research framework"
panel encoder "abstract illustration of stacked neural network encoder layers"
panel decoder "abstract illustration of a generative decoder network"

echo "Generated icons in $ICON_DIR and panels in $PANEL_DIR"
