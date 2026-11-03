# PaperBanana Local (offline, vector-first)

This fork adds fully local rendering modes for methodology diagrams. The local
profiles do **not** call a diffusion image model: a local VLM produces a
`DiagramIR`, and Graphviz renders the labels and arrows deterministically.

## Requirements

- Python 3.10+
- Graphviz (`dot`) on PATH
- Ollama >= 0.40.13 (or another OpenAI-compatible local server)
- A vision-capable local model for quality mode (reference images and critic)

## Quick start

```bash
uv sync --extra dev
ollama pull qwen3.5:9b

VLM_PROVIDER=ollama \
OLLAMA_MODEL=qwen3.5:9b \
OLLAMA_JSON_MODE=true \
OLLAMA_REASONING_EFFORT=none \
.venv/bin/paperbanana generate \
  -i examples/sample_inputs/transformer_method.txt \
  -c "Overview of our encoder-decoder architecture" \
  --render-mode ir_fast \
  --vector-export both \
  --output-dir outputs
```

Outputs in `outputs/run_*/`:

- `final_output.png` — preview / Critic input
- `final_output.svg` — editable vector
- `final_output.pdf` — paper-ready vector
- `final_output.tex` — deterministic standalone TikZ (when `export_tikz: true`)
- `diagram_ir.json` — editable graph (nodes / edges / groups)
- `diagram.dot` — Graphviz source

TikZ compilation needs `pgf` and `standalone`:

```bash
tlmgr update --self
tlmgr install pgf standalone
pdflatex final_output.tex
```

## Profiles

| File | Purpose |
|---|---|
| `configs/local/speed.yaml` | one VLM call, 9B, fastest |
| `configs/local/quality.yaml` | full planning + critic, 35B |
| `configs/local/macmini64.yaml` | 64GB Mac mini, 35B Q4 / 32K |
| `configs/local/studio128-speed.yaml` | Tailnet Studio, 9B |
| `configs/local/studio128-quality.yaml` | Tailnet Studio, 122B |
| `configs/local/spark.yaml` | DGX Spark, Ollama 35B |
| `configs/local/spark-vllm.yaml` | DGX Spark, vLLM FP8 |

```bash
.venv/bin/paperbanana generate \
  -i method.txt -c "Overview" \
  --config configs/local/speed.yaml
```

## Render modes

- `diffusion` — original PaperBanana behavior (cloud/local image model)
- `ir_fast` — one VLM call -> IR -> Graphviz; no critic; 1 iteration
- `ir_quality` — full planning -> IR -> critic loop -> Graphviz

## Infographic theme (hybrid D)

The `infographic` theme renders a deterministic 16:9 SVG/PDF with the exact IR
labels, pastel panels, a title, icons, and an optional locally generated
background. It does not ask a diffusion model to draw text.

```bash
.venv/bin/paperbanana generate \
  --config configs/local/m1ultra-infographic.yaml \
  -i method.txt -c "Overview of the method"
```

Set `pipeline.infographic_background` to an image path (or
`INFOGRAPHIC_BACKGROUND`) to composite a local background. The repository ships
`assets/local/watercolor-bg.png`, generated offline with mflux/FLUX.2 klein 4B:

```bash
uv venv /tmp/mflux-venv
uv pip install --python /tmp/mflux-venv/bin/python mflux
/tmp/mflux-venv/bin/mflux-generate-flux2 \
  --model flux2-klein-4b --quantize 4 --steps 4 \
  --width 1024 --height 576 --seed 42 \
  --prompt "soft pastel abstract academic background, light blue mint and peach watercolor paper texture, very subtle geometric shapes, lots of white space, clean minimal, no text, no letters" \
  --output assets/local/watercolor-bg.png
```

The theme also composites locally generated per-node icons from
`assets/local/icons/` and per-panel watercolor washes from
`assets/local/panels/`. Regenerate both with:

```bash
./scripts/generate_local_icons.sh
```

To generate panel illustrations from the actual IR groups at run time (cached
under `~/.cache/paperbanana`), use:

```bash
MFLUX_BIN=/tmp/mflux-venv/bin/mflux-generate-flux2 \
.venv/bin/paperbanana generate \
  --config configs/local/m1ultra-infographic-diffusion.yaml \
  -i method.txt -c "Overview of the method"
```

The diffused panel art is blurred before compositing (diffusion can paint
pseudo-text); all real text remains in the SVG layer. If mflux is unavailable,
the renderer falls back to `assets/local/panels/`.

The theme supports Japanese labels; `rsvg-convert` is used to produce PNG/PDF.
Without `rsvg-convert`, the SVG is still written.

## img2img / ControlNet (experimental)

`infographic_img2img: true` renders a textless structure, runs local img2img
(FLUX.2 klein 4B), then overlays the exact IR text. This is available in
`configs/local/lmstudio-infographic-img2img-experimental.yaml`, but it is not
recommended for final figures yet: local img2img can shift cards, so the
overlaid text may no longer align. ControlNet would preserve geometry, but the
FLUX.1-dev ControlNet checkpoints are gated, and the Qwen-Image ControlNet
variant still shifts boxes slightly. A card-level calibration pass
(`paperbanana/local/calibrate.py`) matches each card independently and rejects
the style pass if any card is low-confidence. After calibration,
`paperbanana/local/mask_composite.py` keeps the deterministic card and panel
geometry, compositing the styled image underneath with configurable weights
(`infographic_img2img_card_weight`, `infographic_img2img_panel_weight`). Set
`card_weight: 1.0` to make cards completely flat (structure only) while keeping
panel/background texture; 0.72 keeps a little diffusion texture inside cards.
This keeps card size uniform and the SVG text aligned; if calibration fails, the
pipeline falls back to the vector-only render. Keep the default deterministic infographic + local panel/icon
assets for publication output.

## Routing between local/self-hosted providers

`vlm.provider: routing` selects a provider per agent role. This is useful when
one machine has the big vision model and another has a fast text model.

```yaml
vlm:
  provider: routing
  routing:
    default:
      provider: ollama
      base_url: http://100.127.45.60:11434/v1
      model: pb-quality
    structurer:
      provider: tensorfold
      base_url: http://100.127.45.60:8080/v1
      model: qwen3.6-35b-a3b
    fallback:
      provider: ollama
      base_url: http://127.0.0.1:11434/v1
      model: qwen3.5:9b
```

Supported self-hosted provider names:

| Provider | Default endpoint | Notes |
|---|---|---|
| `ollama` | `http://127.0.0.1:11434/v1` | OpenAI-compatible `/v1` |
| `lmstudio` | `http://127.0.0.1:1234/v1` | LM Studio server; no `json_object` mode |
| `omlx` | `http://127.0.0.1:8000/v1` | Apple Silicon MLX server |
| `tensorfold` | `http://127.0.0.1:8080/v1` | text-focused; use Ollama/oMLX for vision |
| `vllm` | `http://127.0.0.1:8000/v1` | DGX Spark / CUDA |
| `openai_local` | `http://127.0.0.1:8000/v1` | any OpenAI-compatible server |

Per-role options: `base_url`, `model`, `api_key`, `json_mode`,
`reasoning_effort`, `timeout`, `image_max_edge`, `image_format`, `extra_body`.
The fallback must be self-hosted; cloud providers are rejected as fallbacks.

### LM Studio for IR quality

LM Studio's server is OpenAI-compatible but rejects `response_format=json_object`
(it accepts `json_schema`/`text`). Keep `json_mode: false`; the prompts already
demand JSON-only output. Thinking must be disabled with `reasoning_effort: none`.

```yaml
vlm:
  provider: lmstudio
  model: qwen/qwen3.6-35b-a3b
  lmstudio_base_url: http://127.0.0.1:1234/v1
  lmstudio_json_mode: false
```

Use `configs/local/lmstudio-quality.yaml` to run the full IR-quality pipeline
against the installed 35B model. On this machine the loaded model was
`qwen/qwen3.6-35b-a3b` (Qwen3.5-family 35B), not a separate Qwen3.5 tag.

The IR visualizer flattens duplicate container nodes (for example an
`Encoder Stack` node inside an `Encoder Block` group) and reconnects the flow
to the internal components. The final critic revision is rendered once more
deterministically after the last iteration, so the last review is not lost.

Profiles:

- `configs/local/lmstudio-quality.yaml` — LM Studio 35B + infographic theme.
- `configs/local/tailnet-hybrid.yaml` — vision roles on Studio 122B, text/IR roles local, local 9B fallback.
- `configs/local/m1ultra-remote.yaml` — M1 Ultra Studio `gemma4:12b` via Ollama, fast IR, local 9B fallback.
- `configs/local/studio128-quality.yaml` — all roles on Studio 122B.
- `configs/local/spark-vllm.yaml` — DGX Spark vLLM endpoint.

## Ollama notes

- Set `OLLAMA_CONTEXT_LENGTH` (32K for 64GB Mac mini, 64K for 128GB) or create a
  model alias with `PARAMETER num_ctx`.
- Thinking models must run with `reasoning_effort=none`; otherwise `content` can
  be empty. The provider sends this by default.
- `OLLAMA_JSON_MODE=true` is required for reliable structured output.
- Vision inputs are downscaled to `OLLAMA_IMAGE_MAX_EDGE` (default 1280) and
  JPEG-encoded to cut vision tokens.

## Remote Tailnet

```dotenv
VLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://100.127.45.60:11434/v1
OLLAMA_MODEL=pb-quality
OLLAMA_JSON_MODE=true
OLLAMA_REASONING_EFFORT=none
```

Keep rendering on the client. Only prompts and reference images cross Tailnet.
Protect the Ollama port with Tailscale ACLs and `tailscale serve`.

See `docs/local/STATUS.md` for the current implementation checkpoint.
