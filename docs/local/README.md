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
