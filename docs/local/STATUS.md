# PaperBanana Local — implementation status

This file is the resume point. Read it first after an interrupted session.

- Repository: `petadimensionlab/paperbanana` (public fork; repo name unchanged)
- Branch: `local` (upstream `origin/main` is at `87e01fd`, v0.3.0 + 2 commits)
- Last updated: 2026-11-02
- Checkpoint commit: `2938f2f` (Phase 2 + 3); previous: `d5e3ff4` (Phase 1)
- Scope: fully offline / local-only operation, label-accurate vector diagrams
- Non-goal: diffusion image generation in the speed/quality profiles

## Decisions

| Topic | Decision |
|---|---|
| Offline requirement | Fully offline; no cloud fallback |
| Methodology output | DiagramIR -> Graphviz SVG/PDF (TikZ later) |
| Speed profile | one VLM call (`FastIRPlanner`) + deterministic render |
| Quality profile | full planning + IR structurer + critic loop |
| 64GB target | Mac mini 64GB (unified memory), not NVIDIA |
| DGX target | DGX Spark GB10 128GB only |
| Remote server | `shinjimac-studio` 128GB, Ollama (9B speed / 122B quality) |
| Fork | keep `paperbanana` name, public under `petadimensionlab` |

## Phase status

| Phase | Status | Commit / notes |
|---|---|---|
| 0. Update to latest upstream | DONE | fast-forwarded to `87e01fd`, branch `local` |
| 1. Ollama hardening | DONE | reasoning_effort, timeout, image downscale, empty-content guard |
| 2a. Graphviz PNG + IR visualizer | DONE | `render_mode=ir_fast` works end-to-end |
| 2b. FastIR planner | DONE | one-call text-only IR planner |
| 2c. Speed profile docs/configs | DONE | `configs/local/speed.yaml` |
| 3. Quality IR critic | DONE | `ir_quality` uses `IRCriticAgent` + `revised_ir` |
| 3b. TikZ from IR | DONE | `dot -Tplain` -> standalone TikZ, `pdflatex` verified |
| 4. Routing (Tailnet/oMLX/TensorFold) | TODO | per-role provider selection is next |
| 5. Fork/public release | TODO | `gh repo fork`, CI, release tag |
| 6. Benchmarks | TODO | Mac mini 64 / M3 Max 128 / Studio / Spark |

## Verified locally

- `.venv/bin/python -m pytest -q` -> 902+ passed, 11 skipped (before new profile tests).
- Ollama `qwen3.5:9b` with `reasoning_effort=none`, JSON mode, one reference image:
  `{"has_arrows":true,"approx_nodes":12}` (image downscaled to 768px, 349 vision tokens).
- `paperbanana generate --render-mode ir_fast` on `examples/sample_inputs/transformer_method.txt`:
  - 12.6s total with `qwen3.5:9b` on M3 Max
  - 11 IR nodes, 11 edges, 2 groups
  - `final_output.png`, `final_output.svg`, `final_output.pdf`, `diagram_ir.json`
- Graphviz labels are XML-escaped (`Multi&#45;head`), so label-fidelity tests must unescape.
- TikZ export from IR compiles with `pdflatex` after `tlmgr install pgf standalone`.
  A sample 11-node figure produced a 29KB PDF with exact labels.
- `tlmgr` on this Mac needed `tlmgr update --self` before installing `pgf`/`standalone`.

## Known gaps

1. `--format svg` is rejected by `Settings.validate_output_format` (existing bug).
   Use `--format png --vector-export both`; the vector files are written separately.
2. `openai_local` lacks API key / `extra_body` / `reasoning_effort` support.
3. No per-role `VLM_ROUTING` yet, so Tailnet hybrid routing (oMLX/TensorFold/vLLM)
   is not active. Ollama remote works by setting `OLLAMA_BASE_URL`.
4. Graphviz theme is still plain (Helvetica 10pt, minimal palette).
5. `server.json` still identifies the upstream MCP server.
6. `paperbanana local ...` CLI (init/doctor/render/benchmark) is not implemented;
   profiles are used with `--config`.
7. IR mode still calls the `structurer` progress label during vector export even
   when it reuses the cached IR (cosmetic).

## Next step

Phase 4: add a per-role provider factory (`VLM_ROUTING`) and a generic
OpenAI-compatible provider with API key / `extra_body` / `reasoning_effort`,
then wire `planner`/`critic` to `shinjimac-studio` and `structurer`/`retriever`
to a text-only TensorFold endpoint. Keep local fallback explicit.

## Resume commands

```bash
cd /Users/petadimensionlab/workspace/paperbanana
git status
git log --oneline --decorate -8
.venv/bin/python -m pytest -q
VLM_PROVIDER=ollama OLLAMA_MODEL=qwen3.5:9b OLLAMA_JSON_MODE=true \
  OLLAMA_REASONING_EFFORT=none \
  .venv/bin/paperbanana generate -i examples/sample_inputs/transformer_method.txt \
  -c "Overview of our encoder-decoder architecture" \
  --render-mode ir_fast --vector-export both --output-dir /tmp/pb-ir-test
```
