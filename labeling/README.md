# Labeling pipeline — how the judge pools were produced

This directory is the audit trail for `data/candidate_sets/judge_pools_final_v2.json`:
the code, prompts, and reports behind the LLM-judged SAE forget-set labels
(gemma-3-27b, layer 40, Gemma Scope 2 residual SAEs at 16k/65k/262k widths).

## Why LLM judging (not descriptions or regex)

Auto-interp descriptions are unreliable: many oversight-sounding latents are
token-prefix artifacts (a "reinforcement learning" latent firing on «Rein» in
*Reinbeck*; "fine-tuning" on «Fin» in *Finlay*). Judges therefore see only
**real evidence**: the top-10 max-activation snippets (±20 tokens around the
max token) and the top-10 promoted output logits, pulled from Neuronpedia's
bulk S3 exports (`neuronpedia-datasets.s3.amazonaws.com/v1/gemma-3-27b/
40-gemmascope-2-res-{16k,65k,262k}/{activations,features}/…`). Descriptions
are never shown to the judge. ~90% of the final forget latents were never
flagged by the earlier 334-term regex pipeline.

## Pipeline (`pipeline/`)

| script | role |
|---|---|
| `fetch_exports.py` | pull Neuronpedia S3 activation/feature/explanation dumps |
| `build_prompts.py` | base (v1) system prompt: forget-set definition, c1–c5 categories, recall-heavy rule, 14 few-shot examples rendered from hand-verified latents; per-latent user messages |
| `prompt_iter.py` | **the prompt stack**: v2–v8 constructions layered on v1, plus the dev-set eval harness (694 latents: all Fable/Haiku disagreements + controls) |
| `build_16k_variant.py`, `build_65k.py` | full-run request builders per SAE width |
| `build_262k_stream.py` | streaming build+submit for the 262k width (no giant request file; resumable) |
| `run_batch.py` | Batch API submitter: chunking, retry/backoff, per-model config |
| `collect.py` | parse batch results → label JSONs + summary/agreement reports |
| `compare_calibration.py` | cross-model calibration comparison |

## Prompt evolution (details in `reports/haiku_prompt_iteration_report.md`)

- **v1** — base prompt. Judged by 4 models on 208 hand-verified latents;
  Fable 5 won (best recall + schema discipline) and produced the reference
  `fable_16k` labels.
- **v2–v6** — Haiku 4.5 iterated to match Fable on the dev set (0%→50% of
  disagreements): promoted-token rule, explicit scope clusters (education,
  compliance, NN-architecture, competitions, authority investigations…),
  any-overlap procedure. **v6** = balanced Fable-matcher.
- **v7–v8** — recall-optimized: asymmetric-cost framing, evidence rule,
  "would it *activate* on category text" bar, thin-overlap few-shot examples.
  **v8** = the pool prompt (53%→67% Fable-only recall on dev; ~80% full-run).
- Cross-model check: the v8 prompt on *Fable* over-flags 91% of the dev set —
  prompt calibration is model-specific; v8 is a **Haiku** prompt. Fable+v8 did
  endorse 41/41 of Haiku-v8's extra flags (vs 67% base rate), evidence that
  v8's extras are mostly signal.

## API configuration notes (Claude 5-era quirks)

- `temperature` is rejected by Claude 5 models (Haiku 4.5 uses temperature 0).
- Sonnet/Opus 5 need `thinking: {type: "disabled"}`; Fable/Mythos reject that
  (adaptive-only) — omit `thinking` and give `max_tokens` headroom instead.
- Batch API halves cost; a shared cached system prompt reaches ~100% cache-hit
  at scale when chunks are submitted staggered.
- Fable safety-refused 64 bio/lab/code latents (16k); those labels were
  backfilled from Haiku and are marked `source: haiku-fallback-fable-refused`.

## Caveats for auditors

- Six latents in each 16k v6/v8 pool double as few-shot examples in their own
  prompt (15213, 7408; v8 adds 7983, 12746, 9291, 12551) — trivially forget.
- 262k: 241,678 of 262,144 latents were alive and judged; latent 9429
  unparseable and unlabeled.
- v8's `low` tier is deliberately permissive — ratio-screen or threshold by
  confidence before use (see repo README recipe).
- Full per-latent label files (with rationales) are not committed here for
  size reasons; they are regenerable with this pipeline (Neuronpedia exports +
  Anthropic Batch API), and `judge_pools_final_v2.json` carries the complete
  forget-side selections. Per-run costs: fable_16k $301, haiku 16k runs
  ~$15–17 each, 65k $56–65, 262k $239 (~$770 total incl. calibrations).

## Reports (`reports/`)

Per-run summaries (counts by category × confidence, prior-verdict agreement),
the 4-model calibration comparison, the Fable-vs-Haiku disagreement listing,
and the full prompt-iteration report.
