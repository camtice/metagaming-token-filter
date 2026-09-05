# Metagaming Token Filter — SAE selection codebase

Code, results, and reports for selecting SAE features that flag *metagaming* content
(models reasoning about their evaluation/oversight) in pretraining data, following
[Rathi & Radford, arXiv:2601.21571](https://arxiv.org/abs/2601.21571).

**Handing off?** `HANDOFF_NATHANIEL.md` collects every resource for the category-split rebuild (regex lists, data, reference results, protocol constraints).

**Start here:** `artifacts_html/project_overview.html` (the collaborator handbook — task,
scope, data map, evaluation protocol) and `artifacts_html/gen_atlas.html` (every
quantitative result, interactive). Open them locally in a browser.

## The current recommended recipe

Judge-curated feature pool → **ratio screen** (keep features firing ≥ρ× more per token on
forget training text than on retain controls; statistics from the training split only) →
**Rathi labeling rule** with a **token vote of k≥4** co-firing features per seed (paper
default is 2) and adjacency windowing, thresholds at mean+4SD.

Best measured set: haiku-v8 pool · 65k SAE · ρ≥10 · k=4 — held-out generalization F2 .821
(in-distribution .802), span recall .675, held-out-domain (c2) coverage .835, control FP
1.1%, 682 features. See `notes/reports/` and `report/` for the full findings,
including the judge-vs-full-dictionary comparison (tied on real hand-labelled text at
matched FP; the no-judge edge is confined to the held-out synthetic category and lacks a
real-text check) and why simple rules beat combinatorial optimization on held-out data.
Frozen deployment candidates: `data/candidate_sets/frozen_ladder_{judge65k,alldict65k}_v1.json`
— nested sets at ~1/3/5/10% token-removal budgets.

Tested and rejected: the 262k SAE at the same layer (best cell F2 .583). Only `l0_small`
ships at that width (≈21 active latents/token vs ≈60 for 16k/65k `l0_medium`), and the
sparse code starves the co-firing vote — k=4 collapses recall, so the deficit is
attributable to L0, not dictionary size. Rows in `out/pool_fairgrid.json` and the atlas.

## What is here

| path | contents |
|---|---|
| `report/` | the selection-rule ablation report: PDF, Typst source, figure code, results.json (single source of truth), known-answer tests |
| `scripts/` | all pipeline code: harvesting (`harvest_acts.py`, `sae_local.py`), the official scorer (`score_split.py`), selection/screening/ablation studies, artifact builders |
| `notes/reports/` | dated markdown reports for every study (read in order of date) |
| `artifacts_html/` | self-contained interactive result pages (atlas, frontrunners, threshold study, feature-shift, handbook) |
| `data/candidate_sets/` | the SAE feature sets (latent indices + provenance) incl. judge pools and frozen selections |
| `data/splits/` | the sealed split manifests (document **ids only**) and validation access log |
| `data/annotation_exports/` | human span-label exports (word indices only, no text) |
| `data/latent_list_2026-07-29.json` | the original regex-selected blocklist features |
| `out/` | aggregate result JSONs backing every report and artifact page |

## What is deliberately NOT here (and where it lives)

- **Document texts** (rollouts, dense docs, controls): private HF dataset
  `camgeodesic/metagaming-labeling` — request org access. All corpora used here are
  reconstructable from the split manifests (ids) + that dataset at the pinned revisions.
- **Activation harvests** (`.npz`, tens of GB): regenerate with
  `scripts/harvest_acts.py` (one GPU pass per corpus; see the handbook's pipeline section).
- **Sealed validation outputs**: held by the project owner; VAL-A/VAL-B results are
  revealed once, at the end, per the protocol.
- **SAE weights / captions**: public — `google/gemma-scope-2-27b-it`
  (resid_post layer 40, 16k & 65k l0_medium; 262k l0_small also evaluated) with base model
  `unsloth/gemma-3-27b-it`; Neuronpedia caption dumps are re-downloadable.

## Reproducing the headline result

```bash
# 1. materialize corpora from the HF dataset per data/splits/split_manifest_v5.json
# 2. one GPU pass per corpus/width:
python scripts/harvest_acts.py gemma3-l40-65k test_docs_v5.jsonl out/harvest_test_v5_65k
# 3. selection + evaluation are CPU-only:
python scripts/generalization_study.py      # the train/test split study
# fair rho x k grids: see scripts referenced in notes/reports/qualitative_investigation_*.md
```

Known environment gotchas (measured, don't rediscover): Gemma-3 needs a prepended BOS;
use bf16 (fp16 NaNs); the Gemma Scope hook `layers[L].output` equals HF
`hidden_states[L+1]`.

## Protocol guardrails

- Selection statistics come from the training split only; TEST is iterate-freely;
  VAL-A/VAL-B are sealed (never train, tune, or select on their ids — including their
  rows in the annotations export).
- Same-corpus statistics inflate held-out F2 by +.02–.04 (measured) — don't do it.
- Internal metrics overweight synthetic-style transfer: externally validate selections on
  real text (westover/kimi differential AUC) before trusting them.
