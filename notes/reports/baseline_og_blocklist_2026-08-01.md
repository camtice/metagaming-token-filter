# Baseline: the OG blocklist features against human token labels

**Date:** 2026-08-01 · **SAE:** `gemma-scope-2-27b-it`, `resid_post/layer_40_width_16k_l0_medium` · **Status:** this is the project baseline; all future feature-selection work is measured against it on the same labeled data.

## The feature set

**The OG blocklist: 222 features** (`data/latent_list_2026-07-29.json`), flagged by matching the 338-term, 5-category oversight blocklist (`scripts/regex_v3.py`) against the Neuronpedia autointerp captions of the 16,384 SAE features (`scripts/search_explanations_v3.py` → `out/candidates_v3_40-gemmascope-2-res-16k.jsonl`). No activation data was involved in flagging them. This set, unmodified, is the baseline.

## The evaluation

**Data:** the human span annotations in `camgeodesic/metagaming-labeling` (HF) — the `annotations` config marks metagaming spans (whitespace-word indices, end-exclusive; basis verified by exact word-count match) over documents drawn from the sibling configs. Usable: **103 reviewed documents** — 70 with ≥1 human-labeled metagaming span, 33 reviewed-clean — 35,920 gemma tokens, 27.2% labeled positive. Two Kimi docs dropped (word-count mismatch with the annotation tool, likely truncation). Assembly: `data/labeled_eval_docs.jsonl`.

**Prediction rule** (Rathi & Radford, arXiv:2601.21571 §5.1): a token is flagged when ≥2 set features activate at ≥ their own mean + 4 SD (statistics over this corpus), then flags grow onto adjacent tokens with any positive activation until convergence.

**Pipeline:** `scripts/harvest_acts.py` (one GPU pass, sparse per-token activations) → `scripts/score_labeled.py` → `out/labeled_eval_results.json`. CPU-reproducible from the harvest.

## Baseline numbers

| metric | OG blocklist (222) |
|---|---|
| token precision | 0.50 |
| token recall | 0.65 |
| token F1 | 0.57 |
| recall, Apollo CoT docs | 0.57 |
| recall, Kimi chat docs | 0.34 |
| false-positive rate on reviewed-clean docs | 12.7% |

## Why earlier selection work was retired

Between 2026-07-29 and 07-30, several feature-selection generations were built and scored using **whole documents** as forget/retain units (document-level separation statistics, and token budgets measured against whole control documents). The human labels show that framing was wrong: even in curated positive documents, only ~27% of tokens are actually metagaming content, so document-level testing systematically rewards features that detect a document's *topic* rather than the labeled behavior. Those selections, their scores, and their datasets are not comparable to this baseline and are excluded from future comparisons (archived under `out/retired_2026-08-01/`, `scripts/retired/`, `notes/reports/retired/`).

## Path forward

Improvements must beat the table above **on human-labeled data**, with selection and evaluation kept on separate labeled subsets as annotation coverage grows. Known gaps to attack: chat-register recall (0.34 — chat rollouts were absent from everything that came before; only 36 Kimi docs are annotated so far), and deployment thresholds still require activation statistics from a real pretraining corpus rather than these small probe corpora.
