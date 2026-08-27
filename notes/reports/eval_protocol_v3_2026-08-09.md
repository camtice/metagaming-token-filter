# Evaluation protocol v3 — the standing three-way split

> **Amendment (manifest v4, 2026-08-12)**: c5/SWE scope reversed to **retain** — `ctl_swe_tests`
> (250 ordinary Dolci-Instruct coding docs) scores as assumed-clean FP, not coverage. Split
> membership unchanged; `data/splits/split_manifest_v4.json` (sha 8fa54624) supersedes the
> label semantics below. Baseline TEST retain-FP incl. SWE: 14.1%.

**Date:** 2026-08-09 · **Dataset:** `camgeodesic/metagaming-labeling` @ rev `f7ae7700` · **Manifest:** `data/splits/split_manifest_v3.json` (sha256 `1f30b5ee…`, seed 20260810) · supersedes v1. Changes require v4, never edits.

## The split

| split | n | contents | usage |
|---|---|---|---|
| **TEST (820)** | open | mirror half: 45 annotated docs (25 spanned / 20 clean), 150 dense (assumed-forget), 125 ctl_swe (swe-forget), 500 real-world retain controls (assumed-clean: dolci-instruct/think, fineweb, wildchat) | iterate freely |
| **VAL-A (814)** | sealed | in-distribution mirror of TEST (same strata, incl. the 12 never-seen v1 chat docs force-assigned here); dense corpus included per user instruction | frozen sets only; always-sealed output; logged |
| **VAL-B (57)** | sealed | the frontier (Apollo) rollouts, 100% human-labeled — the highest-quality validation | same lock |

Provenance tags on every doc: `seeded_random` / `kw_enriched` / `assumed_clean` / `assumed_forget` — metrics reportable by tag. SWE content is **forget-scope** (user decision): `ctl_swe_tests` scores as coverage, not FP. Two tool-mismatch kimi docs remain excluded.

## Scoring

Rathi rule, token level; F2 headline (recall-weighted). `score_split.py` handles: annotated P/R/F2 + per-register recall; dense & SWE coverage; FP on assumed-clean by control family. Validation runs require `--unlock-validation` + `--frozen-set`; output always sealed to `out/sealed/`; every access logged.

## Baseline (OG blocklist 222) on TEST v3

| metric | value |
|---|---|
| annotated P / R / **F2** | 0.383 / 0.682 / **0.590** |
| recall: chat / agentic / control-spans / prose | 0.427 / 0.398 / 0.831 / 0.820 |
| FP on real-world retain: instruct / think / fineweb / wildchat | 10.0% / 14.1% / 19.1% / 12.7% (overall **13.6%**) |
| dense coverage (c1–c5) | 0.716 |
| SWE-forget coverage | **0.171** ← noted gap: SWE is in scope but the baseline barely covers it |

VAL-A and VAL-B baseline references exist **sealed** (`out/sealed/score_harvest_val{A,B}_v3_*`), computed unread; reveal only at the final milestone comparison alongside a frozen challenger.

## Open items

Per-feature FP trim of the judge sets against the 500 TEST retain controls (the identified path to a deployable challenger); remaining minimal labels (22 chat picks strengthen both mirror halves via v3.1 test-side refresh — sealed splits never change); spot-check chips (ctl_spotcheck, dense samples) convert assumptions into measured error rates; annotation export re-upload to the dataset still pending.
