# Evaluation protocol v1 — test & validation splits for SAE feature-set comparison

**Date:** 2026-08-09 · **Dataset:** `camgeodesic/metagaming-labeling` pinned at revision `244b6d00aeee7287ed3dc00fe591f00ec206774f` · **Manifest:** `data/splits/split_manifest_v1.json` (sha256-sealed) · changes require a v2 manifest, never edits to v1.

## The split

| tier | contents | usage |
|---|---|---|
| **VALIDATION (69 docs, locked)** | all 57 `frontier_metagame_rollouts` (Apollo; 45 span-annotated, 5 reviewed-clean, 19 unannotated) + 12 annotated `kimi_metagame_rollouts` docs (seeded stratified pick over families × span-presence) | scored only for **frozen** feature sets via `score_split.py --frozen-set`; every access appended to `data/splits/validation_access_log.jsonl` |
| **TEST (353 docs)** | 53 remaining annotated docs — 25 chat, 21 controls (incl. the 6 annotated `ctl_*` hard-vocabulary docs whose source config was removed; texts from the verified 08-01 assembly), 7 prose — plus all 300 `dense_c1–c5` docs | scored freely every iteration |
| **SELECTION** | unannotated bulk: ~2,200 `kimi_metagame_rollouts`, 500 `kimi_agentic_rollouts`, `westover_general_split`, `westover_metagame_split`, Needham configs (doc-level gold only) | free for discovery; nothing ever migrates from test/val to here |
| **EXCLUDED (for now)** | `claude_AI_forget`, `claude_AI_mix`, `claude_human_forget`, `claude_general_retain` | not dense enough for token-level scoring (user call, 2026-08-09) |

Dropped as unusable: `kimi:dolci_if_contradictory/4625`, `kimi:doluschat/aviation_diversity_omission` (annotation-tool word counts disagree with the stored text).

**Dense-docs assumption** (spot-checked): the `dense_c*` docs are wall-to-wall in-category synthetic text, scored as ~all-forget *coverage* — reported per category and never folded into precision/recall. Note `dense_c3` is *human-domain* oversight; its coverage column keeps that scope decision visible.

## Scoring

Rathi rule (≥2 set features ≥ their own mean+4SD over the scored corpus; adjacency windowing), token-level, spans as ground truth. **Headline metric: F2** (recall weighted 4× precision — false negatives are the costly error in this filtering project; user decision 2026-08-09). F1 reported alongside for reference. Pipeline: `build_split_manifests.py` → `build_eval_docs.py` → `harvest_acts.py` → `score_split.py`. Continuity: the new scorer reproduces the 08-01 baseline on the old 103-doc assembly (P .503 / R .647 / F1 .566; the clean-doc FP rate shifts 12.7%→11.0% from a sharper clean definition — label-derived rather than status-derived).

## Baseline reference rows (OG blocklist, 222 features)

**TEST v1** (iterate against this freely):

| metric | value |
|---|---|
| annotated P / R / **F2** (F1) | 0.567 / 0.667 / **0.644** (0.613) |
| recall: chat / control-spans / prose | 0.207 / 0.823 / 0.694 |
| FP rate on reviewed-clean docs | 10.7% |
| dense coverage c1 / c2 / c3 / c4 / c5 | 0.653 / 0.642 / 0.751 / 0.628 / 0.662 (overall 0.664) |

**VALIDATION v1**: the baseline reference row exists but is **sealed** (`out/sealed/score_harvest_val_v1_OG_blocklist_222.json`) — held out until the end of the project, revealed only at the final milestone comparison. (Its numbers were briefly displayed during protocol setup on 2026-08-09 — a disclosed leak; they are not to be re-displayed or consulted during design work.)

## Rules of engagement

1. Feature selection uses SELECTION-tier data (and any future annotation batches explicitly designated for selection at export time).
2. TEST may be scored at any cadence; report the full table, not just F2.
3. VALIDATION is scored only for a frozen, named feature-set file; `score_split.py` refuses otherwise and logs every access. Keep accesses rare.
4. **Results-viewing discipline** (tightened 2026-08-09): validation does not run by default — `score_split.py` refuses validation harvests unless given `--unlock-validation` *and* `--frozen-set`, and validation output is always sealed to `out/sealed/` (never displayed). Reveal happens once, at the end of the project. Disclosure: the baseline's validation row above was displayed at protocol setup — a mild, logged information leak (validation-is-harder-than-test and its rough register profile are now known; both were largely derivable from the 08-01 eval on the same docs, but future validation runs use sealed mode until a milestone reveal).
5. New annotation exports are folded in by publishing manifest v2 (same script, new seed only if the chat-slice changes) — v1 results stay reproducible at the pinned revision.

## Next data work (highest leverage)

Annotation of the 500 `kimi_agentic_rollouts` (agentic register is entirely unlabeled today) and more chat annotations; both directly grow TEST and future validation without touching v1.
