# Haiku prompt iteration report — matching Fable's rollouts

**Goal:** close the gap between Haiku 4.5's labels and Fable 5's on the
gemma-3-27b L40 16k SAE (baseline: 97.3% binary agreement, but Haiku caught
only 241 of Fable's 649 forget latents).

**Eval:** all 444 usable binary disagreements + 250 agreement controls
(100 forget-agree, 150 keep-agree, seeded sample). Metric = agreement with
Fable's label. Haiku, temperature 0, Batch API (~$1.4/round).

## Iterations

| variant | change | disagreements matched | controls held | overall |
|---|---|---|---|---|
| v1 (baseline) | production prompt | 0% (by construction) | 100% | 64% |
| v2 | promoted-token rule; "use low freely, prefer forget-low when torn"; 2 low-confidence few-shot examples from Fable rollouts (15213, 7408) | 11% | 96% | 41%* |
| **v3** | v2 + explicit scope clarifications naming Haiku's systematic exclusions as in-scope | **35%** | **99%** | **58%** |

*overall = weighted mix of the (adversarial) disagreement set and controls; the
baseline row is 250/694.

## What made the difference

v2's generic recall-pushing barely moved Haiku. Error analysis showed Haiku
was *perceiving* content correctly but drawing category boundaries tighter
than Fable — its own rationales said "regulatory compliance", "homework
assignments", "CNN architecture"… followed by "no oversight sense". v3 names
those clusters as in-scope:

1. **Educational materials** (worksheets, homework, drills, lesson content)
   are c3 even without explicit grading — assessment is intrinsic to the
   setting.
2. **Regulatory/institutional compliance** (licensing, standards,
   certification, inspections) is c3 — regulated entities are being monitored.
3. **NN architecture / ML-methods content** is c4 — co-occurs with training
   machinery.
4. **"It's just a broad topic feature" is never by itself a reason to keep.**

## v3 decomposition

- Fable-only latents flipped to forget: 149/406 (4/4 of Fable's high,
  ~2/3 of its medium, 102/335 of its low tier)
- Haiku-only latents flipped to keep: 6/38
- Keep-controls held: 150/150 (no new false-forgets)
- Forget-controls held: 98/100
- Category agreement when both say forget: 242/247 (98%)

## Projection to full 16k

Binary agreement 97.3% → ~98.2%; v3-Haiku captures ~95% of Fable's medium+
forget set (vs ~60% at baseline). The residual gap is almost entirely Fable's
`low` tier — temperament on "plausible connection only" latents, which no
instruction fully transfers.

## Rounds 3–5 (extension authorized by user)

| variant | change | disagreements matched | keep-ctls | projected full-16k agreement |
|---|---|---|---|---|
| v4 | v3 + ANY-OVERLAP procedure (one on-category snippet/token ⇒ forget/low) + 5 clusters (legal/statutory, privacy/tracking, workplace supervision, crime/deception, ML-research artifacts) | 43% | 149/150 | 97.8% |
| v5 | v4 + 5 more clusters (competitions/online judges, homework/answer keys, authority investigations, CI/coverage, oversight-vocabulary features) + coercive EVIDENCE RULE | 50% | 146/150 | 96.1% |
| **v6** | **v5 minus the evidence rule** (ablation) | **50%** | **149/150** | **98.0%** |

The evidence rule ("if you cannot affirm zero overlap, answer forget") bought
3 extra disagreement fixes but cost ~4× that in extrapolated false flags —
the clusters do the work; the coercion just adds noise. v6's still-missed set
is 179 Fable-low + 9 Fable-medium latents — largely Fable's coin-flip zone
(Fable itself labels near-identical latents inconsistently there, e.g.
recruitment latents 11087 keep vs 12315 forget).

## Recall track (rounds 6–7, user-directed: optimize recall on 16k)

Objective changed from match-Fable to maximize capture of Fable's forget set;
precision cost reported, not optimized.

| variant | change | Fable-only recall | fgt-ctls | keep-ctls (flip mix) |
|---|---|---|---|---|
| v6 (ref) | balanced winner | 53% | 99/100 | 149/150 (1 low) |
| v7 | + recall-objective framing (asymmetric cost), reinstated evidence rule, "would it ACTIVATE on category text" bar with token-co-occurrence argument, 5 clusters (devops/build, legal consent/privacy, sports officiating, competency assessment, cert-exam mentions) | 61% | 99/100 | 145/150 (5 low) |
| **v8** | + final-sweep rules (entertainment trivia, 'testing' release channels, logit-only signals like 'purportedly'/'aware'/blanks, judged-by-authority, strongest-snippet rule) + 4 few-shot examples rendered from thinnest-overlap misses (7983, 12746, 9291, 12551; excluded from scoring) | **67%** | 99/100 | 144/150 (4 low, 2 med) |

Projected onto the full 16k: v8 captures ~79% of Fable's 649 forget latents
(vs 37% at v1 baseline), at a price of roughly 600 extra non-Fable forget
labels (~2/3 low-confidence, threshold-removable; ~200 medium). What v8 still
misses is almost entirely Fable's faintest tier — single-judge coin-flips.

## Cross-model check: Fable + v8 on the dev set (2026-08-15)

Fable given the v8 (recall) prompt flags **91% of the dev set** (627/688):
perfect self-consistency on its old forgets (501/501), but **59% of
agreed-keep controls flip to forget** (85/88 at low) — extrapolating to ~60%
of the whole SAE. Conclusion: the v8 *content* rules transfer, but the recall
calibration is model-specific — it corrects Haiku's conservatism and shoves
already-recall-leaning Fable off a cliff. The prompt+model pair is the unit
of calibration, not the prompt.

Soft validation extracted anyway: Fable-v8 endorses 41/41 of Haiku-v8's
"extra" flags vs a 67% base rate on all Fable-v1 keeps (p≈1e-7) — Haiku-v8's
extras are systematically more forget-like than random kept latents.
Labels: relabel/iter_v8_fable5_labels.json.

## Decision

**v6 adopted** (balanced) for the 65k SAE run; **v8** is the recall-optimized
alternative when missing forget latents is costlier than over-flagging (`40-gemmascope-2-res-65k`, Haiku 4.5).
Projected agreement with a hypothetical Fable run: ~98.0% binary, with ~95%+
of Fable's medium+ tier captured. Prompt construction:
`relabel/prompt_iter.py` (`build_variant_system("v6", ...)`); per-round labels
in `relabel/iter_v{2..6}_labels.json`.
