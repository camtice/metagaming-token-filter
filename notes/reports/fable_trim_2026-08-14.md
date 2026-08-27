# Fable latent selection — F2 maximization on TEST v4 (2026-08-14)

**Ask:** from the fable_16k judge pool (649 latents), select a subset maximizing
token-level F2 on TEST, targeting recall ≥ 0.90 and control FP ≤ 0.10, by a
method simple enough to justify in a paper. Selection used TEST only (manifest
v4, sha `8fa54624`, harvest `out/harvest_test_v3`); sealed VAL-A/VAL-B untouched
(access log unchanged, 3 historical entries).

## Headline result

**The two constraints are jointly infeasible** for this pool under the frozen
Rathi rule (seed = ≥2 features ≥ mean+4SD; contiguous-run windowing). The
frontier passes R .88 @ FP .10 and R .90 @ FP .14 — not through (.90, .10).
Three operating points were maximized and frozen (all reproduced by
`score_split.py`, the official arbiter):

| set | n | P | R | F2 | fp_ctl | fp_clean_docs | R chat / agentic | file |
|---|---|---|---|---|---|---|---|---|
| **P1** — fp≤.10 hard, max recall | 141 | .309 | **.880** | .642 | .100 | .389 | .791 / .790 | `data/candidate_sets/fable_trim_p1.json` |
| **P2** — fp≤.10 hard, max F2 | 134 | .548 | .818 | **.745** | .086 | .126 | .682 / .432 | `data/candidate_sets/fable_trim_p2.json` |
| **P3** — R≥.90 hard, min fp | 147 | .269 | **.900** | .613 | .140 | .452 | .818 / .835 | `data/candidate_sets/fable_trim_p3.json` |
| OG blocklist (baseline) | 222 | .382 | .682 | .590 | .141 | .179 | .427 / .398 | `data/latent_list_2026-07-29.json` |
| fable_16k full pool | 649 | .207 | .942 | .551 | .627 | .695 | .908 / .904 | — |

Every point dominates the OG baseline on F2 or recall; P2 dominates it outright
(higher P, R, F2, lower FP on both control families).

Bootstrap 90% CIs (500 doc-level resamples of the 45 annotated docs):
P1 R [.838, .919] F2 [.527, .750] · P2 R [.746, .882] F2 [.661, .816] ·
P3 R [.861, .935] F2 [.492, .732]. The annotated core is 25 span + 20 clean
docs (6,577 gt tokens), so intervals are wide; agentic recall rests on 1 doc.

## Method (the paper story)

1. **Evidence** (`scripts/select_latents_fable_v1.py`): per-latent fire rates
   under the scorer's own mean+4SD thresholds — on human-labeled span tokens
   (q_span), dense c1–c4 assumed-forget tokens, and the 625 retain-control
   docs (r_ctl; 517k tokens incl. SWE-as-retain). Jeffreys smoothing.
2. **One-knob sweeps** (v1): LLR ranking `log(q_span/r_ctl)` prefix sweep;
   per-feature FP cap τ + recall-carrier ranking; span∪dense pooled variant.
   Best feasible-FP point: R .83 @ fp .095 (F2 .609). Insufficient.
3. **Greedy backward elimination** (`v2_greedy`): from all 649, remove the
   latent with the best exact ΔFP/Δrecall ratio (lazy re-evaluation, monotone
   under removal). Pareto path: R .903 @ fp .162.
4. **Simulated annealing** (`v3_anneal`): 45k exact evaluations, 3 chains,
   toggle/swap moves, constraint-penalized objective. All chains converge to
   R ≈ .876–.879 at fp ≤ .10 — an empirical ceiling, not search failure.
5. **Constrained hill-climb polish** (`v4_polish`, seed 20260814): steepest
   ascent over sampled add/remove/swap moves → the three frozen points.

## Why (.90, .10) is unreachable here

- **Per-feature cap curve** (upper bound on recall for any subset whose members
  individually flag ≤ τ of control tokens — take *all* such features):
  τ=.0002→R .26 · τ=.001→R .60 · τ=.002→R .69 · τ=.01→R .85 · τ=.02→R .94.
  Reaching R ≥ .90 forces inclusion of features with individual control rates
  >1–2%, whose collective set-level FP exceeds .10 (already .30 at τ=.01 with
  everything included, and thinning the set only lowers recall).
- **Windowing amplification**: at the greedy R=.903 point, fp_ctl .162 =
  .071 seed + .092 window — the adjacency expansion more than doubles control
  FP because kept recall carriers leave weak positive activations across
  ordinary text, and one stray seed flags the whole contiguous run.
- The last ~4pp of recall ([.88 →.92]) live on span tokens whose only
  above-threshold carriers are these broad features.

## Composition of the frozen sets

All three look alike: ~60–65% c3 (human oversight), ~15% c4, ~15–20% c5,
few c1/c2; confidence mix ~60% l / 25% m / 15% h — the judge's confidence
tiers are only weakly predictive of survival; ~16–17 features overlap OG-222.
Notably ~20–28 c5 (SWE) features survive *despite* SWE controls counting as
retain FP — they carry span recall beyond SWE text.

## Caveats

- Selection and evaluation share TEST (by protocol design; TEST = iterate
  freely). With ~9k exact evaluations against 6,577 gt tokens, adaptive
  overfitting is a real risk — treat TEST numbers as optimistic. The honest
  numbers come from a **single sealed VAL-A/VAL-B spend on one chosen frozen
  set** (`--unlock-validation --frozen-set`), which has not been performed.
- `fp_ctl` is measured against *assumed*-clean controls; any true metagaming
  tokens in dolci_think/wildchat inflate apparent FP.
- P2 buys its F2 by dropping agentic recall to .43 (n=1 doc) and chat to .68 —
  tension with the recall-first goal; P1/P3 hold both registers ≥ .79.

## Recommendation

If the .90 recall floor is hard, take **P3** (fp .14, still under baseline's
.141). If the FP budget is hard, take **P1** (R .88, F2 .642). **P2** is the
F2-max exhibit (.745) and the only set that also tames clean-chat FP (.126).
Decide the operating point, then spend one sealed validation run on that single
frozen set.

## Artifacts

Scripts `scripts/select_latents_fable_v{1,2_greedy,3_anneal,4_polish}.py`;
sweep/search outputs `out/fable_trim_v1_sweeps.json`, `out/fable_trim_v2_greedy.json`,
`out/fable_trim_v3_anneal.json`, `out/fable_trim_v4_polish.json`; official
score files `out/score_harvest_test_v3_fable_trim_p{1,2,3}.json`.
