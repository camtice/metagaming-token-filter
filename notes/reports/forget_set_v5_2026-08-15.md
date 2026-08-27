# Protocol v5: dense c1–c4 join the main forget set — selection rerun + feature shift
2026-08-15 · artifact `Forget-Set Feature Shift` (bbc1ccab) · `notes/feature_shift_2026-08-15.html`

## Decision (user, 2026-08-15)

Main forget set = **human span tokens + all tokens of dense c1–c4 docs** (all-forget
assumption). Rationale: dense docs register-match the retain controls, so selection can
separate forget from retain without the chat-template confound. Human-span metrics stay
as a **separate secondary report**. Dense c5 (SWE) remains retain-side per the 08-12
ruling and is excluded from the forget definition (still reported in dense_coverage).

`score_split.py` now emits both blocks: `main` (headline P/R/F2/F1 over annotated +
dense c1–c4 docs) and `annotated` (human spans, unchanged).

## Selection rerun (v5)

Same pool (fable_16k, 649), same machinery (multi-start constrained hill-climb, seed
20260815), objective on the main metric, fp_ctl ≤ .10 hard.

| set | n | main P/R/F2 | span P/R/F2 | fp_ctl | chat span R |
|---|---|---|---|---|---|
| **D2** (max main F2) — recommended | 187 | .878/.883/**.882** | .417/.793/.672 | .100 | .527 |
| D1 (max main R) — dominated, ignore | 170 | .800/.865/.851 | .306/.865/.633 | .100 | — |
| P2 (span-F2 winner, reference) | 134 | .911/.765/.790 | .548/.818/.745 | .086 | .682 |
| OG baseline (official scorer) | 222 | .854/.708/.733 | .382/.682/.590 | .141 | .427 |

Note: main-metric precision is mechanically inflated for all sets (dense docs are
all-forget → no negative tokens in their share of the precision denominator). Never
compare main-P to span-P. The buggy v5 `references.OG` row (non-pool latent lookup)
must not be used; official scorer numbers above.

## Feature shift (D2 vs P2): 85 kept · 102 added · 49 dropped

- **Added** features are dense-corroborated topic detectors (mean dense fire rate .0125,
  3× the dropped group): large language model (4460), OpenAI generative AI (5447),
  compute costs (13418), exam/test/study (4859), loss function (15691), human
  feedback/judgment (13748), standardized-test prep (9002)…
- **Dropped** features are span-only register/format carriers (mean dense rate .0038):
  "Let, ratio, proportion" (9839, dense rate 0.0000), "prediction accuracy" (9593),
  counting/arithmetic (11794), uncaptioned register latents 1491/1365…
- **10 of the 13 register suspects** from the precision post-mortem are out of D2;
  817 (assistant-scaffold) and 1419 (test-cases) survive because they carry genuine
  dense evidence too.

## The cost, stated plainly

Human-span **chat recall drops .682 → .527** (agentic .432 → .662 actually improves).
The evicted register features were carrying real chat metagaming spans — in chat, the
signal genuinely lives inside assistant-answer scaffolding. The fix is more labeled
chat rollouts (and possibly a dense-style chat-register forget corpus), not re-adding
filler features.

## Files

Frozen sets `data/candidate_sets/fable_trim_d{1,2}.json` (scorer-confirmed);
selection `scripts/select_latents_fable_v5_dense.py`;
artifact generator `scripts/build_shift_artifact.py`;
data `out/fable_trim_v5_dense.json`, `out/feature_shift_d2_p2.json`,
`out/feature_profiles_649.json`. Sealed validation remains unspent.
