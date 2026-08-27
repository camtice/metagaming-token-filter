# Generalization study: which selection methods survive a real train/test split?
2026-08-15 · `scripts/generalization_study.py` · `out/gen_study_*.json` · split `data/splits/gen_split_20260815.json`

## Design

Motivated by the concern that every method so far computed its statistics on the corpus it
was scored on. This study isolates the selection step with a doc-level split **inside**
TEST v5 (sealed VAL-A/VAL-B untouched):

- **G-TRAIN** — seeded halves of dense c1/c3/c4 (15 docs each) + half the retain controls
  (1,312 docs). *All* selection information — R1/R3 statistics, thresholds, hill-climb
  objective — comes from here. **No span labels are ever used for selection.**
- **G-TEST** — the other dense halves (in-domain check), **all 30 dense c2 docs (held-out
  domain — AI-safety oversight never seen)**, **all 25 human-span docs (transfer to real
  rollouts)**, the 20 reviewed-clean docs, the other 1,313 controls.
- Dense c5 excluded (retain-scope). Rathi thresholds fixed from the full harvest for all
  methods. Metrics: `F2_in` (forget = spans + in-domain dense; c2 masked out), `F2_all`
  (c2 as forget), plus the four recalls and control FP. **F2 here is not comparable to
  earlier F2m** (much smaller forget mass against ~1,300 control docs); compare only
  within this study.

## Headline: the ratio rule generalizes best — better than optimization — in every pool

Train-selected operating points (threshold chosen by best G-TRAIN F2 at fp≤.10, then
frozen and read once on G-TEST):

| pool | method | n | F2 train→test | R_span | R_c2 (held-out) | R_in | fp |
|---|---|---|---|---|---|---|---|
| fable 16k | **R3 ρ=10** | 135 | .726→**.690** | .707 | .785 | .835 | .032 |
| | hill-climb | 217 | .726→.671 | .625 | .755 | .796 | .029 |
| | R1 τ=.07% | 321 | .583→.574 | .551 | **.493** | .647 | .029 |
| haiku v6 16k | **R3 ρ=10** | 117 | .744→**.714** | .708 | .776 | .840 | .026 |
| | hill-climb | 196 | .731→.688 | .668 | .759 | .826 | .030 |
| haiku v8 16k | **R3 ρ=10** | 140 | .735→**.704** | .723 | .803 | .874 | .035 |
| | hill-climb | 255 | .708→.655 | .650 | .715 | .828 | .041 |
| haiku v6 65k | **R3 ρ=10** | 579 | .773→**.719** | .707 | .861 | .894 | .033 |
| | hill-climb | 1164 | .611→.577 | .782 | .894 | .929 | .093 |
| haiku v8 65k | **R3 ρ=10** | 682 | .756→**.710** | .721 | .871 | .904 | .039 |
| | hill-climb | 1399 | .684→.632 | .739 | .850 | .909 | .065 |

**R3 beats the hill-climb on held-out F2 in all five pools** (by .02–.14), with smaller
train→test gaps (~.03–.05 vs .04–.06+). Optimization buys nothing that survives the split.
(Climb caveat: budget-limited to ~12k evaluations; at 65k it underperformed even on train —
but that *is* the finding at realistic budgets: the search doesn't find anything beyond the
ratio statistic, and at 65k it actively flails in the larger space.)

## The four specific questions, answered

**1. How much did same-corpus statistics flatter us?** The contaminated variant (R3 with
statistics from the full corpus, old practice) beats the clean train-stats variant on the
*same* G-TEST by **+.02 to +.04 F2** at matched ρ, in every pool. Real, consistent, and
modest — the earlier conclusions survive, shaved by ~.03.

**2. Does selection generalize across domains?** Yes for R3, no for R1. R3 trained with
zero c2 exposure recovers **.78–.87 of held-out c2 tokens**, only ~.03–.06 below its
in-domain recall. R1's train-selected point collapses on c2 (**.39–.66**, a .15–.25 domain
gap) — the control-cap keeps whatever isn't loud on controls, which is domain-arbitrary.
This is the clearest evidence yet that the *ratio* captures topical structure rather than
corpus accidents.

**3. Does dense-trained selection transfer to real rollouts?** Reasonably: span recall
.71–.72 at the strict train-selected ρ=10, rising to .77–.83 at ρ=3 (fp .10–.17). Spans
were never in training, so this is genuine transfer — dense synthetic docs are a usable
(if not sufficient) supervision source for the chat/CoT registers. The looser-ρ trade
suggests the deployment choice depends on whether span recall or FP is the binding
constraint.

**4. Which width / labeler / prompt wins?** Ordering under clean statistics, by held-out F2:
**65k > 16k by ~.01–.03** (v6-65k .719 is the single best cell; 65k also transfers spans
and c2 better at matched ρ). Within a width, **labeler and prompt version are noise**:
fable ≈ haiku-v6 ≈ haiku-v8 within ~.02, same as every previous comparison. The pool is a
recall reservoir; the screen does the work.

## Baselines for context (no training at all)

Embed≥0.9 (caption faithfulness): span recall .76–.90 but fp .18–.34 — recall-rich,
precision-poor, and strictly dominated by R3 at any matched fp. OG∩pool: fp .01 but span
recall .43 and c2 recall **.20** — the regex-picked features barely touch the held-out
domain. Full pools: flood (fp .50–.67).

## Caveats

- The **pools themselves** were built by judges that saw related material, and the ρ/τ
  *grids* were designed after earlier TEST work — this study cleans the selection-statistics
  layer, not the full history. The sealed validations remain the final arbiter.
- The G-TRAIN forget signal is dense-only (90 synthetic docs): deliberately hard, and it
  shows dense supervision alone leaves ~.28 of span tokens uncovered at strict ρ. More
  labeled rollouts remain the highest-leverage data.
- Single split (seed 20260815); halves are small (15 docs/class). Rerunning with 2–3 seeds
  would tighten the gaps' error bars cheaply.

## Recommendation

Adopt **R3 with train-only statistics** as the selection method of record: it wins
generalization in every pool, needs no search, and its statistic can be computed from any
forget/retain corpus pair (including, later, non-TEST corpora — the fully hygienic version).
Set ρ by FP budget (ρ=10 → ~3% fp; ρ=5 → ~7–9%; ρ=3 → ~10–17% on held-out controls). Pool:
**haiku v6 or v8 at 65k** (v6-65k ρ=10 = best held-out F2 .719; drop to ρ=5 if span recall
matters more than FP). Retire R1-alone and the embed screen. Natural next steps: 2–3 more
split seeds for error bars, then the single sealed VAL spend on the chosen frozen point.
