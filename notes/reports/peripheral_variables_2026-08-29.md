# Variables investigated but no longer central

**Date:** 2026-08-29 · **Status:** reference. The mainline is frozen for now:
**haiku-v8 judge pool · 65k SAE (`gemma-scope-2-27b-it` resid_post L40 l0_medium) ·
ratio screen ρ≥10 · token vote k=4 · windowing** — held-out F2_all .821 / F2_in .802,
span recall .675, held-out-domain (c2) recall .835, control FP 1.1%, 682 features.
Everything below was tested on the way and set aside. Numbers are G-split held-out
(F2_all) unless noted; sources in `out/` and the dated reports.

## 1. SAE width (16k vs 65k vs 262k) — settled: 65k

16k → 65k is worth +.01–.05 F2_all at matched cells (ρ10·k4: .775 → .821), and the
gain is concentrated in held-out-domain coverage (Rc2 .695 → .835). 262k was tested
and **rejected**: best cell F2_all .583 (ρ3·k2), and tightening ρ or the vote *hurts*
(Rc2 falls to .12 at ρ10·k4). The cause is a confound, not width itself — only
`l0_small` ships at 262k/L40 (measured 20.9 active latents/token vs ~60 at the other
widths), which starves the co-firing vote. Since l0_small is the only option there,
the practical answer is the same: stay at 65k. (`out/pool_fairgrid.json`)

## 2. Skipping the judge — ratio screen over ALL dictionary features

Two versions, both rejected:

- **No screen at all** (whole judge pool, no ratio): F2_all .27–.32, control FP
  50–67%. A feature pool alone is not a classifier.
- **ALL-dictionary + scaled ρ** (drop the judge, screen the full 16k/65k dictionary):
  *looked better* internally (F2_all up to .858) but this is synthetic-register
  leakage — on real text it flags 83–90% of all westover forum posts and external
  differential AUC drops (.76 → .68). The dense corpus has no AI-but-not-oversight
  negatives, so train statistics can't separate AI-topical from oversight-specific;
  the judge supplies exactly that distinction. A hard-negative denominator (needham)
  did not repair it. **The judge stage is necessary.** Use the ratio statistic to
  *propose additions* to the pool (real misses exist, e.g. "my own internal goals"),
  never to replace it. (`out/all{16,65}k_baseline.json`, atlas findings ledger)

## 3. Judge confidence tiers (h/m/l) — do not screen on them

Direct test (h8-65k, k=4): restricting the pool before the ratio screen to
high-confidence latents collapses performance; low-confidence latents carry real
recall and the ratio screen already removes the bad ones.

| pool restriction | n kept (ρ10) | F2_all ρ≥5 | F2_all ρ≥10 |
|---|---|---|---|
| high only (632 latents) | 161 | .370 | .343 |
| high+medium (1,607) | 377 | .729 | .710 |
| all tiers (3,798) | 682 | **.799** | **.821** |

Consistent with the earlier indirect finding: hill-climb survivors were ~60% low /
25% med / 15% high confidence — the judge's confidence is weakly predictive of which
features matter. Keep confidence as metadata only. (`out/conf_tier_h8_65k.json`)

## 4. The OG regex blocklist (222 features from caption keyword-matching)

The project's starting point (2026-08-01 baseline): token P .50 / R .65 / F1 .57,
chat-register recall .34, FP on reviewed-clean docs 12.7%. Under the current
protocol, the OG set intersected with the judge pools (53–65 features survive
judging) scores F2_all .365 (fable pool) / .390 (h8-16k) — precise (fp ~1%) but
recall-starved (span recall .43–.47) vs the mainline's .74–.82. Only ~16–17 OG
features overlap the sets that selection actually keeps. Caption keyword-matching
was a reasonable bootstrap but is strictly dominated by judge pool + ratio screen.
(`notes/reports/baseline_og_blocklist_2026-08-01.md`; `OG∩pool` rows in
`out/gen_study_*.json`)

## 5. Paulo-embedding scoring (Rathi's ≥0.9 filter)

As a standalone no-training screen: F2_all .38–.48 with control FP 18–34% — far off
the ratio screen at any width. It removes junk but keeps broad topical features the
ratio catches. Not used. (`R5`/`embed>=0.9` rows in `out/screen_rules_*.json`,
`out/gen_study_*.json`)

## 6. Control-cap screen ("fires on >τ% of control tokens")

The first simple rule we tried. Works in-distribution, **collapses on the held-out
domain**: ratio screen beats the train-selected control-cap by +.12–.16 F2_all.
A pure FP cap keeps features that are quiet on controls but also quiet on unseen
forget domains; the ratio demands positive evidence. (`out/gen_study_*.json`)

## 7. Combinatorial optimization (hill-climb / greedy / anneal)

Loses to the fixed ρ≥10 rule in 5/5 pools on held-out F2 (e.g. h8-65k: .695 vs
.762 at k=2) with a larger train→test gap — the classic overfit signature. Simple
pre-registered thresholds won; the whole `fable_trim_*` search line is superseded.
(`out/gen_study_*.json` `HILLCLIMB_train`, `notes/reports/fable_trim_2026-08-14.md`)

## 8. Same-corpus ("contaminated") screen statistics

Computing feature statistics on the corpus you then evaluate inflates F2_all by
+.02–.04. Kept as an audit row only; all reported numbers use train-split
statistics. (`R3_contam` rows)

## 9. Labeler model and prompt (haiku vs fable; v6 vs v8)

Both are noise at matched cells (differences within seed-resampling spread).
The v6 prompt is dropped from reporting (v8 kept, indistinguishable); fable's pool
performs ~like haiku's at 16k. Choose the cheap labeler.

## 10. Vote threshold SD (mean+NSD)

Inert across the tested range — recall movement comes from windowing and the vote
quorum k, not the per-feature activation threshold. Fixed at the paper's 4SD.

## 11. Other SAE hosts (early bring-up)

gemma-2-9b L31 (the paper's exact SAE), gemma-3 L46, the `resid_post_all` family,
and OLMo-3-32B 131k batch-topk were probed during bring-up (`out/probe_*.json`,
`out/validate_*.json`) before settling on gemma-3-27b-it resid_post L40. Not
revisited since the token-level protocol landed.

---

**Still central, for contrast:** the judge pool (necessary, §2), ratio screen ρ,
vote k (k=2→4 = +.03–.13 F2_all via 60–80% FP cuts), windowing (+.054 F2_all @16k,
Rc2 gains at both widths), and the train/test G-split discipline itself.
