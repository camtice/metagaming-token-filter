# Simple screening rules vs direct optimization — TEST v5, all pools, both widths
2026-08-15 · `scripts/screen_rules_sweep.py` · data `out/screen_rules_{fable_16k,h6_16k,h8_16k,h6_65k,h8_65k}.json`

## Question

Can a one-knob screen over a judge pool — no combinatorial search — get close to the
directly optimized sets? Tested six rules on five pools under protocol v5
(main forget = spans + dense c1–c4; constraint fp_ctl ≤ .10) on the **new TEST v5**
(manifest sha `0f6f83c7`: 2,820 docs, controls now 2,625 incl. the 08-15 refresh half).

The rules (each keeps every latent the rule admits; full Rathi rule evaluated on the kept set):

| rule | keep latents with… | labels needed |
|---|---|---|
| R1 | control-token fire rate ≤ τ *(user's proposal)* | control corpus only |
| R2 | corpus-wide fire rate ≤ τ | none (fully label-free) |
| R3 | forget-rate / control-rate ratio ≥ ρ | forget + control |
| R4 | stopword/punct share of firings ≤ σ | none (surface only) |
| R5 | Paulo et al. embedding score ≥ s (Rathi & Radford use s = 0.9) | captions + SAE examples |
| R6 | R5 at 0.9, then R1 on survivors *(Rathi's actual pipeline)* | both |

Embedding scores per the Paulo et al. recipe: MiniLM embed of the Neuronpedia caption
vs 8 top-activating + 8 random contexts from `examples.safetensors`, AUC separation
(`scripts/embed_score_captions.py`, `out/embed_scores_{16k,65k}.json`).

## Headline result

**One simple rule — R3, the forget/control fire-rate ratio — recovers 98–100% of the
directly-optimized F2 in every pool at both widths.** The user's R1 control-cap rule
alone gives ~85–90%. Everything label-free or caption-based trails badly. Best
feasible (fp ≤ .10) main-F2 by rule, with the optimized hill-climb sets as the bar:

| pool | optimized | **R3 ratio** | R1 ctl-cap | R4 stop-frac | R6 embed→ctl | R5 embed | R2 corpus |
|---|---|---|---|---|---|---|---|
| fable 16k (649) | .882 (D2) | **.873** | .805 | .802 | .782 | .665 | .581 |
| haiku v6 16k (724) | .889 | **.875** | .777 | .712 | .694 | .470 | .383 |
| haiku v8 16k (921) | — | **.889** | .790 | .692 | .701 | .514 | .383 |
| haiku v6 65k (3,043) | .916 | **.900** | .747 | .833 | .737 | .657 | .421 |
| haiku v8 65k (3,798) | — | **.906** | .774 | .722 | .763 | .639 | .425 |

(Optimized rows re-scored on TEST v5 by the official scorer. Note the frozen h6-16k/h6-65k
optima now exceed .10 fp on v5 — .132/.118 — because the new fineweb half runs hotter for
them; the rule sweeps here are v5-native and respect the cap. D2 sits at .102, P2 at .060.)

## Reading the rules

**R3 (ratio) is the finding.** One statistic per latent — smoothed
(forget fire-rate)/(control fire-rate) — with one threshold, no search, lands within
.009–.016 F2 of thousands-of-evaluations hill-climbing, and at 65k v8 it reaches **.906
with recall .930**, the best recall of anything feasible we've measured. It's also the
paper-friendly story: "keep features that fire ≥ρ× more on forget than on retain text."
The threshold is *scale-free across widths* (ρ ≈ .13–.20 everywhere: .133 fable, .130/.146
haiku 16k, .152/.200 haiku 65k) — no 65k rescaling needed, unlike the rate caps.

**R1 (the proposed control cap) is directionally right but leaves 8–17 points of F2.**
Its failure mode is one-sided: it removes flooders but keeps useless quiet latents,
so at the cap that reaches fp ≤ .10 it has also silently discarded strong forget-carriers
that fire moderately on controls. It needs the forget side of the ledger — which is
exactly what R3 adds. Note its τ *does* need width rescaling (feasible τ ≈ .0017 at 16k
vs .0005 at 65k — sparser latents at 65k fire less everywhere).

**The Rathi/Paulo embedding screen (R5/R6) transfers poorly to this task.** At the
paper's 0.9 floor it keeps 35–55% of each pool, but the survivors still flood controls
(fp .18–.34) and lose recall; the best fp-feasible combo (R6) tops out at .70–.78 F2 —
*worse than the plain control cap alone*. Two reasons visible in the data: (a) the score
measures caption *faithfulness*, not task relevance — a faithfully-captioned "list
separators" feature passes at 1.0 and still floods; (b) our sensitive registers are absent
from the SAE's example corpus, so exactly the frontier/chat-relevant features get
mediocre scores. Caveats: our reproduction approximates theirs (MiniLM embedder, AUC
scoring, their embedder/protocol unspecified), and captions come from gemini-flash-lite.

**Label-free rules can't do it.** R2 (corpus rate) is the worst everywhere (.38–.58) —
overall firing rate barely correlates with forget-specificity. R4 (stopword share) is a
surprisingly decent register screen at 65k (.833 on v6) but unstable across pools.

## Width and pool comparisons

- **Width uplift is real and survives the simple rule**: R3 gains +.025 (v6: .875→.900)
  and +.017 (v8: .889→.906) going 16k→65k — nearly identical to the +.028 uplift under
  full optimization. The 65k dictionary's advantage is a property of the feature space,
  not of the optimizer.
- **Pools matter far less than widths or rules**: at 16k, fable/v6/v8 all land within
  .873–.889 under R3; at 65k, v6/v8 within .900–.906. The judge prompt (and judge model)
  choice is nearly irrelevant once a decent screen is applied — consistent with the
  earlier finding that D2 ≈ h6-16k under full optimization.
- **Bigger pools help the rule**: v8 ≥ v6 at both widths under R3 (more recall carriers
  to keep after screening), the opposite of the raw-pool ordering by FP.

## Practical recommendation

For the paper and for the collaborator's non-SAE baseline: **screen by the R3 ratio with
ρ ≈ 0.15** (report the small sensitivity band .13–.20). It is one sentence to justify,
needs only the forget/retain corpora (no captions, no embeddings, no search), transfers
across SAE widths without rescaling, and matches direct optimization to ~1 point of F2.
Direct optimization's remaining edge (~.01) is within the adaptive-overfitting noise we
estimated for TEST-tuned sets, so the ratio rule is arguably the *more* honest headline.

Sealed validation remains unspent; a natural single spend is the R3-screened h8_65k set
(K=993, ρ=.20 → F2m .906, Rm .930, fp .099) or the optimized h6-65k trim, once you pick.

## Files

Sweeps `out/screen_rules_*.json` (full threshold curves per rule) · embed scores
`out/embed_scores_{16k,65k}.json` · v5 reference scores `out/score_harvest_test_v5*` ·
manifest `data/splits/split_manifest_v5.json` (sha 0f6f83c7) · harvests
`out/harvest_test_v5{,_65k}.npz`, `out/harvest_valA_v5.npz` (VAL-A 16k merged, unscored).
