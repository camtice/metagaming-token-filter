# Qualitative investigation of the candidate rules — full program
2026-08-15→16 overnight · reader artifact `Rule Reader` (0d18f342) · data `out/quali_*.json`, `out/gen_study_*_s*.json`

## What ran

Eight hypothesis tests (five CPU, three needing fresh GPU harvests of ~8M tokens across
four corpora), a token-level error anatomy, a cross-width feature census, and an
interactive reader over 109 documents. All selection statistics G-TRAIN-only (dense
c1/c3/c4 halves + control half); everything examined is held-out or unlabeled. Sealed
VAL-A/VAL-B untouched throughout. Rules compared: **A** R3-65k ρ=5 · **B** R3-65k
ρ=1/seed≥4 · **C** R3-16k ρ=5 · **D** R1-16k τ=.1% · **E** D2 (optimized) · **F** full
65k pool.

## The mechanism findings (what makes the rule work, and where it fails)

**1. Misses are doc-concentrated selection gaps, not coverage gaps.** The worst span
docs (umwp-unanswerable math-trick rollouts, recall .14–.32) have pool features firing
above threshold on 91–98% of their missed tokens — the ratio screen dropped the carriers
(math/trick-question features have no support in the dense training corpus). Partial
rescue exists (ρ=1/seed≥4 lifts the worst doc .14→.34) but the register is fundamentally
under-supervised: **dense synthetic docs contain no math-trick chat, so no ratio
threshold can justify keeping its carriers.** The durable fix is labeled (or dense-style)
math-trick data, not rule tuning.

**2. The Rathi rule's load-bearing parts are not the ones the paper emphasizes.**
Ablations (fixed rule-A set): the mean+4SD choice is irrelevant (3SD ≈ 4SD ≈ 5SD);
windowing is essential (+.09–.13 span recall, and +.21 held-out-domain recall at 16k);
and the seed count ≥2 is suboptimal — **seed≥3–4 halves control FP at ~2pp recall
cost**. The best frontier we found anywhere: *loosen the screen, tighten the vote*
(ρ=1–2, seed≥4) — at matched FP it beats ρ-only operating points by 2–7pp span recall
(ρ=1/seed≥4: span .833, c2 .924, fp .119 at 65k).

**3. What 65k actually adds**: finer safety/refusal/deception detectors
(refusal-to-demonstrate-harm, safety-guidelines, deception-and-falsehood) that carry
span tokens 16k misses. Net span gain is modest (+393/−290 tokens vs 16k) — but the two
widths flood *different* control tokens (FP Jaccard .30), which suggests a
width-ensemble (flag only where both agree) as an unexplored precision lever.

**4. FP structure**: ~half of control FP comes from window expansion rather than seeds
for every rule; swe_tests is the worst config for all rules; wildchat is specifically
R1's blind spot (.18). Several FP-seeding features are caption-unfaithful (a feature
captioned "robust invariant risk classification" with train ratio 7,224 fires mostly on
`, the of .` in ordinary text) — captions cannot be trusted for selection or audit;
surface profiles can.

**5. Rules that score alike flag different tokens.** Token-level Jaccard between R3-16k
and D2 is only .54 (R1 vs D2: .33). Aggregate scoreboards hide substantial disagreement —
the reader's diff mode makes this visible token by token.

## The real-world validation (fresh GPU harvests, doc-level gold, no token labels)

**Westover (real Alignment-Forum posts, human-split): AUC .76** for rules A/B —
the strongest external validation in the project. The rule's doc-level flag rate
separates metagame-labeled posts from general posts far above chance, on a corpus and
register it never saw.

**Kimi trick-vs-clean differential: directionally confirmed, register-dependent.**
Mean pairwise AUC .61–.65 for the R3 family vs **chance (.51–.52) for R1 and the full
pool** — the ratio screen is what creates discrimination. Per family: umwp .79–.85
(trick arm flagged 2× more), if_conflict .63–.67, squad2 .62–.68, hawthorne .53–.68.
One inversion: **falseqa (true-premise answers flagged *more* than false-premise,
AUC .36–.45)** — flagged for follow-up; and hawthorne rollouts run very hot in absolute
terms (39–54% of tokens flagged) because refusal/safety features fire wall-to-wall on
harm-adjacent chat — a scope question (is refusal-reasoning forget?) more than an error.

**Needham eval-transcripts: AUC ≈ .44–.50 (chance), r(human_p) slightly negative — an
instructive null.** The rules do not distinguish docs that *are* eval transcripts from
ones that aren't. On reflection this is the correct behavior for a filter that targets
*discussion/awareness of evaluation* rather than *being evaluation content* — but it
sharpens a scope question the c1 definition currently blurs: benchmark transcripts
themselves are flagged at only ~21–32% by rule A. If "content of evals" should be
removed wholesale, a different mechanism (doc-level provenance) is needed; token-level
metagaming features won't do it.

## Robustness checks that all passed

Split-seed stability (3 seeds: rankings unchanged, ±.02); threshold provenance
(deployment-corpus vs TEST statistics: recall shifts .001); annotation-category
breakdown (uninformative — the export's categories are ~all c1).

## The reader artifact

**https://claude.ai/code/artifact/0d18f342-fca7-492c-b8db-312e3d2a592c** — 109 docs in
8 groups (human-labeled rollouts, held-out c2, kimi trick/clean arms, dense, clean chat,
highest-FP controls, random controls). Click one rule to highlight, two to diff
(blue/orange/purple), or agreement mode (opacity = how many of the 6 rules flag the
token). Human spans underlined; per-doc flag shares and span recall shown per rule; each
rule's top FP-seeding features (with captions) in the details panel. Private; embeds
rollout text — share deliberately.

## Bottom line for the classifier decision

The ratio family is validated well beyond its TEST numbers: it generalizes across
domains (c2), transfers to real rollouts, discriminates trick-vs-clean elicitations
where R1 and the raw pool sit at chance, and separates real-world AF posts at AUC .76.
Its two live weaknesses are data problems, not rule problems: the math-trick chat
register (unsupervised by dense docs) and the refusal/harm-chat scope question. The rule
I would freeze for the sealed validation: **65k haiku pool, ρ=2, seed≥4** (span .789 /
c2 .907 / fp .067 on held-out) or its ρ=1 sibling if recall is prioritized — with
seed-count now a documented, principled part of the recipe.

## Inventory

Scripts: `quali_error_anatomy.py`, `quali_token_dive.py`, `quali_width_census.py`,
`quali_rathi_ablation.py`, `quali_docgold_analysis.py`, `build_reader_data.py`,
`build_reader_artifact.py`, seeded gen-study variant. Harvests:
`out/harvest_qkimi_{16k,65k}` (2,191 kimi chat rollouts), `out/harvest_qgold_{16k,65k}`
(693 westover + 906 needham). Results: `out/quali_*.json`. Corpora:
`data/quali_kimi_chat.jsonl`, `data/quali_docgold.jsonl` (TEST/VAL members excluded).

## Addendum (second iteration round)

**X1 — agentic register (498 fresh rollouts, 3 arms).** Weak but consistent positive
differential: notice > baseline for all four rules tested (AUC .55–.61), scratchpad ≈
baseline. Absolute rates are very high (65k rules flag 39–57% of agentic tokens; 16k
19–29%) because the agentic-misalignment *scenario text itself* is oversight-topical in
every arm — content dominates awareness in this register. For a pretraining filter this
is arguably correct behavior (the whole scenario is c1/c2 material); as an
awareness-detector the token rule is weak in-context here.

**X2 — width-ensemble: refuted.** Flagging only where the 16k AND 65k rules agree lands
on the single-width frontier, not beyond it (ρ=1/seed4 ensemble: span .775/fp .065 vs
single-65k ρ=2/seed4 .789/.067). The low FP-Jaccard between widths does not convert into
a usable precision lever. Dropped.

**X3 — both kimi anomalies explained.** falseqa's inversion is a c3-scope effect: the
true-premise arm is education/tutoring-saturated (tutoring-services, courses-and-degrees
features), which is in-scope forget by definition — not missed awareness; the
false-premise arm carries proportionally more awareness features. hawthorne's 39–54%
heat is refusal/safety/deception features on harm-adjacent chat; notably the
"test in isolation" feature fires only in the *fake* arms — the eval-awareness signal
working. Open scope question for the owner: is safety-refusal reasoning forget?

**Reader updated** with agentic notice/baseline examples (same URL).
