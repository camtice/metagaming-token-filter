#import "@local/geodesic-report:0.1.0": report, design-box, appendices, pct, pp, num

#let R = json("results.json")

#let fixed(x, digits: 3) = if x == none { "—" } else {
  let str-val = str(calc.round(float(x), digits: digits))
  let parts = str-val.split(".")
  let frac = if parts.len() > 1 { parts.at(1) } else { "" }
  parts.at(0) + "." + frac + "0" * (digits - frac.len())
}

#show: doc => report(
  title: "Selection-rule ablations for the metagaming token filter",
  date: "2026-08-29",
  doc,
)

#design-box[
  *Experiment design and motivation.* The metagaming token filter labels
  pretraining tokens with a rule built from SAE features in three stages: a
  candidate pool (features an LLM judge marked as oversight-related), a ratio
  screen (keep pool features whose above-threshold firing rate is at least
  #{sym.rho} times higher on forget training text than on retain controls),
  and a token rule (a token is seeded when k kept features co-fire above
  threshold, and labels then extend to nearby tokens by a boundary rule).
  Before freezing a feature set for deployment, this report measures how the
  components move held-out performance: (1) whether restricting the screen to
  the judged pool changes token-level discrimination relative to screening
  the full dictionary, compared both as ROC curves of a continuous token
  score and as F2 frontiers at matched false-positive rates; (2) how the vote
  size k trades recall against control false positives; (3) how the boundary
  rules that extend labels beyond seed tokens — none, growth over
  above-threshold tokens, and the paper's growth over any active token —
  interact with k; (4) how the 16k-width and 65k-width SAEs at the same layer
  compare under the same pipeline. All cells are evaluated on held-out
  documents under a train/test split whose training side contains no
  human-labelled text. Full configuration is given in
  #link(<app-details>)[the details appendix].
]

#figure(
  image("build/figs/fig_roc.pdf", width: 100%),
  caption: [
    *Token-level ROC curves, judged pool against full dictionary* (65k SAE).
    The score is the summed above-threshold excess activation over the kept
    set (see Methods); each curve sweeps its decision threshold. True
    positive rate over (A) all held-out forget tokens, (B) human-labelled
    span tokens, (C) tokens of the held-out synthetic category; the false
    positive rate is measured on held-out control tokens throughout. Solid
    segments cover the thresholded region (tokens with a positive score);
    the dotted continuation to (1, 1) covers the tokens with score zero — no
    kept feature above threshold — which are tied and ranked arbitrarily.
    The x axis is logarithmic. AUC values with clustered SEs (ties receive
    half credit) are tabulated in #link(<app-table>)[the appendix].
  ],
) <fig-roc>

= Methods

Every cell applies the same pipeline to the same frozen activation harvests:
Gemma Scope 2 SAE activations of the 65k-width and 16k-width residual-stream
SAEs at layer 40 of gemma-3-27b-it, over the TEST-v5 corpus (span-annotated
rollouts, reviewed-clean chat documents, synthetic dense forget documents in
five categories, and natural-text controls). Documents are split into a
training half and a held-out half; per-feature firing statistics, screen
ratios and activation thresholds (mean + 4 SD per feature) are computed on
the training half only. The training forget side is synthetic dense text from
three categories; one synthetic category and all human-span-annotated
documents appear only on the held-out side.

Two complementary views are reported. The *ROC view* (@fig-roc, @fig-width)
scores every token with a continuous statistic — the sum over kept features
of the activation excess above the feature's own threshold, in units of the
feature's SD — and sweeps the decision threshold, so feature sets are
compared independently of any single operating point; AUC is computed by the
Mann–Whitney statistic with tie correction, and boundary rules are not
applied in this view. The *rule view* (@fig-judge, @fig-vote,
@fig-boundary-k) evaluates the deployed labelling rule (vote k plus a
boundary rule) and reports F2 decomposed as F2#sub[in] (held-out
in-distribution material), F2#sub[out] (the fully held-out synthetic
category), and F2#sub[all] (both), with span recall, held-out-category
recall, and the control false-positive rate. Uncertainty on every estimate
is a doc-clustered bootstrap SE (400 resamples for rule metrics, 200 for
AUC, stratified by document role).

The ablations:

- *Judge inclusion* (@fig-roc, @fig-judge): the ratio screen is applied
  either to the 3,798-feature judged pool or to all 65,536 dictionary
  features. The ROC view compares the screened sets and both unscreened
  pools; the rule view sweeps #{sym.rho} at k = 4 so the two arms cover the
  same range of control false-positive rates.
- *Vote k* (@fig-vote): k #sym.in {1, 2, 3, 4, 6, 8} at screen #{sym.rho}
  #sym.gt.eq 5 and #{sym.rho} #sym.gt.eq 10, paper boundary rule. The vote
  changes token labelling only; the kept feature set is unchanged.
- *Boundary rule × k* (@fig-boundary-k): three rules — seeds only; growth
  restricted to contiguous tokens where at least one kept feature is above
  its mean + 4 SD threshold; and the paper rule, growth across contiguous
  tokens where any kept feature is active at any level — each crossed with
  the full k grid at both screen thresholds.
- *SAE width* (@fig-width): the judged pool of each width under the same
  screen sweep at k = 4, and the ROC comparison of the two widths'
  #{sym.rho} #sym.gt.eq 10 sets. Both widths ship with a comparable code
  density (approximately 60 active latents per token).

= Results

*Judge inclusion.* In the ROC view the two screened sets are close on
human-labelled spans (@fig-roc\B) and separated on the held-out synthetic
category, where the full-dictionary set traces the higher curve
(@fig-roc\C); over all forget material combined the full-dictionary set has
the higher AUC (@fig-roc\A; exact values in #link(<app-table>)[the
appendix]). Removing the screen collapses both pools — the unscreened full
dictionary scores near chance on every population — so the ratio screen
carries most of the discrimination in either arm. The F2 frontier view at
matched false-positive rates (@fig-judge) shows the same structure:
overlapping F2#sub[in] curves, separation concentrated in F2#sub[out]. At
equal false-positive rates the judged pool reaches its frontier with roughly
three to five times fewer kept features.

*Vote k.* Raising k lowers the control false-positive rate by roughly a
factor of two to three per step (@fig-vote\C) while span recall declines
gradually (@fig-vote\B). F2#sub[all] peaks at k = 6 for #{sym.rho} #sym.gt.eq
5 and at k = 4 for #{sym.rho} #sym.gt.eq 10 (@fig-vote\A); the
#{sym.rho} #sym.gt.eq 5, k = 6 cell attains the highest F2#sub[all] of any
judged-pool cell measured in this report.

*Boundary rule × k.* Without any growth stage, F2#sub[all] falls
monotonically once k exceeds 2 and the seeds recover at most half of the
span tokens (@fig-boundary-k\A, B). Both growth rules restore recall and
hold their F2#sub[all] nearly flat across k #sym.gt.eq 3; the strict rule —
growth only across tokens with at least one kept feature above its 4 SD
threshold — tracks the paper rule within one clustered SE at every (ρ, k)
cell, at the same false-positive rate (@fig-boundary-k\C). In the k-swept
operating view (@fig-boundary-k\C) the growth rules dominate seeds-only at
every matched false-positive rate.

*SAE width.* The 65k frontier sits above the 16k frontier on F2#sub[all]
across the low-false-positive range, and the 16k frontier peaks earlier (at
#{sym.rho} #sym.gt.eq 6) and lower (@fig-width\A). The ROC comparison of the
two #{sym.rho} #sym.gt.eq 10 sets shows equal span curves (@fig-width\B) and
a 65k advantage on the held-out category (@fig-width\C).

#figure(
  image("build/figs/fig_judge.pdf", width: 100%),
  caption: [
    *Held-out F2 against control false-positive rate, judged pool against
    full dictionary* (65k SAE). Screen threshold #{sym.rho} is swept along
    each curve (#{sym.rho} 1–30 for the judged pool, 5–75 for the full
    dictionary); vote k = 4 and the paper boundary rule are fixed. (A) F2
    over all held-out forget material. (B) F2 over the in-distribution
    portion. (C) F2 over the held-out synthetic category only. Bands are ±1
    doc-clustered bootstrap SE; the x axis is logarithmic.
  ],
) <fig-judge>

#figure(
  image("build/figs/fig_vote.pdf", width: 100%),
  caption: [
    *Vote size k at two screen thresholds.* Judged pool, paper boundary rule.
    (A) F2 over all held-out forget material. (B) Recall on human-labelled
    span tokens. (C) Control false-positive rate, logarithmic y axis.
    Whiskers are ±1 doc-clustered bootstrap SE.
  ],
) <fig-vote>

#figure(
  image("build/figs/fig_boundary_k.pdf", width: 100%),
  caption: [
    *Boundary rules crossed with the vote k.* Judged pool. Rules: seeds only;
    growth restricted to above-threshold tokens; the paper rule (growth over
    any active token). (A) F2 over all held-out forget material against k at
    #{sym.rho} #sym.gt.eq 5. (B) The same at #{sym.rho} #sym.gt.eq 10.
    (C) Span recall against control false-positive rate as k sweeps 1–8 at
    #{sym.rho} #sym.gt.eq 10; the x axis is logarithmic. Whiskers are ±1
    doc-clustered bootstrap SE.
  ],
) <fig-boundary-k>

#figure(
  image("build/figs/fig_width.pdf", width: 100%),
  caption: [
    *16k-width against 65k-width SAE, judged pools.* (A) F2 over all held-out
    forget material against control false-positive rate, screen #{sym.rho}
    swept at k = 4 with the paper boundary rule; bands are ±1 doc-clustered
    bootstrap SE. (B, C) ROC curves of the continuous token score for each
    width's #{sym.rho} #sym.gt.eq 10 set, on human-labelled spans and on the
    held-out category; x axes are logarithmic.
  ],
) <fig-width>

#show: appendices

= Replication values <app-table>

Exact values for cells named in the main text. Rule cells (upper block):
kept features n, F2#sub[in], F2#sub[out], F2#sub[all], span recall,
held-out-category recall, control false-positive rate. AUC cells (lower
block): full AUC of the continuous score per population, with ±1
doc-clustered bootstrap SE on the span and all-forget columns.

#let row(label, r) = (
  label, str(r.n), fixed(r.F2in), fixed(r.F2out), fixed(r.F2all),
  fixed(r.Rspan), fixed(r.Rc2), fixed(r.fp, digits: 4),
)
#table(
  columns: 8,
  align: (left, right, right, right, right, right, right, right),
  table.header([cell], [n], [F2#sub[in]], [F2#sub[out]], [F2#sub[all]],
               [R#sub[span]], [R#sub[c2]], [fp]),
  ..row([judge #{sym.rho}≥10 k=4 (reference)], R.judge_frontier_k4.at("judge rho=10")),
  ..row([judge #{sym.rho}≥5 k=4], R.judge_frontier_k4.at("judge rho=5")),
  ..row([judge #{sym.rho}≥5 k=6], R.vote_k.at("rho=5 k=6")),
  ..row([full dict. #{sym.rho}≥35 k=4], R.judge_frontier_k4.at("all rho=35")),
  ..row([full dict. #{sym.rho}≥20 k=4], R.judge_frontier_k4.at("all rho=20")),
  ..row([boundary: seeds only (#{sym.rho}≥10 k=4)], R.boundary_k.at("rho=10 k=4 seed")),
  ..row([boundary: growth above-thr. (#{sym.rho}≥10 k=4)], R.boundary_k.at("rho=10 k=4 window_ab")),
  ..row([boundary: paper rule (#{sym.rho}≥10 k=4)], R.boundary_k.at("rho=10 k=4 window")),
  ..row([16k judge #{sym.rho}≥6 k=4 (16k peak)], R.width_16k_frontier_k4.at("judge16 rho=6")),
  ..row([16k judge #{sym.rho}≥10 k=4], R.width_16k_frontier_k4.at("judge16 rho=10")),
)

#let arow(label, r) = (
  label, str(r.n),
  fixed(r.cont_auc_span) + " ± " + fixed(r.se_cont_auc.span),
  fixed(r.cont_auc_c2),
  fixed(r.cont_auc_all_forget) + " ± " + fixed(r.se_cont_auc.all_forget),
  fixed(r.cont_pauc015_all_forget),
)
#table(
  columns: 6,
  align: (left, right, right, right, right, right),
  table.header([set (65k unless noted)], [n], [AUC span], [AUC c2],
               [AUC all forget], [pAUC#sub[1.5%] all]),
  ..arow([judge #{sym.rho}≥10], R.roc_65k.at("judge rho>=10")),
  ..arow([judge #{sym.rho}≥5], R.roc_65k.at("judge rho>=5")),
  ..arow([judge, no screen], R.roc_65k.at("judge pool (no screen)")),
  ..arow([full dict. #{sym.rho}≥35], R.roc_65k.at("all rho>=35")),
  ..arow([full dict. #{sym.rho}≥20], R.roc_65k.at("all rho>=20")),
  ..arow([full dict., no screen], R.roc_65k.at("full dictionary (no screen)")),
  ..if "roc_16k" in R { arow([16k judge #{sym.rho}≥10], R.roc_16k.at("judge rho>=10")) } else { () },
)

= Experimental details <app-details>

- Base model: #R.meta.model; SAEs: google/gemma-scope-2-27b-it resid_post
  layer 40, width_65k_l0_medium and width_16k_l0_medium (JumpReLU;
  approximately 60 active latents per token on this corpus at either width).
- Activation harvests: out/harvest_test_v5_65k.npz and
  out/harvest_test_v5.npz — BOS-prefixed 1,023-token windows with 64-token
  context-only overlap over every TEST-v5 document
  (#num(R.meta.corpus_tokens) tokens).
- Candidate pools: haiku_v8_65k (3,798 judged-forget latents) and
  haiku_v8_16k (921) — features an LLM judge (claude-haiku, prompt v8)
  marked oversight-related from their Neuronpedia captions.
- Split: #R.meta.split — document-level halves within TEST-v5. Training
  side: 45 synthetic dense documents (three categories) and 1,312 controls.
  Held-out side: 25 span-annotated rollouts (6,577 positive tokens), 20
  reviewed-clean chat documents, 45 dense documents of training categories,
  30 documents of the held-out category, 1,313 controls.
- Ratio screen: Jeffreys-smoothed ratio of above-threshold firing rates,
  training forget over training controls; activation thresholds are
  per-feature mean + 4 SD over the scored corpus.
- Continuous ROC score: #sym.sum over kept features of
  max(activation − threshold, 0) / SD; AUC by Mann–Whitney with tie
  correction; pAUC is the standardised partial AUC over FPR #sym.in
  [0, 0.015].
- Uncertainty: doc-clustered bootstrap stratified by document role, fixed
  seed; 400 resamples (rule metrics), 200 (AUC).

= References <app-refs>

- Rathi & Radford, #link("https://arxiv.org/abs/2601.21571")[arXiv:2601.21571]
  — source of the screen/vote/boundary rule family and its defaults (k = 2,
  mean + 4 SD, adjacency growth).
- Corpus and annotations: HF dataset `camgeodesic/metagaming-labeling`
  (private) at the revision pinned in the split manifest.
- Code and aggregate results:
  #link("https://github.com/camtice/metagaming-token-filter")[github.com/camtice/metagaming-token-filter].
- SAE weights: #link("https://huggingface.co/google/gemma-scope-2-27b-it")[google/gemma-scope-2-27b-it].

= Reproduction <app-repro>

- Split manifest: `data/splits/split_manifest_v5.json` and
  `data/splits/gen_split_20260815.json` in the code repository above.
- Candidate pools: `data/candidate_sets/haiku_fable_forget_latents_v1.json`
  (sets `haiku_v8_65k`, `haiku_v8_16k`).
- Aggregate cells behind every figure: `out/ablation_report_empirics.json`,
  `out/ablation_report_empirics2.json`, `out/roc_auc_sets_65k.json` and
  `out/roc_auc_sets_16k.json` in the code repository.
- The activation harvests regenerate from the corpus and SAEs named in
  #link(<app-details>)[the details appendix] with one GPU pass per width.
