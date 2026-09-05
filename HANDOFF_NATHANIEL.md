# Handoff resources — category-split rebuild (Nathaniel)

Companion to the "Nathaniel Filtering Handover" plan: refine the category regex
lists (AI1/AI2/AI3/H1/M1), build a category-based train/test split, compare
SAE-feature selection against directly trained BERT-style classifiers with
round-robin generalisation, then train the final model on all labelled data.
This file collects every resource the rebuild needs, most-current versions
only. The plan assumes a fresh repository; nothing here needs to be salvaged
as code — treat this repo as the reference implementation and results record.

## 1. Category taxonomy and regex lists

The new taxonomy maps onto the existing 5-category blocklist like so:

| new | scope | existing category | notes |
|---|---|---|---|
| AI1 | AI capability evals / oversight (benchmarks, leaderboards, rubrics) | cat 1 `1_capability_evals_oversight` / dense c1 | |
| AI2 | AI safety oversight (alignment evals, monitoring, control) | cat 2 `2_ai_safety_oversight` / dense c2 | was the held-out generalisation category in all prior results |
| AI3 | AI training pipeline (RL mechanisms, rewards, GRPO, envs) | cat 4 `4_ai_training_pipeline` / dense c4 | |
| H1 | human oversight (grading, exams, reviews, surveillance) | cat 3 `3_human_oversight_testing` / dense c3 | consistently the *hardest* dense category (70% coverage at 1% budget vs ≥95% for the others) |
| M1 | non-AI evaluations (machinery inspection, software QA) | partially cat 5 `5_swe_tests` / dense c5 | new category; note c5 was ruled RETAIN-scope 2026-08-12 (manifest v5 rules block) — M1 broadens it and reopens that scoping decision deliberately |

- **The current term lists (the thing to refine)**:
  `data/dataset_configs/metagaming_labeling/assets/pools/blocklist_v3.json` —
  338 term groups across the 5 categories, verbatim from dataset-builder.
  Original regex construction: `scripts/regex_v3.py`.
- The term-hits screening kernel that consumes it is documented inside each
  `data/dataset_configs/metagaming_labeling/ctl_*.yaml` (the `term_hits`
  transform steps).

## 2. Data — everything lives in one HF dataset

**`camgeodesic/metagaming-labeling`** (private; request access) — every config
is one corpus; pin revisions, never read HEAD blind.

- **Pinned revisions used by all published results**: `f7ae7700` (main corpus)
  and `b3798ce5` (fresh-controls tranche). The split manifests record both.
- **Forget synthetic**: `dense_c{1..5}_*` (60 docs each, Opus-generated,
  per-doc `topic`/`doc_type`/`gen_model` metadata) and the new larger
  `dense_c{1..5}_*_pooled` variants; `dense_retain_far` (60 same-register,
  far-topic negatives — the register-leakage diagnostic, so far unused).
  ⚠ Gap: the dense **generation prompts/scripts are not yet uploaded**
  anywhere shared — they live in Cam's local `sae-probe` project. Get them
  before regenerating or extending categories (M1 needs them).
- **Real labelled text**: `annotations` (153 hand-span-labelled docs,
  whitespace-word indices, 0-based end-exclusive — the gold coordinate
  convention), `frontier_metagame_rollouts` (57 frontier CoTs, fully span
  labelled, SEALED — see §5), `kimi_metagame_rollouts`,
  `kimi_agentic_rollouts`, `low_quality_synthetic_annotated` (24 docs),
  `westover_*`, `needham_*` (doc-level gold).
- **Controls**: `ctl_fineweb`, `ctl_wildchat`, `ctl_dolci_instruct`,
  `ctl_dolci_think`, `ctl_swe` / `ctl_swe_tests` (a designed matched pair —
  disjoint halves of one draw differing only in the cat-5 term bound).
  Build configs + purity gates: `data/dataset_configs/metagaming_labeling/`
  (this repo) and the private
  [GeodesicResearch/dataset-builder](https://github.com/GeodesicResearch/dataset-builder)
  (`configs/metagaming_labeling/`); every HF config directory also carries a
  machine-readable `_provenance.json`.

## 3. Reference results the rebuild is measured against

- **This repo, `main`** — the full SAE-selection record: `README.md` (recipe
  + guardrails), `report/sae-selection-ablations.pdf` (the ablation report:
  judge inclusion via ROC + FP-matched F2 frontiers, vote k, boundary rules,
  16k-vs-65k width; every estimate with doc-clustered bootstrap SEs;
  `report/results.json` is the single source of truth), dated studies in
  `notes/reports/` (`peripheral_variables_2026-08-29.md` summarises every
  settled design variable), aggregates in `out/`.
- **Frozen SAE reference bars**:
  `data/candidate_sets/frozen_ladder_judge65k_v1.json` and
  `frozen_ladder_alldict65k_v1.json` — nested feature sets at ~1/3/5/10%
  token-removal budgets (latent ids + full recipe + G-TEST metrics in the
  file meta). Any new classifier should be compared at these budgets: at
  matched removal, span recall runs .67→.83 and the two families tie on real
  hand-labelled text.
- **Interactive views** (HTML copies in `artifacts_html/`): collaborator
  handbook (`project_overview.html`), the quantitative atlas
  (`gen_atlas.html`), frontrunner sets.
- **The paper being replicated**: Rathi & Radford,
  [arXiv:2601.21571](https://arxiv.org/abs/2601.21571). §5.1 is the SAE
  labelling pipeline; §5.2 is directly the "train BERT directly" arm of the
  comparison (ModernBERT baseline F1 .794 → task-specific biLM probes .856
  val / .894 test — small task-specific bidirectional base models beat large
  general ones, probes over frozen reps beat finetuning for robustness).
- **The prompt-classifier arm** (third baseline, no SAE): Puria's
  `metagaming_filtering` repo — `utils/document_tagging/` (three
  interchangeable LLM span-labelling methods, all emitting the gold word
  coordinates; its README is the coordinate-convention reference) and
  `utils/prompt_iter/` (train/val/test splits seed 20260827, six prompt
  iterations, val F2_gold up to .90).

## 4. SAE stack, only as needed for the comparison arm

- SAEs: `google/gemma-scope-2-27b-it`, resid_post layer 40, `width_65k_l0_medium`
  (deployment width) and `width_16k_l0_medium`; base model
  `unsloth/gemma-3-27b-it`. 262k exists but only as `l0_small` — measured
  unusable with co-firing rules (F2 .58; L0 confound). Registry + loader:
  `scripts/sae_local.py`.
- Harvest recipe + measured gotchas (do not rediscover): prepend BOS; bf16
  (fp16 NaNs); Gemma Scope hook `layers[L].output` = HF `hidden_states[L+1]`;
  1023-token windows with 64-token context-only overlap
  (`scripts/harvest_acts.py`). Existing harvests on the cluster:
  `~cwtice.a5k/sae-exploration/out/harvest_test_v5*.npz` (16k/65k/262k over
  the full TEST corpus) — reusable, no GPU pass needed for TEST-v5 docs.
- Thresholds (mean+4SD) are computed per scored corpus and are **portable**:
  control-only vs full-corpus moments select identical feature sets
  (fp Δ≤0.2pp, measured).

## 5. Protocol constraints that survive the rebuild

1. **The sealed validation sets stay sealed**, whatever else is rebuilt.
   `data/splits/split_manifest_v5.json` (sha256 `0f6f83c7…`) defines VAL-A
   (2,814 docs, in-distribution mirror) and VAL-B (the 57 frontier rollouts);
   ids only, never train/tune/select on them — including their rows in the
   `annotations` config. `data/splits/validation_access_log.jsonl` has
   exactly 3 entries (the original blocklist baseline); keep appending on any
   access. These are the only corpora that make the final numbers credible —
   burning them in the rebuild forfeits the endgame measurement.
2. **Never compute selection statistics on the corpus you evaluate** —
   same-corpus statistics inflate held-out F2 by +.02–.04 (measured).
3. **Internal metrics overweight synthetic-register transfer.** The dense
   corpus has no AI-topical-but-not-oversight negatives, so anything selected
   against it can key on register/topic: a no-judge SAE selection looked best
   internally yet flagged 83–90% of ordinary AI-forum posts. Validate on real
   text (westover/kimi differential behaviour, `dense_retain_far` FP) before
   believing a generalisation number. This is the main trap the round-robin
   design should be built to catch — rotating retain families (not just
   forget categories) is what actually tests it.
4. **Mind class imbalance when reading F2**: with ~19:1 negatives in the eval
   pool, a 10% control FP rate means precision ≈ .31 even at recall .96.
   Compare classifiers as ROC/TPR-at-FPR (prevalence-free) plus F2 at fixed
   removal budgets, not F2 alone.
5. Known data weak spots the new split should attack: real chat-register
   recall (Kimi chat spans: 26–47% across all budgets — the weakest cell
   everywhere), H1 diffuseness, and the c5/M1 scope question.

## 6. Round-robin design notes (sketched 2026-09-04, unimplemented)

Rotation units are *families*, not documents: forget = {AI1, AI2, AI3, H1,
(M1)}, retain = the 6–7 control families. Prefer two independent
leave-one-family-out rotations (attribution stays clean) over paired
round-robin; report a transfer matrix (held-out-family recall / FP vs the
same fold's in-distribution val), selection-stability overlap across folds,
and optionally a consensus selection. Keep the hand-labelled spans and the
sealed VALs outside every fold — they are the outer test for all arms. For
the SAE arm each fold is one recount (seconds of CPU on an existing
harvest); for the BERT arm it is one training run per fold, so the fold
count sets the compute budget.

## 7. Cluster practicalities

- SAE venv: `/projects/a5k/public/data_cwtice.a5k/python_envs/saes/.venv`
  (torch needs `env -i HOME=$HOME PATH=/usr/bin:/bin` on login nodes).
- HF cache: `HF_HOME=/projects/a5k/public/data_<user>/.cache/huggingface`;
  compute nodes are HF-offline (pre-cache), but have outbound API access.
- dataset-builder is the required path for any new/changed HF config
  (`python -m dataset_builder configs/... --build-and-push`); its CLAUDE.md
  documents the cache-hygiene rules that protect the shared environment.
