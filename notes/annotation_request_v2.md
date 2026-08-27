# Annotation request v2 — enabling the two-validation split (2026-08-09)

**STATUS UPDATE 2 (2026-08-09 18:00)**: labeler now also carries `enriched_kimi` (24 docs:
top-4 per positive-expected family by v3-keyword density — SAE-independent ranker; SAMPLING
IS ENRICHED, flag in manifest v3 and report metrics with/without) and `ideal_batch` (50:
30 agentic + 15 dense-calibration + 5 westover-general retain). claude_* still excluded.

**STATUS UPDATE (2026-08-09 17:45)**: tool export received (137 docs). Section 1 (frontier
completion) is DONE — VAL-B fully labeled. Remaining minimal residue: 48 docs = section 2
(33 chat) + section 3 (10 controls) + section 4 (5 stragglers). Export pending re-upload to
the dataset (replace assets/metagaming_labeling/annotations.json, rebuild).

Design goal: TEST + VAL-A as mirror halves of one distribution; VAL-B = fully-labeled frontier (Apollo) rollouts. Seeded picks (seed 20260810); chat/agentic/dense from rev 244b6d00, control pools from rev 33a59476. Annotators should work blind to feature-firing views.

## MINIMAL (~67 docs)

### 1. Frontier completion -> VAL-B (19)
- cotprobe:fig51_broken_promises
- cotprobe:fig51_lazy_checklist
- cotprobe:fig51_log_falsification
- cotprobe:fig51_subagent_collusion
- cotprobe:sa_oversight_chunk01
- cotprobe:sa_oversight_chunk02
- cotprobe:sa_oversight_chunk03
- cotprobe:sa_oversight_chunk04
- cotprobe:sa_oversight_chunk05
- cotprobe:sa_oversight_chunk06
- cotprobe:sa_oversight_chunk07
- cotprobe:sa_oversight_chunk08
- cotprobe:sa_oversight_chunk09
- cotprobe:sa_oversight_chunk10
- cotprobe:sa_oversight_chunk11
- cotprobe:sa_oversight_chunk12
- cotprobe:sa_oversight_chunk13
- cotprobe:sa_oversight_chunk14
- cotprobe:sa_oversight_chunk15

### 2. Chat family diversity (33; 3/family)
- kimi:dolci_if_contradictory/20006
- kimi:dolci_if_contradictory/11039
- kimi:dolci_if_contradictory/20968
- kimi:dolci_if_satisfiable/26710
- kimi:dolci_if_satisfiable/12757
- kimi:dolci_if_satisfiable/10055
- kimi:falseqa_false_premise/3615
- kimi:falseqa_false_premise/791
- kimi:falseqa_false_premise/611
- kimi:falseqa_true_premise/4363
- kimi:falseqa_true_premise/2896
- kimi:falseqa_true_premise/4349
- kimi:hawthorne_benign_fake/fraud__nemotron-super-120b-r2-mix__000
- kimi:hawthorne_benign_fake/phishing__nemotron-super-120b-r2-mix__008
- kimi:hawthorne_benign_fake/disinfo__nemotron-super-120b-r2-mix__002
- kimi:hawthorne_benign_real/fraud__nemotron-super-120b-r2-mix__003
- kimi:hawthorne_benign_real/disinfo__nemotron-super-120b-r2-mix__006
- kimi:hawthorne_benign_real/fraud__nemotron-super-120b-r2-mix__012
- kimi:hawthorne_harmful_fake/fraud__nemotron-super-120b-r2-mix__014
- kimi:hawthorne_harmful_fake/disinfo__nemotron-super-120b-r2-mix__012
- kimi:hawthorne_harmful_fake/forgery__nemotron-super-120b-r2-mix__007
- kimi:hawthorne_harmful_real/phishing__nemotron-super-120b-r2-mix__005
- kimi:hawthorne_harmful_real/harassment__nemotron-super-120b-r2-mix__014
- kimi:hawthorne_harmful_real/malware__nemotron-super-120b-r2-mix__010
- kimi:mmlu/8549
- kimi:mmlu/2269
- kimi:mmlu/7636
- kimi:squad2_answerable/572746addd62a815002e9a5e
- kimi:squad2_answerable/5733da01d058e614000b63fa
- kimi:squad2_answerable/57115ac550c2381900b54a78
- kimi:squad2_unanswerable/5ad153e9645df0001a2d1798
- kimi:squad2_unanswerable/5ad295d4d7d075001a429afd
- kimi:squad2_unanswerable/5a6ce6b54eec6b001a80a6a7

### 3. Clean-side breadth (10; review, mark clean or spans as found)
- control_ml_distractor:ml_distractor_0021
- control_ml_distractor:ml_distractor_0043
- control_ml_distractor:ml_distractor_0045
- control_ml_distractor:ml_distractor_0077
- control_ml_distractor:ml_distractor_0040
- control_human_oversight:near_miss_0055
- control_human_oversight:near_miss_0024
- control_human_oversight:near_miss_0097
- control_human_oversight:near_miss_0031
- control_human_oversight:near_miss_0034

### 4. Finish blocking todo/skip (5)
- cotprobe:ctl_human_exam
- cotprobe:ctl_rl_robotics
- pr123:metagaming_probe_docs:rl_env_0005
- pr123:metagaming_probe_docs:rl_env_0007
- pr123:metagaming_probe_docs:rl_env_0008

## IDEAL (adds ~120 docs)

### 5. Chat to 8/family (+55)
- kimi:dolci_if_contradictory/10139
- kimi:dolci_if_contradictory/24763
- kimi:dolci_if_contradictory/11228
- kimi:dolci_if_contradictory/28383
- kimi:dolci_if_contradictory/28796
- kimi:dolci_if_satisfiable/26696
- kimi:dolci_if_satisfiable/13275
- kimi:dolci_if_satisfiable/25661
- kimi:dolci_if_satisfiable/19598
- kimi:dolci_if_satisfiable/11149
- kimi:falseqa_false_premise/2792
- kimi:falseqa_false_premise/3491
- kimi:falseqa_false_premise/1074
- kimi:falseqa_false_premise/3654
- kimi:falseqa_false_premise/3381
- kimi:falseqa_true_premise/4538
- kimi:falseqa_true_premise/4514
- kimi:falseqa_true_premise/1230
- kimi:falseqa_true_premise/2214
- kimi:falseqa_true_premise/3237
- kimi:hawthorne_benign_fake/phishing__nemotron-super-120b-r2-mix__003
- kimi:hawthorne_benign_fake/fraud__nemotron-super-120b-r2-mix__005
- kimi:hawthorne_benign_fake/fraud__nemotron-super-120b-r2-mix__015
- kimi:hawthorne_benign_fake/phishing__nemotron-super-120b-r2-mix__002
- kimi:hawthorne_benign_fake/malware__nemotron-super-120b-r2-mix__007
- kimi:hawthorne_benign_real/extremism__nemotron-super-120b-r2-mix__008
- kimi:hawthorne_benign_real/malware__nemotron-super-120b-r2-mix__007
- kimi:hawthorne_benign_real/fraud__nemotron-super-120b-r2-mix__013
- kimi:hawthorne_benign_real/extremism__nemotron-super-120b-r2-mix__001
- kimi:hawthorne_benign_real/malware__nemotron-super-120b-r2-mix__008
- kimi:hawthorne_harmful_fake/malware__nemotron-super-120b-r2-mix__013
- kimi:hawthorne_harmful_fake/fraud__nemotron-super-120b-r2-mix__010
- kimi:hawthorne_harmful_fake/disinfo__nemotron-super-120b-r2-mix__008
- kimi:hawthorne_harmful_fake/fraud__nemotron-super-120b-r2-mix__003
- kimi:hawthorne_harmful_fake/privacy-violation__nemotron-super-120b-r2-mix__003
- kimi:hawthorne_harmful_real/disinfo__nemotron-super-120b-r2-mix__003
- kimi:hawthorne_harmful_real/phishing__nemotron-super-120b-r2-mix__003
- kimi:hawthorne_harmful_real/harassment__nemotron-super-120b-r2-mix__001
- kimi:hawthorne_harmful_real/disinfo__nemotron-super-120b-r2-mix__014
- kimi:hawthorne_harmful_real/malware__nemotron-super-120b-r2-mix__001
- kimi:mmlu/13743
- kimi:mmlu/13864
- kimi:mmlu/1385
- kimi:mmlu/12145
- kimi:mmlu/12646
- kimi:squad2_answerable/57281d494b864d19001644c1
- kimi:squad2_answerable/57308f6b8ab72b1400f9c581
- kimi:squad2_answerable/572847ff3acd2414000df86c
- kimi:squad2_answerable/57265e455951b619008f70bc
- kimi:squad2_answerable/572648e8dd62a815002e8076
- kimi:squad2_unanswerable/5a580d05770dc0001aeeff97
- kimi:squad2_unanswerable/5ad0213377cf76001a686b0d
- kimi:squad2_unanswerable/5ad4ca235b96ef001a10a0b8
- kimi:squad2_unanswerable/5a81f0f031013a001a334fab
- kimi:squad2_unanswerable/5ad4b0f25b96ef001a109dc0

### 6. Agentic register (30 long docs; chunk-annotate)
- kimi_agentic:agentic_misalignment_baseline/murder_swap-individualism_restriction
- kimi_agentic:agentic_misalignment_baseline/blackmail_swap-ethical_none
- kimi_agentic:agentic_misalignment_baseline/leaking_latent-individualism_restriction
- kimi_agentic:agentic_misalignment_baseline/leaking_swap-collectivism_restriction
- kimi_agentic:agentic_misalignment_baseline/blackmail_latent-america_restriction
- kimi_agentic:agentic_misalignment_baseline/leaking_swap-america_none
- kimi_agentic:agentic_misalignment_baseline/leaking_swap-global_none
- kimi_agentic:agentic_misalignment_baseline/leaking_explicit-acceleration_none
- kimi_agentic:agentic_misalignment_baseline/blackmail_latent-none_replacement
- kimi_agentic:agentic_misalignment_baseline/leaking_explicit-global_restriction
- kimi_agentic:agentic_misalignment_baseline/blackmail_latent-ethical_replacement
- kimi_agentic:agentic_misalignment_baseline/murder_latent-none_none
- kimi_agentic:agentic_misalignment_notice/leaking_explicit-acceleration_restriction
- kimi_agentic:agentic_misalignment_notice/leaking_swap-collectivism_restriction
- kimi_agentic:agentic_misalignment_notice/leaking_latent-global_replacement
- kimi_agentic:agentic_misalignment_notice/blackmail_swap-ethical_restriction
- kimi_agentic:agentic_misalignment_notice/leaking_swap-ethical_restriction
- kimi_agentic:agentic_misalignment_notice/leaking_swap-global_restriction
- kimi_agentic:agentic_misalignment_notice/murder_swap-safety_none
- kimi_agentic:agentic_misalignment_notice/leaking_swap-ethical_none
- kimi_agentic:agentic_misalignment_notice/blackmail_latent-pragmatic_none
- kimi_agentic:agentic_misalignment_notice/leaking_explicit-ethical_restriction
- kimi_agentic:agentic_misalignment_notice/murder_swap-none_restriction
- kimi_agentic:agentic_misalignment_notice/blackmail_latent-global_none
- kimi_agentic:agentic_misalignment_scratchpad/blackmail_swap-global_none
- kimi_agentic:agentic_misalignment_scratchpad/murder_explicit-america_none
- kimi_agentic:agentic_misalignment_scratchpad/blackmail_explicit-none_restriction
- kimi_agentic:agentic_misalignment_scratchpad/leaking_swap-none_restriction
- kimi_agentic:agentic_misalignment_scratchpad/blackmail_latent-pragmatic_restriction
- kimi_agentic:agentic_misalignment_scratchpad/blackmail_explicit-safety_replacement

### 7. Dense-density calibration (15)
- dense_c1_ai_evals:dense_c1_0050
- dense_c1_ai_evals:dense_c1_0001
- dense_c1_ai_evals:dense_c1_0028
- dense_c2_ai_safety:dense_c2_0029
- dense_c2_ai_safety:dense_c2_0008
- dense_c2_ai_safety:dense_c2_0040
- dense_c3_human_oversight:dense_c3_0035
- dense_c3_human_oversight:dense_c3_0005
- dense_c3_human_oversight:dense_c3_0020
- dense_c4_training_mechanics:dense_c4_0043
- dense_c4_training_mechanics:dense_c4_0009
- dense_c4_training_mechanics:dense_c4_0025
- dense_c5_swe_tests:dense_c5_0015
- dense_c5_swe_tests:dense_c5_0017
- dense_c5_swe_tests:dense_c5_0016

### 8. Retain breadth (+15)
- control_ml_distractor:ml_distractor_0029
- control_ml_distractor:ml_distractor_0044
- control_ml_distractor:ml_distractor_0069
- control_ml_distractor:ml_distractor_0093
- control_ml_distractor:ml_distractor_0014
- control_human_oversight:near_miss_0063
- control_human_oversight:near_miss_0056
- control_human_oversight:near_miss_0060
- control_human_oversight:near_miss_0049
- control_human_oversight:near_miss_0041
- westover_general:westover:post2706.txt
- westover_general:westover:post2879.txt
- westover_general:westover:post1640.txt
- westover_general:westover:post2061.txt
- westover_general:westover:post1543.txt

### 9. Tool re-export fixes (no new labeling)
- kimi:dolci_if_contradictory/4625 (word-count mismatch)
- kimi:doluschat/aviation_diversity_omission (word-count mismatch)
